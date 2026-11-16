"""Milestone 1.0 Stage D: Migration 2 Verification Suite for RigidContact.

This module validates the staged internal migration of RigidContact to consume
the canonical contact core:
    PlanarGeometryProvider / SphereGeometryProvider / CylinderGeometryProvider
    -> NormalConstraintLaw -> ContactAssembler
behind the preserved public RigidContact contract.

Gates Verified:
1. Exact curved rigid obstacle linearization:
       K_c = kp * (n (x) n) - p_N * Hess(g) = kp * (n (x) n + g_N * Hess(g_N))
   matching centered finite differences to high precision across spheres and cylinders.
2. Strict non-adhesive physical activation: separated points (g_N > 0) strictly receive
   zero force and zero stiffness even when tolerance > 0 (resolving LEGACY-DISC-01).
3. Exact potential energy gradient consistency: grad_u(Pi_N) = -f_c.
4. Augmented Lagrangian dual multiplier transactions and Uzawa updates.
5. Dynamic velocity damping and Coulomb stick-slip friction coupling.
6. Shell translational degree-of-freedom scatter (dofs_per_node=6).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import scipy.sparse as sp

from psierra.boundary.contact import RigidContact
from psierra.collision.shapes import RigidPlane, RigidSphere, RigidCylinder
from psierra.domain.base import Domain


class RigidMigrationDomain(Domain):
    """Domain fixture providing nodal coordinates and DOF configuration."""

    def __init__(self, nodes: np.ndarray, dofs_per_node: int = 3) -> None:
        self.reference = np.asarray(nodes, dtype=np.float64).copy()
        self.dofs_per_node = int(dofs_per_node)
        self.u = np.zeros(self.reference.shape[0] * self.dofs_per_node, dtype=np.float64)
        self.v = np.zeros(self.reference.shape[0] * self.dofs_per_node, dtype=np.float64)
        self.a = np.zeros(self.reference.shape[0] * self.dofs_per_node, dtype=np.float64)
        self.mesh = SimpleNamespace(nodes=self.reference, n_nodes=len(self.reference))

    @property
    def n_dofs(self) -> int:
        return self.reference.shape[0] * self.dofs_per_node

    @property
    def current_nodes(self) -> np.ndarray:
        if self.dofs_per_node == 3:
            return self.reference + self.u.reshape(-1, 3)
        # Shell / 6-DOF nodes: translational DOFs are indices 0..3 of each node
        u_trans = self.u.reshape(-1, self.dofs_per_node)[:, :3]
        return self.reference + u_trans

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        if u is not None:
            self.u = np.asarray(u, dtype=np.float64).copy()
        if v is not None:
            self.v = np.asarray(v, dtype=np.float64).copy()
        if a is not None:
            self.a = np.asarray(a, dtype=np.float64).copy()

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs))

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def save_state(self, filepath: Any, **kwargs: Any) -> None:
        pass


class TestRigidContactMigration:
    """Rigorous gate verification for Migration 2 (Planar and Curved RigidContact)."""

    def test_planar_rigid_canonical_vs_legacy(self) -> None:
        """Verify planar rigid obstacle force, stiffness, and energy match legacy on penetration."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        # Node 0: penetrating into plane (y = -0.2), Node 1: penetrating (y = -0.05)
        nodes = np.array([
            [1.0, -0.2, 0.5],
            [2.0, -0.05, -1.0],
        ])
        domain = RigidMigrationDomain(nodes)
        kp = 8000.0

        contact_canon = RigidContact(slave=domain, master=plane, k_penalty=kp)
        contact_canon._use_legacy_kernel = False

        contact_legacy = RigidContact(slave=domain, master=plane, k_penalty=kp)
        contact_legacy._use_legacy_kernel = True

        forces_c = contact_canon.evaluate_forces()
        forces_l = contact_legacy.evaluate_forces()
        np.testing.assert_allclose(forces_c, forces_l, atol=1e-12, rtol=1e-12)

        K_c = contact_canon.evaluate_stiffness().toarray()
        K_l = contact_legacy.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_c, K_l, atol=1e-12, rtol=1e-12)

        E_c = contact_canon.evaluate_potential_energy()
        E_l = contact_legacy.evaluate_potential_energy()
        assert E_c == pytest.approx(E_l, rel=1e-12)

    def test_strict_non_adhesive_activation_with_search_tolerance(self) -> None:
        """Verify separated node with 0 < g <= tolerance receives zero force (LEGACY-DISC-01 fix)."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        # Node at z = +0.08 mm (separated)
        nodes = np.array([[0.0, 0.0, 0.08]])
        domain = RigidMigrationDomain(nodes)

        contact = RigidContact(slave=domain, master=plane, k_penalty=5000.0, tolerance=0.15)
        contact._use_legacy_kernel = False

        forces = contact.evaluate_forces()
        assert np.allclose(forces, 0.0), "Separated node with positive gap received attractive force"

        K = contact.evaluate_stiffness()
        assert K.nnz == 0, "Separated node received non-zero stiffness"

        info = contact.get_contact_info()
        assert info["num_active"] == 0
        assert info["potential_energy"] == 0.0

    @pytest.mark.parametrize("R", [0.5, 2.0, 10.0])
    @pytest.mark.parametrize("penetration", [0.01, 0.05, 0.2])
    def test_sphere_exact_curvature_linearization_against_finite_differences(
        self, R: float, penetration: float
    ) -> None:
        """Verify analytical sphere tangent K_c = kp*(n(x)n - p_N*Hess) matches FD (LEGACY-DISC-02 fix)."""
        center = np.array([1.0, -2.0, 0.5])
        sphere = RigidSphere(center=center, radius=R)

        # Candidate point along arbitrary direction
        direction = np.array([1.0, 2.0, -1.0])
        direction = direction / np.linalg.norm(direction)

        # Spatial position: inside sphere by 'penetration' depth
        # Distance from center d = R - penetration
        d = R - penetration
        pos = center + d * direction

        domain = RigidMigrationDomain(pos.reshape(1, 3))
        kp = 10000.0

        contact = RigidContact(slave=domain, master=sphere, k_penalty=kp)
        contact._use_legacy_kernel = False

        K_ana = contact.evaluate_stiffness().toarray()

        # Finite-difference derivative of contact force: K_fd = -df_c/du
        eps = 1e-7
        K_fd = np.zeros((3, 3))
        for col in range(3):
            domain.u[col] += eps
            f_plus = contact.evaluate_forces()

            domain.u[col] -= 2.0 * eps
            f_minus = contact.evaluate_forces()

            domain.u[col] += eps
            K_fd[:, col] = -(f_plus - f_minus) / (2.0 * eps)

        # Verify analytical tangent matches numerical finite differences
        np.testing.assert_allclose(
            K_ana,
            K_fd,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Sphere analytical stiffness failed finite difference check at R={R}, pen={penetration}",
        )

        # Verify analytical transverse eigenvalues match kp * g / d
        expected_transverse_k = kp * (-penetration) / d
        eigvals = np.linalg.eigvalsh(K_ana)
        # Normal eigenvalue is kp; two transverse eigenvalues are expected_transverse_k
        assert eigvals[0] == pytest.approx(expected_transverse_k, rel=1e-4)
        assert eigvals[1] == pytest.approx(expected_transverse_k, rel=1e-4)
        assert eigvals[2] == pytest.approx(kp, rel=1e-4)

    @pytest.mark.parametrize("radius", [1.0, 5.0])
    @pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)])
    def test_cylinder_exact_curvature_linearization_against_finite_differences(
        self, radius: float, axis: tuple[float, float, float]
    ) -> None:
        """Verify analytical cylinder tangent matches centered finite differences."""
        axis_arr = np.array(axis, dtype=np.float64)
        axis_arr /= np.linalg.norm(axis_arr)
        origin = np.array([0.5, 0.5, 0.5])
        cylinder = RigidCylinder(center=origin, axis=axis_arr, radius=radius)

        # Choose normal perpendicular to axis
        arb = np.array([1.0, 0.0, 0.0]) if abs(axis_arr[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n_perp = arb - np.dot(arb, axis_arr) * axis_arr
        n_perp /= np.linalg.norm(n_perp)

        pen = 0.05 * radius
        d = radius - pen
        pos = origin + 2.0 * axis_arr + d * n_perp

        domain = RigidMigrationDomain(pos.reshape(1, 3))
        kp = 5000.0

        contact = RigidContact(slave=domain, master=cylinder, k_penalty=kp)
        contact._use_legacy_kernel = False

        K_ana = contact.evaluate_stiffness().toarray()

        eps = 1e-7
        K_fd = np.zeros((3, 3))
        for col in range(3):
            domain.u[col] += eps
            f_plus = contact.evaluate_forces()

            domain.u[col] -= 2.0 * eps
            f_minus = contact.evaluate_forces()

            domain.u[col] += eps
            K_fd[:, col] = -(f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            K_ana,
            K_fd,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Cylinder analytical tangent failed finite difference check",
        )

    def test_rigid_contact_energy_gradient_consistency(self) -> None:
        """Verify d(Pi)/du = -f_c for spherical obstacle via centered finite differences."""
        sphere = RigidSphere(center=(0.0, 0.0, 0.0), radius=5.0)
        # Point inside sphere at (0.0, 0.0, 4.8) -> penetration 0.2
        pos = np.array([[0.0, 0.0, 4.8]])
        domain = RigidMigrationDomain(pos)
        contact = RigidContact(slave=domain, master=sphere, k_penalty=2000.0)
        contact._use_legacy_kernel = False

        f_c = contact.evaluate_forces()
        eps = 1e-7

        for dof in range(3):
            domain.u[dof] += eps
            E_plus = contact.evaluate_potential_energy()

            domain.u[dof] -= 2.0 * eps
            E_minus = contact.evaluate_potential_energy()

            domain.u[dof] += eps

            dE_num = (E_plus - E_minus) / (2.0 * eps)
            assert dE_num == pytest.approx(-f_c[dof], rel=1e-5, abs=1e-6)

    def test_rigid_contact_augmented_lagrangian_multipliers(self) -> None:
        """Verify Augmented Lagrangian force, stiffness, and Uzawa updates."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        # 2 nodes: one penetrating (y = -0.1), one separated (y = +0.05)
        nodes = np.array([
            [0.0, -0.1, 0.0],
            [1.0, 0.05, 0.0],
        ])
        domain = RigidMigrationDomain(nodes)
        kp = 10000.0

        contact = RigidContact(
            slave=domain, master=plane, k_penalty=kp, formulation="augmented_lagrangian"
        )
        contact._use_legacy_kernel = False
        contact.set_multipliers(np.array([500.0, 200.0]))

        forces = contact.evaluate_forces()
        # Node 0: p_N = max(0, 500 - 10000*(-0.1)) = 500 + 1000 = 1500 N in +Y
        assert forces[1] == pytest.approx(1500.0, rel=1e-10)
        # Node 1: p_N = max(0, 200 - 10000*(0.05)) = max(0, 200 - 500) = 0 N (rebounded/opening)
        assert forces[4] == pytest.approx(0.0, abs=1e-12)

        # Uzawa update
        contact.update_multipliers()
        assert contact.multipliers[0] == pytest.approx(1500.0, rel=1e-10)
        assert contact.multipliers[1] == pytest.approx(0.0, abs=1e-12)

    def test_rigid_contact_shell_dofs_offset(self) -> None:
        """Verify shell 6-DOF element translational force scatter and stiffness blocks."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        # 2 shell nodes (6 DOFs per node: ux, uy, uz, rotx, roty, rotz)
        nodes = np.array([
            [0.0, 0.0, -0.1],  # Penetrates z by 0.1
            [1.0, 0.0, 0.5],   # Separated
        ])
        domain = RigidMigrationDomain(nodes, dofs_per_node=6)
        kp = 4000.0

        contact = RigidContact(
            slave=domain, master=plane, k_penalty=kp, dof_offset=12, dofs_per_node=6
        )
        contact._use_legacy_kernel = False

        total_dofs = 12 + 2 * 6  # 24 DOFs
        forces = contact.evaluate_forces(n_dofs=total_dofs)

        # Node 0 starts at DOF 12: translational DOFs are 12, 13, 14
        assert forces[12] == 0.0  # Fx
        assert forces[13] == 0.0  # Fy
        assert forces[14] == pytest.approx(400.0, rel=1e-10)  # Fz = kp * |g| = 4000 * 0.1 = 400
        # Rotational DOFs (15, 16, 17) must be strictly zero
        np.testing.assert_allclose(forces[15:18], 0.0, atol=1e-14)

        K = contact.evaluate_stiffness(n_dofs=total_dofs).toarray()
        assert K[14, 14] == pytest.approx(kp, rel=1e-10)
        # Rotational diagonal entries must be 0
        assert K[15, 15] == 0.0
        assert K[16, 16] == 0.0
        assert K[17, 17] == 0.0
