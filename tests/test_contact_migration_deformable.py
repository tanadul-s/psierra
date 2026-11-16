"""Milestone 1.0 Stage D: Migration 1 Verification Suite for DeformableContact.

This module validates the staged internal migration of unsmoothed deformable
node-to-triangle contact to consume the canonical contact core:
    TriangleGeometryProvider -> NormalConstraintLaw -> ContactAssembler
behind the preserved public DeformableContact contract.

Gates Verified:
1. Canonical Core vs. Legacy Kernel Bitwise / Floating-Point Equivalence.
2. Two-body discrete action-reaction linear momentum balance (sum f_s + sum f_m = 0).
3. Exact force-potential consistency: grad_u(Pi_N) = -f_c.
4. Exact tangent consistency: K_c = -df_c/du via centered finite differences.
5. Strict non-adhesive physical activation on opening gaps (g_N > 0).
6. Multibody degree-of-freedom offsets and dynamic damping / Coulomb friction coupling.
"""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pyvista as pv
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from psierra.boundary.contact import DeformableContact
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh


class MigrationTestDomain(Domain):
    """Prescribed coordinate domain for contact verification without constitutive coupling."""

    def __init__(self, nodes: np.ndarray, faces: np.ndarray | None = None) -> None:
        self.reference = np.asarray(nodes, dtype=np.float64).copy()
        self.u = np.zeros(self.reference.size, dtype=np.float64)
        self.v = np.zeros(self.reference.size, dtype=np.float64)
        self.a = np.zeros(self.reference.size, dtype=np.float64)
        if faces is not None:
            polydata = pv.PolyData(nodes, np.column_stack((np.full(len(faces), 3), faces)).ravel())
            self.mesh = SimpleNamespace(
                surface=SurfaceMesh(polydata, node_indices=np.arange(len(nodes))),
                nodes=self.reference,
            )

    @property
    def n_dofs(self) -> int:
        return self.reference.size

    @property
    def current_nodes(self) -> np.ndarray:
        return self.reference + self.u.reshape(-1, 3)

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


def _setup_two_body_system() -> tuple[MigrationTestDomain, MigrationTestDomain]:
    """Create a 2-triangle master plate and 2 penetrating slave nodes."""
    master_nodes = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    master_faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    master = MigrationTestDomain(master_nodes, master_faces)

    slave_nodes = np.array([
        [0.8, 0.4, -0.05],  # Penetrating into triangle 0 by 0.05 mm
        [0.4, 1.2, -0.02],  # Penetrating into triangle 1 by 0.02 mm
    ])
    slave = MigrationTestDomain(slave_nodes)
    return slave, master


class TestDeformableContactMigration:
    """Rigorous gate verification for Migration 1 (Unsmoothed DeformableContact)."""

    def test_canonical_vs_legacy_penalty_frictionless(self) -> None:
        """Verify bitwise/exact floating-point agreement between canonical and legacy kernels."""
        slave, master = _setup_two_body_system()
        kp = 5000.0

        # Canonical Core
        contact_canon = DeformableContact(slave=slave, master=master, k_penalty=kp)
        contact_canon._use_legacy_kernel = False

        # Legacy Kernel
        contact_legacy = DeformableContact(slave=slave, master=master, k_penalty=kp)
        contact_legacy._use_legacy_kernel = True

        gaps_c = contact_canon.evaluate_gaps()
        gaps_l = contact_legacy.evaluate_gaps()
        np.testing.assert_allclose(gaps_c, gaps_l, atol=1e-14, rtol=1e-12)

        forces_c = contact_canon.evaluate_forces()
        forces_l = contact_legacy.evaluate_forces()
        np.testing.assert_allclose(forces_c, forces_l, atol=1e-12, rtol=1e-12)

        K_c = contact_canon.evaluate_stiffness().toarray()
        K_l = contact_legacy.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_c, K_l, atol=1e-10, rtol=1e-10)

        E_c = contact_canon.evaluate_potential_energy()
        E_l = contact_legacy.evaluate_potential_energy()
        assert E_c == pytest.approx(E_l, rel=1e-12)

        info_c = contact_canon.get_contact_info()
        info_l = contact_legacy.get_contact_info()
        assert info_c["num_active"] == info_l["num_active"] == 2
        assert info_c["max_penetration"] == pytest.approx(info_l["max_penetration"], rel=1e-12)

    def test_canonical_vs_legacy_augmented_lagrangian(self) -> None:
        """Verify Augmented Lagrangian forces and Uzawa updates match legacy kernel."""
        slave, master = _setup_two_body_system()
        kp = 8000.0

        contact_canon = DeformableContact(slave=slave, master=master, k_penalty=kp, formulation="augmented_lagrangian")
        contact_canon._use_legacy_kernel = False

        contact_legacy = DeformableContact(slave=slave, master=master, k_penalty=kp, formulation="augmented_lagrangian")
        contact_legacy._use_legacy_kernel = True

        # Assign non-zero multipliers
        multipliers = np.array([120.0, 45.0])
        contact_canon.set_multipliers(multipliers)
        contact_legacy.set_multipliers(multipliers)

        forces_c = contact_canon.evaluate_forces()
        forces_l = contact_legacy.evaluate_forces()
        np.testing.assert_allclose(forces_c, forces_l, atol=1e-12, rtol=1e-12)

        K_c = contact_canon.evaluate_stiffness().toarray()
        K_l = contact_legacy.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_c, K_l, atol=1e-10, rtol=1e-10)

        # Uzawa multiplier update
        res_c = contact_canon.update_multipliers()
        res_l = contact_legacy.update_multipliers()
        np.testing.assert_allclose(contact_canon.multipliers, contact_legacy.multipliers, atol=1e-12)
        assert res_c["max_multiplier"] == pytest.approx(res_l["max_multiplier"], rel=1e-12)

    def test_canonical_vs_legacy_dynamic_damping_and_friction(self) -> None:
        """Verify damping and Coulomb friction match between canonical and legacy kernels."""
        slave, master = _setup_two_body_system()
        kp = 6000.0
        c_damp = 15.0
        mu = 0.25

        contact_canon = DeformableContact(
            slave=slave, master=master, k_penalty=kp, c_damping=c_damp, friction=mu
        )
        contact_canon._use_legacy_kernel = False

        contact_legacy = DeformableContact(
            slave=slave, master=master, k_penalty=kp, c_damping=c_damp, friction=mu
        )
        contact_legacy._use_legacy_kernel = True

        # Velocities: slave moving downwards (penetrating deeper) and sliding laterally
        total_dofs = slave.n_dofs + master.n_dofs
        velocities = np.zeros(total_dofs)
        velocities[0] = 10.0   # vx slave 0
        velocities[2] = -5.0   # vz slave 0 (penetrating)
        velocities[3] = -8.0   # vx slave 1
        velocities[5] = -3.0   # vz slave 1 (penetrating)

        forces_c = contact_canon.evaluate_forces(velocities=velocities)
        forces_l = contact_legacy.evaluate_forces(velocities=velocities)
        np.testing.assert_allclose(forces_c, forces_l, atol=1e-10, rtol=1e-10)

        a1 = 20.0
        K_c = contact_canon.evaluate_stiffness(a1=a1).toarray()
        K_l = contact_legacy.evaluate_stiffness(a1=a1).toarray()
        np.testing.assert_allclose(K_c, K_l, atol=1e-8, rtol=1e-8)

    def test_canonical_action_reaction_linear_momentum_balance(self) -> None:
        """Verify exact discrete action-reaction balance (sum f_slave + sum f_master = 0)."""
        slave, master = _setup_two_body_system()
        s_offset = 6
        m_offset = 18

        contact = DeformableContact(
            slave=slave, master=master, k_penalty=12000.0,
            slave_offset=s_offset, master_offset=m_offset,
        )
        forces = contact.evaluate_forces(n_dofs=36)

        # Sum slave forces
        s_dofs = s_offset + np.arange(slave.n_dofs)
        m_dofs = m_offset + np.arange(master.n_dofs)

        f_slave_sum = np.sum(forces[s_dofs].reshape(-1, 3), axis=0)
        f_master_sum = np.sum(forces[m_dofs].reshape(-1, 3), axis=0)

        net_force = f_slave_sum + f_master_sum
        np.testing.assert_allclose(net_force, np.zeros(3), atol=1e-12, err_msg="Action-reaction balance violated")

    def test_canonical_energy_gradient_consistency(self) -> None:
        """Verify d(Pi)/dq matches -f_c on interior facet projection via centered finite differences."""
        slave, master = _setup_two_body_system()
        s_offset = 0
        m_offset = slave.n_dofs
        total_dofs = slave.n_dofs + master.n_dofs
        contact = DeformableContact(
            slave=slave, master=master, k_penalty=5000.0,
            slave_offset=s_offset, master_offset=m_offset,
        )

        f_c = contact.evaluate_forces(n_dofs=total_dofs)
        eps = 1e-7

        # Perturb slave 0 DOFs (dof 0: x, 1: y, 2: z)
        for dof in range(3):
            # Forward perturbation
            slave.u[dof] += eps
            E_plus = contact.evaluate_potential_energy()

            # Backward perturbation
            slave.u[dof] -= 2.0 * eps
            E_minus = contact.evaluate_potential_energy()

            # Restore
            slave.u[dof] += eps

            dE_num = (E_plus - E_minus) / (2.0 * eps)
            # Analytical force: f = -dE/du => f + dE/du = 0
            assert dE_num == pytest.approx(-f_c[s_offset + dof], rel=1e-4, abs=1e-5)

    def test_canonical_tangent_consistency_finite_differences(self) -> None:
        """Verify K_c = -df_c/du matches numerical Jacobian via centered finite differences."""
        slave, master = _setup_two_body_system()
        s_offset = 0
        m_offset = slave.n_dofs
        total_dofs = slave.n_dofs + master.n_dofs
        contact = DeformableContact(
            slave=slave, master=master, k_penalty=5000.0,
            slave_offset=s_offset, master_offset=m_offset,
        )

        K_c = contact.evaluate_stiffness(n_dofs=total_dofs).toarray()
        eps = 1e-7

        K_num = np.zeros((total_dofs, total_dofs))

        # Perturb slave nodes
        for i in range(slave.n_dofs):
            slave.u[i] += eps
            f_plus = contact.evaluate_forces(n_dofs=total_dofs)

            slave.u[i] -= 2.0 * eps
            f_minus = contact.evaluate_forces(n_dofs=total_dofs)

            slave.u[i] += eps
            K_num[:, s_offset + i] = -(f_plus - f_minus) / (2.0 * eps)

        # Perturb master nodes
        for j in range(master.n_dofs):
            master.u[j] += eps
            f_plus = contact.evaluate_forces(n_dofs=total_dofs)

            master.u[j] -= 2.0 * eps
            f_minus = contact.evaluate_forces(n_dofs=total_dofs)

            master.u[j] += eps
            K_num[:, m_offset + j] = -(f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            K_c,
            K_num,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Tangent stiffness does not match negative force Jacobian",
        )

    def test_canonical_strict_non_adhesive_separation(self) -> None:
        """Verify separating node with positive gap receives strictly zero force and zero stiffness."""
        master_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        master = MigrationTestDomain(master_nodes, np.array([[0, 1, 2]]))

        # Separated node with positive gap = +0.1 mm
        slave = MigrationTestDomain(np.array([[0.2, 0.2, 0.1]]))

        contact = DeformableContact(slave=slave, master=master, k_penalty=10000.0, tolerance=0.5)

        gaps = contact.evaluate_gaps()
        assert gaps[0] == pytest.approx(0.1, rel=1e-6)

        forces = contact.evaluate_forces()
        np.testing.assert_allclose(forces, np.zeros_like(forces), atol=1e-14, err_msg="Separated point received non-zero force")

        K = contact.evaluate_stiffness()
        assert K.nnz == 0, "Separated point received non-zero stiffness"

        info = contact.get_contact_info()
        assert info["num_active"] == 0
        assert info["potential_energy"] == 0.0

    def test_canonical_rigid_body_invariance(self) -> None:
        """Verify force transforms under 3D rigid rotation and energy is invariant."""
        slave, master = _setup_two_body_system()
        contact = DeformableContact(slave=slave, master=master, k_penalty=5000.0)

        f_ref = contact.evaluate_forces()
        E_ref = contact.evaluate_potential_energy()

        # Apply rigid rotation of 30 deg around axis (1, 1, 1) and translation (2, -1, 3)
        R = Rotation.from_rotvec(np.pi / 6.0 * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)).as_matrix()
        t = np.array([2.0, -1.0, 3.0])

        slave_rot = (slave.reference @ R.T) + t
        master_rot = (master.reference @ R.T) + t

        slave_r = MigrationTestDomain(slave_rot)
        master_r = MigrationTestDomain(master_rot, master.mesh.surface.polydata.regular_faces)
        contact_r = DeformableContact(slave=slave_r, master=master_r, k_penalty=5000.0)

        f_rot = contact_r.evaluate_forces()
        E_rot = contact_r.evaluate_potential_energy()

        assert E_rot == pytest.approx(E_ref, rel=1e-10)

        # Check forces on slave 0
        f_s0_ref = f_ref[:3]
        f_s0_rot = f_rot[:3]
        np.testing.assert_allclose(f_s0_rot, R @ f_s0_ref, atol=1e-10, rtol=1e-10)
