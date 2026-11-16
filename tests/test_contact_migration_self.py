"""Milestone 1.0 Stage D: Migration 3 Verification Suite for SelfContact.

This module validates the staged internal migration of conservative same-body
node-to-triangle self-contact to consume the canonical contact core:
    TriangleGeometryProvider -> NormalConstraintLaw -> ContactAssembler
behind the preserved public SelfContact contract.

Gates Verified:
1. Canonical Core vs. Legacy Kernel Bitwise / Floating-Point Equivalence.
2. Single-body discrete linear and angular momentum conservation (Newton's third law).
3. Exact force-potential consistency and tangent consistency via centered finite differences.
4. Topological BFS exclusion graph (exclusion_rings=0, 1, 4).
5. Area/angle-weighted normal threshold opposition filtering.
6. Multibody degree-of-freedom offsets (dof_offset > 0).
7. Penalty vs. Augmented Lagrangian formulations.
8. Dynamic normal velocity damping and algorithmic tangent.
9. Coulomb stick-slip friction coupling and momentum balance.
10. Spatial regularization with smoothed vertex normals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import numpy as np
import pyvista as pv
import pytest
import scipy.sparse as sp

from psierra.boundary.contact import SelfContact
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh


class SelfMigrationDomain(Domain):
    """Prescribed coordinate domain for self-contact migration verification."""

    def __init__(self, nodes: np.ndarray, faces: np.ndarray) -> None:
        self.reference = np.asarray(nodes, dtype=np.float64).copy()
        self.u = np.zeros(self.reference.size, dtype=np.float64)
        self.v = np.zeros(self.reference.size, dtype=np.float64)
        self.a = np.zeros(self.reference.size, dtype=np.float64)
        polydata = pv.PolyData(nodes, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        self.mesh = SimpleNamespace(
            surface=SurfaceMesh(polydata, node_indices=np.arange(len(nodes))),
            nodes=self.reference,
            n_nodes=len(nodes),
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
            self.u = np.asarray(u, dtype=np.float64).flatten()
        if v is not None:
            self.v = np.asarray(v, dtype=np.float64).flatten()
        if a is not None:
            self.a = np.asarray(a, dtype=np.float64).flatten()

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs))

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def evaluate_internal_forces(self, **kwargs: Any) -> np.ndarray:
        return np.zeros(self.n_dofs, dtype=np.float64)

    def evaluate_tangent_stiffness(self, **kwargs: Any) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs), dtype=np.float64)

    def evaluate_mass_matrix(self, **kwargs: Any) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr", dtype=np.float64)

    def evaluate_damping_matrix(self, **kwargs: Any) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs), dtype=np.float64)

    def save_state(self, filepath: Any, **kwargs: Any) -> None:
        pass


def make_opposing_patches(gap_z: float = -0.05) -> tuple[SelfMigrationDomain, np.ndarray, np.ndarray]:
    """Create two opposing triangle patches on a single domain.

    Upper patch: triangle 0 = [0, 1, 2], normal [0, 0, -1].
    Lower patch: triangle 1 = [3, 4, 5], normal [0, 0, 1].
    Node 0 penetrates into triangle 1 at z = gap_z.
    """
    nodes = np.array([
        # Upper patch (node 0 at [0.3, 0.3, gap_z])
        [0.3, 0.3, gap_z],
        [0.3, 0.6, 0.2],
        [0.6, 0.3, 0.2],
        # Lower patch (triangle in z = 0 plane)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    # Winding: [0, 1, 2] gives cross((v1-v0), (v2-v0)) = [0, 0.3, 0.2-gap] x [0.3, 0, 0.2-gap] -> -z normal
    # [3, 4, 5] gives [1, 0, 0] x [0, 1, 0] = [0, 0, 1] -> +z normal
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    domain = SelfMigrationDomain(nodes, faces)
    return domain, nodes, faces


class TestSelfContactMigration:
    """Rigorous verification suite for SelfContact migration to Contact Core."""

    def test_canonical_vs_legacy_equivalence_penalty(self) -> None:
        """Verify bitwise / floating-point equivalence between canonical core and legacy kernel."""
        domain, _, _ = make_opposing_patches(gap_z=-0.08)

        sc_canonical = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=4000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            formulation="penalty",
        )
        sc_legacy = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=4000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            formulation="penalty",
        )
        sc_legacy._use_legacy_kernel = True

        # 1. Gaps
        gaps_can = sc_canonical.evaluate_gaps()
        gaps_leg = sc_legacy.evaluate_gaps()
        np.testing.assert_allclose(gaps_can, gaps_leg, atol=1e-14)
        assert gaps_can[0] < 0.0  # Penetrating

        # 2. Forces
        f_can = sc_canonical.evaluate_forces()
        f_leg = sc_legacy.evaluate_forces()
        np.testing.assert_allclose(f_can, f_leg, atol=1e-12)
        assert np.linalg.norm(f_can) > 0.0

        # 3. Stiffness
        K_can = sc_canonical.evaluate_stiffness().toarray()
        K_leg = sc_legacy.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_can, K_leg, atol=1e-12)

        # 4. Energy and info
        E_can = sc_canonical.evaluate_potential_energy()
        E_leg = sc_legacy.evaluate_potential_energy()
        np.testing.assert_allclose(E_can, E_leg, atol=1e-12)
        assert E_can > 0.0

        info_can = sc_canonical.get_contact_info()
        info_leg = sc_legacy.get_contact_info()
        assert info_can["num_active"] == info_leg["num_active"] == 1
        np.testing.assert_allclose(info_can["max_penetration"], info_leg["max_penetration"], atol=1e-14)
        np.testing.assert_allclose(info_can["potential_energy"], info_leg["potential_energy"], atol=1e-12)

    def test_canonical_vs_legacy_equivalence_augmented_lagrangian(self) -> None:
        """Verify equivalence under Augmented Lagrangian formulation with multipliers."""
        domain, _, _ = make_opposing_patches(gap_z=-0.04)

        sc_canonical = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=2500.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            formulation="augmented_lagrangian",
        )
        sc_legacy = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=2500.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            formulation="augmented_lagrangian",
        )
        sc_legacy._use_legacy_kernel = True

        sc_canonical.multipliers[0] = 75.0
        sc_legacy.multipliers[0] = 75.0

        f_can = sc_canonical.evaluate_forces()
        f_leg = sc_legacy.evaluate_forces()
        np.testing.assert_allclose(f_can, f_leg, atol=1e-12)

        K_can = sc_canonical.evaluate_stiffness().toarray()
        K_leg = sc_legacy.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_can, K_leg, atol=1e-12)

        E_can = sc_canonical.evaluate_potential_energy()
        E_leg = sc_legacy.evaluate_potential_energy()
        np.testing.assert_allclose(E_can, E_leg, atol=1e-12)

    def test_finite_difference_tangent_consistency(self) -> None:
        """Verify exact tangent consistency K_c = -df_c/du via centered finite differences."""
        domain, _, _ = make_opposing_patches(gap_z=-0.06)

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=5000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
        )

        K_exact = sc.evaluate_stiffness().toarray()
        f0 = sc.evaluate_forces()

        # Numerically differentiate forces w.r.t displacement of active nodes [0, 3, 4, 5]
        eps = 1e-6
        K_fd = np.zeros_like(K_exact)
        active_dofs = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17]

        u_orig = domain.u.copy()
        for dof in active_dofs:
            u_plus = u_orig.copy()
            u_plus[dof] += eps
            domain.update_state(u=u_plus)
            f_plus = sc.evaluate_forces()

            u_minus = u_orig.copy()
            u_minus[dof] -= eps
            domain.update_state(u=u_minus)
            f_minus = sc.evaluate_forces()

            K_fd[:, dof] = -(f_plus - f_minus) / (2.0 * eps)

        domain.update_state(u=u_orig)

        # Compare K_exact and K_fd on active DOFs
        np.testing.assert_allclose(
            K_exact[np.ix_(active_dofs, active_dofs)],
            K_fd[np.ix_(active_dofs, active_dofs)],
            rtol=1e-4,
            atol=1e-4,
        )
        # Symmetry check
        np.testing.assert_allclose(K_exact, K_exact.T, atol=1e-12)

    def test_single_body_linear_momentum_conservation(self) -> None:
        """Verify Newton's third law: total internal self-contact force sums to zero."""
        domain, _, _ = make_opposing_patches(gap_z=-0.07)

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=6000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
        )

        forces = sc.evaluate_forces().reshape(-1, 3)
        total_force = np.sum(forces, axis=0)

        # Force magnitude on slave node
        f_slave = forces[0]
        assert np.linalg.norm(f_slave) > 100.0

        # Total force must be exactly zero (conservation of linear momentum)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-12)

    def test_single_body_angular_momentum_conservation(self) -> None:
        """Verify rotational invariance: total internal contact moment sums to zero."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=4500.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
        )

        forces = sc.evaluate_forces().reshape(-1, 3)
        coords = domain.current_nodes

        total_moment = np.zeros(3)
        for x, f in zip(coords, forces):
            total_moment += np.cross(x, f)

        # Total moment must be identically zero
        np.testing.assert_allclose(total_moment, 0.0, atol=1e-12)

    def test_topological_bfs_exclusion_rings(self) -> None:
        """Verify that BFS exclusion graph suppresses self-contact across incident & ring facets."""
        # 4-triangle strip:
        # T0: [0, 1, 2], T1: [1, 3, 2], T2: [3, 4, 2], T3: [4, 5, 2]
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
            [4, 5, 2],
        ], dtype=np.int64)
        domain = SelfMigrationDomain(nodes, faces)

        # exclusion_rings=0 excludes only incident faces
        sc_ring0 = SelfContact(domain, exclusion_rings=0)
        # Node 0 is in face 0 only
        assert sc_ring0._excluded_faces[0] == {0}

        # exclusion_rings=1: node 0 is connected to nodes 1, 2.
        # Nodes 1, 2 are in faces 0, 1, 2.
        sc_ring1 = SelfContact(domain, exclusion_rings=1)
        assert {0, 1, 2}.issubset(sc_ring1._excluded_faces[0])

        # exclusion_rings=4: excludes all connected faces in small strip
        sc_ring4 = SelfContact(domain, exclusion_rings=4)
        assert sc_ring4._excluded_faces[0] == {0, 1, 2, 3}

    def test_normal_threshold_opposition_filtering(self) -> None:
        """Verify that candidates with slave/master normals not opposing are filtered out."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)

        # Normal case: opposing normals -> active
        sc_opposing = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=3000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            normal_threshold=-0.2,
        )
        f_opposing = sc_opposing.evaluate_forces()
        assert np.linalg.norm(f_opposing) > 0.0

        # Strict normal threshold requiring near-antiparallel normals (dot < -0.999)
        # Here slave normal dot face normal is around -0.83 < -0.2, but > -0.999
        sc_strict = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=3000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            normal_threshold=-0.999,
        )
        f_strict = sc_strict.evaluate_forces()
        # Filtered out by strict threshold
        np.testing.assert_allclose(f_strict, 0.0, atol=1e-12)

    def test_multibody_dof_offset_scattering(self) -> None:
        """Verify that dof_offset scatters forces and stiffness strictly into the body DOF range."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)
        dof_offset = 15
        total_dofs = dof_offset + domain.n_dofs + 9

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=4000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
            dof_offset=dof_offset,
        )

        forces = sc.evaluate_forces(n_dofs=total_dofs)
        # Prior to dof_offset
        np.testing.assert_allclose(forces[:dof_offset], 0.0, atol=1e-12)
        # After domain DOFs
        np.testing.assert_allclose(forces[dof_offset + domain.n_dofs:], 0.0, atol=1e-12)
        # Active inside body range
        assert np.linalg.norm(forces[dof_offset : dof_offset + domain.n_dofs]) > 0.0

        K_c = sc.evaluate_stiffness(n_dofs=total_dofs).toarray()
        np.testing.assert_allclose(K_c[:dof_offset, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(K_c[:, :dof_offset], 0.0, atol=1e-12)
        np.testing.assert_allclose(K_c[dof_offset + domain.n_dofs:, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(K_c[:, dof_offset + domain.n_dofs:], 0.0, atol=1e-12)

    def test_dynamic_velocity_damping_and_algorithmic_tangent(self) -> None:
        """Verify normal velocity damping forces and algorithmic stiffness scaling."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=4000.0,
            c_damping=100.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
        )

        vel = np.zeros(domain.n_dofs)
        vel[2] = -2.0  # Node 0 moving deeper into penetration (v_z = -2.0)

        f_undamped = sc.evaluate_forces()
        f_damped = sc.evaluate_forces(velocities=vel)

        # Damping opposes penetration velocity (+z direction on node 0)
        assert f_damped[2] > f_undamped[2]

        # Algorithmic tangent test with a1 > 0
        a1 = 25.0
        K_undamped = sc.evaluate_stiffness(a1=0.0).toarray()
        K_alg = sc.evaluate_stiffness(a1=a1).toarray()

        diff = K_alg - K_undamped
        # diff should equal a1 * c_damping * outer(gradient, gradient)
        queries = sc._find_active_contacts()
        assert len(queries) == 1
        grad = queries[0].geometry.gradient
        indices = sc._dof_indices(queries[0])
        expected_diff = np.zeros_like(K_alg)
        expected_diff[np.ix_(indices, indices)] = a1 * sc.c_damping * np.outer(grad, grad)

        np.testing.assert_allclose(diff, expected_diff, atol=1e-10)

    def test_coulomb_friction_coupling_and_momentum_balance(self) -> None:
        """Verify Coulomb friction stick-slip state transition and action-reaction linear balance."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)

        sc = SelfContact(
            domain,
            contact_nodes=[0],
            k_penalty=5000.0,
            friction=0.3,
            k_tangential=2000.0,
            search_radius=3.0,
            contact_thickness=1.5,
            exclusion_rings=0,
        )

        # 1. Initial evaluation at zero tangential shift
        f0 = sc.evaluate_forces()
        assert sc.friction_state.states[0] == 1  # STICK with zero tangential force
        np.testing.assert_allclose(sc.friction_state.forces[0], 0.0, atol=1e-12)

        # 2. Apply small tangential relative shift (Stick regime: k_T * delta = 2000 * 0.002 = 4 < mu * p_N = 75)
        sc.friction_state.anchors[0] = np.array([0.002, 0.0, -0.05])
        f_stick = sc.evaluate_forces().reshape(-1, 3)
        assert sc.friction_state.states[0] == 1  # STICK
        np.testing.assert_allclose(f_stick[0, 0], 4.0, atol=1e-12)

        # Total internal force (normal + frictional) must strictly sum to zero
        np.testing.assert_allclose(np.sum(f_stick, axis=0), 0.0, atol=1e-12)

        # 3. Apply large tangential relative shift (Slip regime: k_T * delta = 400 > mu * p_N = 75)
        sc.friction_state.anchors[0] = np.array([0.20, 0.0, -0.05])
        f_slip = sc.evaluate_forces().reshape(-1, 3)
        assert sc.friction_state.states[0] == 2  # SLIP
        # Slip force magnitude saturates at mu * p_N = 0.3 * 250 = 75
        p_N = 5000.0 * 0.05
        np.testing.assert_allclose(f_slip[0, 0], 0.3 * p_N, atol=1e-12)

        # Internal linear momentum conserved even in slip regime
        np.testing.assert_allclose(np.sum(f_slip, axis=0), 0.0, atol=1e-12)

    def test_smoothed_normals_support(self) -> None:
        """Verify SelfContact with smoothed vertex normals evaluates cleanly and conserves momentum."""
        domain, _, _ = make_opposing_patches(gap_z=-0.05)

        for weight in ("angle", "area"):
            sc = SelfContact(
                domain,
                contact_nodes=[0],
                k_penalty=5000.0,
                search_radius=3.0,
                contact_thickness=1.5,
                exclusion_rings=0,
                smoothed_normals=True,
                smoothing_weight=weight,
            )

            forces = sc.evaluate_forces().reshape(-1, 3)
            assert np.linalg.norm(forces[0]) > 0.0
            np.testing.assert_allclose(np.sum(forces, axis=0), 0.0, atol=1e-12)

            K = sc.evaluate_stiffness().toarray()
            assert K.shape == (domain.n_dofs, domain.n_dofs)
            np.testing.assert_allclose(K, K.T, atol=1e-12)
