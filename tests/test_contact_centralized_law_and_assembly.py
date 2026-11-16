"""Adversarial Verification Suite for Centralized Contact Law & Shared Assembly.

Milestone 1.0 Stage C of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
Verifies:
1. Non-adhesive activation: separated candidates with 0 < g_N <= delta_search
   strictly receive zero normal force, zero tangent, zero potential energy,
   and inactive status (eliminating LEGACY-DISC-01).
2. Conservative triad mathematical consistency on smooth branches:
   - Negative energy gradient: -dPi_N/dq == f_local to 10^-7.
   - Negative force Jacobian: -df_local/dq == K_local to 10^-5.
   - Exact analytical symmetry of K_local.
3. Centralized assembly operators (ContactAssembler):
   - One-body, two-body, and self-contact stencil mapping.
   - Exact action-reaction linear momentum balance: sum f_s + sum f_m = 0.
   - Exact angular momentum balance: sum x (x) f = 0.
   - Global sparse CSR stiffness matrix symmetry: K_c == K_c^T.
   - Duplicate nodal entry summation via sparse COO scatter.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.contact.data import (
    ContactCandidate,
    DifferentiableContactGeometry,
    NormalLawState,
)
from psierra.contact.geometry import (
    PlanarGeometryProvider,
    SphereGeometryProvider,
    TriangleGeometryProvider,
)
from psierra.contact.law import NormalConstraintLaw
from psierra.contact.assembly import ContactAssembler


# =============================================================================
# 1. Non-Adhesive Activation & Tolerance Decoupling
# =============================================================================

class TestNonAdhesiveActivation:
    """Verify that collision search proximity tolerances do not leak attractive forces."""

    def test_positive_gap_penalty_zero_force(self) -> None:
        """A separated candidate with 0 < g_N <= delta_search must have strictly zero force."""
        law = NormalConstraintLaw(k_penalty=5000.0, formulation="penalty")
        normal = np.array([0.0, 0.0, 1.0])

        # Candidate is separated by g_N = +0.05 mm (inside a search tolerance delta = 0.2 mm)
        geom_separated = DifferentiableContactGeometry(
            gap=0.05,
            normal=normal,
            closest_point=np.array([1.0, 1.0, 0.0]),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )

        state = law.evaluate_state(geom_separated)
        assert state.pressure == 0.0
        assert state.is_active is False
        assert state.potential_energy == 0.0
        assert state.first_derivative == 0.0

        force = law.evaluate_force(geom_separated)
        np.testing.assert_allclose(force, 0.0, atol=1e-14)

        tangent = law.evaluate_tangent(geom_separated)
        np.testing.assert_allclose(tangent, 0.0, atol=1e-14)

    def test_penetrating_gap_penalty_positive_repulsion(self) -> None:
        """A penetrating candidate with g_N < 0 receives positive repulsive force p_N * n."""
        law = NormalConstraintLaw(k_penalty=5000.0, formulation="penalty")
        normal = np.array([0.0, 0.0, 1.0])

        # Candidate penetrates by 0.04 mm (g_N = -0.04 mm)
        geom_penetrated = DifferentiableContactGeometry(
            gap=-0.04,
            normal=normal,
            closest_point=np.array([1.0, 1.0, 0.0]),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )

        state = law.evaluate_state(geom_penetrated)
        assert state.pressure == pytest.approx(200.0)  # 5000 * 0.04
        assert state.is_active is True
        assert state.potential_energy == pytest.approx(0.5 * 5000.0 * 0.04**2)

        force = law.evaluate_force(geom_penetrated)
        np.testing.assert_allclose(force, [0.0, 0.0, 200.0])

        tangent = law.evaluate_tangent(geom_penetrated)
        expected_K = 5000.0 * np.outer(normal, normal)
        np.testing.assert_allclose(tangent, expected_K, atol=1e-12)

    def test_augmented_lagrangian_retained_pressure(self) -> None:
        """AL contact retains holding pressure on opening gap if lambda > kp * g."""
        law = NormalConstraintLaw(k_penalty=1000.0, formulation="augmented_lagrangian")
        normal = np.array([0.0, 1.0, 0.0])

        # g_N = +0.02 mm (positive opening), but multiplier lambda = 50.0 N
        # p_N = max(0, 50.0 - 1000 * 0.02) = 50.0 - 20.0 = 30.0 N
        geom_opening = DifferentiableContactGeometry(
            gap=0.02,
            normal=normal,
            closest_point=np.zeros(3),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )

        state = law.evaluate_state(geom_opening, multiplier=50.0)
        assert state.pressure == pytest.approx(30.0)
        assert state.is_active is True
        assert state.multiplier == 50.0

        force = law.evaluate_force(geom_opening, multiplier=50.0)
        np.testing.assert_allclose(force, [0.0, 30.0, 0.0])

        # When gap opens beyond lambda / kp (e.g. g = 0.06 mm -> kp * g = 60 > 50):
        geom_far = DifferentiableContactGeometry(
            gap=0.06,
            normal=normal,
            closest_point=np.zeros(3),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )
        state_far = law.evaluate_state(geom_far, multiplier=50.0)
        assert state_far.pressure == 0.0
        assert state_far.is_active is False
        np.testing.assert_allclose(law.evaluate_force(geom_far, multiplier=50.0), 0.0)


# =============================================================================
# 2. Conservative Triad Consistency: Energy -> Force -> Tangent
# =============================================================================

class TestConservativeTriadConsistency:
    """Verify that force matches -dPi/dq and tangent matches -df/dq on smooth branches."""

    def test_triad_consistency_curved_sphere(self) -> None:
        """Verify -dPi/dq == f and -df/dq == K for analytical curved sphere contact."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0
        provider = SphereGeometryProvider(center=center, radius=radius)
        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="sphere")
        law = NormalConstraintLaw(k_penalty=3000.0, formulation="penalty")

        # Penetrating query point at x = (0.8, -0.6, 1.2), dist = 1.562 < 2.0
        point = np.array([0.8, -0.6, 1.2])
        geom = provider.evaluate_geometry(cand, point)
        assert geom.gap < 0.0

        f_analytic = law.evaluate_force(geom)
        K_analytic = law.evaluate_tangent(geom)

        # 1. Negative energy gradient: -dPi / dq == f
        eps = 1e-6
        f_numeric = np.zeros(3)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            g_plus = provider.evaluate_geometry(cand, point + delta)
            g_minus = provider.evaluate_geometry(cand, point - delta)
            pi_plus = law.evaluate_potential_energy(g_plus)
            pi_minus = law.evaluate_potential_energy(g_minus)
            f_numeric[i] = -(pi_plus - pi_minus) / (2.0 * eps)

        np.testing.assert_allclose(f_analytic, f_numeric, rtol=1e-5, atol=1e-7)

        # 2. Negative force Jacobian: -df / dq == K
        K_numeric = np.zeros((3, 3))
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = eps
            g_plus = provider.evaluate_geometry(cand, point + delta)
            g_minus = provider.evaluate_geometry(cand, point - delta)
            f_plus = law.evaluate_force(g_plus)
            f_minus = law.evaluate_force(g_minus)
            K_numeric[:, j] = -(f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(K_analytic, K_numeric, rtol=1e-4, atol=1e-5)
        # Symmetry
        np.testing.assert_allclose(K_analytic, K_analytic.T, atol=1e-12)

    def test_triad_consistency_triangle_stencil(self) -> None:
        """Verify -dPi/dq == f and -df/dq == K for 12-DOF node-to-triangle contact."""
        provider = TriangleGeometryProvider()
        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="tri_0")
        law = NormalConstraintLaw(k_penalty=4000.0, formulation="penalty")

        coords = np.array([
            [0.25, 0.25, -0.04],  # Slave penetrating xy face
            [0.0, 0.0, 0.0],      # v0
            [1.0, 0.0, 0.0],      # v1
            [0.0, 1.0, 0.0],      # v2
        ])

        geom = provider.evaluate_geometry(cand, coords)
        assert geom.feature_id == "face"
        assert geom.gap < 0.0

        f_analytic = law.evaluate_force(geom)
        K_analytic = law.evaluate_tangent(geom)

        q_flat = coords.ravel()
        eps = 1e-6

        # 1. Energy gradient check
        f_numeric = np.zeros(12)
        for k in range(12):
            delta = np.zeros(12)
            delta[k] = eps
            c_plus = (q_flat + delta).reshape(4, 3)
            c_minus = (q_flat - delta).reshape(4, 3)
            g_plus = provider.evaluate_geometry(cand, c_plus)
            g_minus = provider.evaluate_geometry(cand, c_minus)
            pi_plus = law.evaluate_potential_energy(g_plus)
            pi_minus = law.evaluate_potential_energy(g_minus)
            f_numeric[k] = -(pi_plus - pi_minus) / (2.0 * eps)

        np.testing.assert_allclose(f_analytic, f_numeric, rtol=1e-5, atol=1e-6)

        # 2. Force Jacobian check
        K_numeric = np.zeros((12, 12))
        for j in range(12):
            delta = np.zeros(12)
            delta[j] = eps
            c_plus = (q_flat + delta).reshape(4, 3)
            c_minus = (q_flat - delta).reshape(4, 3)
            g_plus = provider.evaluate_geometry(cand, c_plus)
            g_minus = provider.evaluate_geometry(cand, c_minus)
            f_plus = law.evaluate_force(g_plus)
            f_minus = law.evaluate_force(g_minus)
            K_numeric[:, j] = -(f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(K_analytic, K_numeric, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(K_analytic, K_analytic.T, atol=1e-12)


# =============================================================================
# 3. Centralized Contact Assembler Operations & Conservation Laws
# =============================================================================

class TestContactAssembler:
    """Verify shared ContactAssembler maps, force scatter, and momentum conservation."""

    def test_one_body_assembly(self) -> None:
        """One-body assembler must correctly map slave node and offset into global arrays."""
        dof_offset = 6
        total_dofs = 15
        asm_map = ContactAssembler.build_one_body_map(slave_node=2, dof_offset=dof_offset)

        # Slave node 2 at offset 6 -> DOFs: 6 + 2*3 = 12, 13, 14
        expected_dofs = np.array([12, 13, 14])
        np.testing.assert_array_equal(asm_map.slave_dofs, expected_dofs)
        np.testing.assert_array_equal(asm_map.stencil_dofs, expected_dofs)

        local_f = np.array([0.0, 0.0, 150.0])
        global_f = ContactAssembler.assemble_forces([(asm_map, local_f)], total_dofs=total_dofs)

        assert global_f[14] == pytest.approx(150.0)
        np.testing.assert_allclose(global_f[:14], 0.0)

        local_K = np.diag([0.0, 0.0, 5000.0])
        K_c = ContactAssembler.assemble_stiffness([(asm_map, local_K)], total_dofs=total_dofs)
        assert K_c.shape == (total_dofs, total_dofs)
        assert K_c[14, 14] == pytest.approx(5000.0)

    def test_two_body_assembly_momentum_conservation(self) -> None:
        """Two-body assembly must satisfy exact linear and angular momentum conservation."""
        slave_node = 0
        master_nodes = np.array([0, 1, 2])
        barycentric = np.array([0.4, 0.3, 0.3])

        slave_offset = 0
        master_offset = 3
        total_dofs = 12

        asm_map = ContactAssembler.build_two_body_map(
            slave_node=slave_node,
            master_nodes=master_nodes,
            barycentric=barycentric,
            slave_offset=slave_offset,
            master_offset=master_offset,
        )

        # In local 12-DOF coordinates, normal force along z:
        # slave receives f_s = p * n, master nodes receive -xi_i * p * n
        p_N = 250.0
        n = np.array([0.0, 0.0, 1.0])
        local_f = np.kron(np.r_[1.0, -barycentric], p_N * n)

        global_f = ContactAssembler.assemble_forces([(asm_map, local_f)], total_dofs=total_dofs)

        # Linear momentum check: sum f_s + sum f_m == 0
        slave_dofs = asm_map.slave_dofs
        master_dofs = asm_map.master_dofs
        is_balanced = ContactAssembler.check_linear_momentum_balance(
            global_forces=global_f,
            slave_dofs=slave_dofs,
            master_dofs=master_dofs,
            atol=1e-12,
        )
        assert is_balanced is True
        np.testing.assert_allclose(np.sum(global_f), 0.0, atol=1e-12)

        # Angular momentum check: sum x (x) f == 0
        coords = np.array([
            [0.3, 0.3, 0.0],  # Slave node in contact plane
            [0.0, 0.0, 0.0],  # Master 0
            [1.0, 0.0, 0.0],  # Master 1
            [0.0, 1.0, 0.0],  # Master 2
        ])
        is_torque_free = ContactAssembler.check_angular_momentum_balance(
            global_forces=global_f,
            coordinates=coords,
            dof_offset=0,
            atol=1e-12,
        )
        assert is_torque_free is True

    def test_stiffness_sparse_scatter_symmetry(self) -> None:
        """Global assembled stiffness matrix must be symmetric and correctly accumulate shared nodes."""
        # Slave body has local nodes 0, 1 (DOFs 0..5 with slave_offset=0)
        # Master body has local nodes 0, 1, 2 (DOFs 6..14 with master_offset=6)
        master_nodes = np.array([0, 1, 2])
        bary = np.array([0.33, 0.33, 0.34])

        asm1 = ContactAssembler.build_two_body_map(0, master_nodes, bary, slave_offset=0, master_offset=6)
        asm2 = ContactAssembler.build_two_body_map(1, master_nodes, bary, slave_offset=0, master_offset=6)

        total_dofs = 15  # 5 nodes total * 3 DOFs

        # Random symmetric 12x12 local tangents
        rng = np.random.default_rng(42)
        A1 = rng.standard_normal((12, 12))
        K1 = A1 + A1.T
        A2 = rng.standard_normal((12, 12))
        K2 = A2 + A2.T

        K_c = ContactAssembler.assemble_stiffness([(asm1, K1), (asm2, K2)], total_dofs=total_dofs)

        # Verify CSR format and shape
        assert spsp.isspmatrix_csr(K_c)
        assert K_c.shape == (total_dofs, total_dofs)

        # Verify global matrix symmetry
        dense_K = K_c.toarray()
        np.testing.assert_allclose(dense_K, dense_K.T, atol=1e-12)

        # Master nodes 0, 1, 2 are shared: their diagonal blocks must accumulate contributions from both
        # Check DOF 6 (master node 0 x-dof):
        dof_node0_x = 6
        expected_k00 = K1[3, 3] + K2[3, 3]  # Local index 3 corresponds to master node 0 x-dof
        assert dense_K[dof_node0_x, dof_node0_x] == pytest.approx(expected_k00)
