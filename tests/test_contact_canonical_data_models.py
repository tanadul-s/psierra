"""Test Suite for Canonical Internal Contact Data Models & Geometry Providers.

Milestone 1.0 Stage B of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
Verifies:
1. Strict dataclass typing, immutability, and validation for:
   - ContactCandidate
   - DifferentiableContactGeometry
   - NormalLawState
   - FrictionalConstraintState
   - ContactAssemblyMap
2. Mathematical correctness and finite-difference consistency of geometry providers:
   - PlanarGeometryProvider (C^inf, vanishing Hessian)
   - SphereGeometryProvider (exact curvature Hessian (I - n (x) n) / d)
   - CylinderGeometryProvider (circumferential curvature Hessian (I - a (x) a - n (x) n) / d)
   - TriangleGeometryProvider (exact 12-DOF gap gradient and analytical Hessian)
3. Smooth-branch validity, physical units, and orientation conventions.
4. Downstream consumer integration (NormalLawState and ContactAssemblyMap).
"""

from __future__ import annotations

import numpy as np
import pytest

from psierra.boundary.friction import StickSlipState
from psierra.contact.data import (
    ContactCandidate,
    DifferentiableContactGeometry,
    NormalLawState,
    FrictionalConstraintState,
    ContactAssemblyMap,
)
from psierra.contact.geometry import (
    PlanarGeometryProvider,
    SphereGeometryProvider,
    CylinderGeometryProvider,
    TriangleGeometryProvider,
)


# =============================================================================
# 1. Internal Record Data Models & Validation
# =============================================================================

class TestContactDataModels:
    """Verify dataclass validation, immutability, and typing for internal contact records."""

    def test_contact_candidate_validation(self) -> None:
        """ContactCandidate must reject negative indices and negative search distances."""
        # Valid instance
        cand = ContactCandidate(
            candidate_index=0,
            slave_node=12,
            master_feature_id="obstacle_1",
            master_nodes=(0, 1, 2),
            search_distance=2.5,
        )
        assert cand.candidate_index == 0
        assert cand.slave_node == 12
        assert cand.is_admissible is True

        # Immutability
        with pytest.raises(AttributeError):
            cand.search_distance = 5.0  # type: ignore[misc]

        # Negative indices
        with pytest.raises(ValueError, match="candidate_index must be nonnegative"):
            ContactCandidate(candidate_index=-1, slave_node=0, master_feature_id="f")
        with pytest.raises(ValueError, match="slave_node must be nonnegative"):
            ContactCandidate(candidate_index=0, slave_node=-5, master_feature_id="f")
        with pytest.raises(ValueError, match="search_distance must be finite and nonnegative"):
            ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="f", search_distance=-0.1)

    def test_differentiable_contact_geometry_validation(self) -> None:
        """DifferentiableContactGeometry must validate dimensions, normal unit length, and Hessian symmetry."""
        normal = np.array([0.0, 0.0, 1.0])
        closest = np.array([0.5, 0.5, 0.0])
        grad = np.array([0.0, 0.0, 1.0])
        hess = np.zeros((3, 3))

        geom = DifferentiableContactGeometry(
            gap=-0.02,
            normal=normal,
            closest_point=closest,
            gradient=grad,
            hessian=hess,
            feature_id="plane",
        )
        assert geom.gap == -0.02
        assert geom.is_smooth_branch is True
        assert geom.units == "mm"
        assert geom.orientation_convention == "master_outward"

        # Immutability
        with pytest.raises(AttributeError):
            geom.gap = 0.0  # type: ignore[misc]

        # NaN gap rejection
        with pytest.raises(ValueError, match="Contact gap cannot be NaN"):
            DifferentiableContactGeometry(
                gap=float("nan"), normal=normal, closest_point=closest,
                gradient=grad, hessian=hess, feature_id="f",
            )

        # Non-unit normal rejection
        with pytest.raises(ValueError, match="Normal vector must have unit length"):
            DifferentiableContactGeometry(
                gap=-0.1, normal=np.array([0.0, 0.0, 2.0]), closest_point=closest,
                gradient=grad, hessian=hess, feature_id="f",
            )

        # Unsymmetric Hessian rejection
        unsym_hess = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="Hessian matrix must be symmetric"):
            DifferentiableContactGeometry(
                gap=-0.1, normal=normal, closest_point=closest,
                gradient=grad, hessian=unsym_hess, feature_id="f",
            )

        # Dimensional mismatch between gradient and Hessian
        with pytest.raises(ValueError, match="Hessian dimensions .* must match gradient length"):
            DifferentiableContactGeometry(
                gap=-0.1, normal=normal, closest_point=closest,
                gradient=grad, hessian=np.zeros((12, 12)), feature_id="f",
            )

    def test_normal_law_state_evaluation(self) -> None:
        """NormalLawState must correctly project pressure, potential energy, and derivative."""
        normal = np.array([0.0, 1.0, 0.0])
        geom_penetrated = DifferentiableContactGeometry(
            gap=-0.05,
            normal=normal,
            closest_point=np.zeros(3),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )

        # 1. Penalty evaluation on penetrating point (g = -0.05, kp = 1000)
        state_penalty = NormalLawState.evaluate(
            geometry=geom_penetrated, k_penalty=1000.0, formulation="penalty"
        )
        assert state_penalty.pressure == pytest.approx(50.0)
        assert state_penalty.multiplier == 0.0
        assert state_penalty.is_active is True
        assert state_penalty.first_derivative == -1000.0
        assert state_penalty.potential_energy == pytest.approx(0.5 * 1000.0 * 0.05**2)

        # 2. Augmented Lagrangian with stored multiplier (lambda = 20.0, kp = 1000)
        # p = max(0, 20 - 1000 * (-0.05)) = 20 + 50 = 70 N
        state_al = NormalLawState.evaluate(
            geometry=geom_penetrated, k_penalty=1000.0, multiplier=20.0, formulation="augmented_lagrangian"
        )
        assert state_al.pressure == pytest.approx(70.0)
        assert state_al.multiplier == 20.0
        assert state_al.is_active is True
        assert state_al.potential_energy == pytest.approx((70.0**2 - 20.0**2) / (2.0 * 1000.0))

        # 3. Separated point (g = +0.10) in penalty
        geom_separated = DifferentiableContactGeometry(
            gap=0.10,
            normal=normal,
            closest_point=np.zeros(3),
            gradient=normal,
            hessian=np.zeros((3, 3)),
            feature_id="plane",
        )
        state_sep = NormalLawState.evaluate(
            geometry=geom_separated, k_penalty=1000.0, formulation="penalty"
        )
        assert state_sep.pressure == 0.0
        assert state_sep.is_active is False
        assert state_sep.potential_energy == 0.0
        assert state_sep.first_derivative == 0.0

    def test_frictional_constraint_state_validation(self) -> None:
        """FrictionalConstraintState must validate tensor shapes for force and stiffness blocks."""
        state = FrictionalConstraintState(
            state=StickSlipState.STICK,
            anchor=np.array([0.0, 0.0, 0.0]),
            trial_anchor=np.array([0.0, 0.0, 0.0]),
            force=np.array([10.0, 0.0, 0.0]),
            tangential_stiffness=np.eye(3) * 500.0,
            normal_coupling_stiffness=np.zeros((3, 3)),
            slip_direction=np.array([1.0, 0.0, 0.0]),
            slip_distance=0.0,
            trial_slip_increment=0.0,
        )
        assert state.state == StickSlipState.STICK
        assert state.force[0] == 10.0

        # Invalid shape
        with pytest.raises(ValueError, match="Friction force must be 3D"):
            FrictionalConstraintState(
                state=StickSlipState.STICK,
                anchor=np.zeros(3),
                trial_anchor=np.zeros(3),
                force=np.zeros(2),  # Invalid 2D
                tangential_stiffness=np.zeros((3, 3)),
                normal_coupling_stiffness=np.zeros((3, 3)),
                slip_direction=np.zeros(3),
            )

    def test_contact_assembly_map_validation(self) -> None:
        """ContactAssemblyMap must validate DOF slice dimensions and stencil compatibility."""
        slave_dofs = np.array([0, 1, 2], dtype=np.int64)
        master_nodes = np.array([10, 11, 12], dtype=np.int64)
        master_dofs = np.arange(30, 39, dtype=np.int64)
        stencil_dofs = np.r_[slave_dofs, master_dofs]
        weights = np.array([1.0, -0.33, -0.33, -0.34])

        asm = ContactAssemblyMap(
            slave_node=0,
            slave_dofs=slave_dofs,
            master_nodes=master_nodes,
            master_dofs=master_dofs,
            stencil_dofs=stencil_dofs,
            interpolation_weights=weights,
            slave_offset=0,
            master_offset=30,
        )
        assert len(asm.stencil_dofs) == 12
        assert len(asm.interpolation_weights) == 4

        # Mismatched weights length
        with pytest.raises(ValueError, match="Weights count .* must match stencil DOFs length"):
            ContactAssemblyMap(
                slave_node=0,
                slave_dofs=slave_dofs,
                master_nodes=master_nodes,
                master_dofs=master_dofs,
                stencil_dofs=stencil_dofs,
                interpolation_weights=np.array([1.0, -0.5]),  # length 2 != 4
            )


# =============================================================================
# 2. Geometry Providers & Analytical / Finite Difference Verification
# =============================================================================

class TestPlanarGeometryProvider:
    """Verify exact analytical planar geometry provider."""

    def test_planar_geometry_and_finite_differences(self) -> None:
        """Verify planar gap, normal, and vanishing Hessian via centered finite differences."""
        origin = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        provider = PlanarGeometryProvider(origin=origin, normal=normal)

        assert provider.units == "mm"
        assert provider.orientation_convention == "master_outward"

        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="plane")
        # Point penetrating into plane at y = 0.8 mm (gap = -0.2 mm)
        point = np.array([2.5, 0.8, -1.2])

        geom = provider.evaluate_geometry(cand, point)
        assert geom.gap == pytest.approx(-0.2)
        np.testing.assert_allclose(geom.normal, normal)
        np.testing.assert_allclose(geom.closest_point, [2.5, 1.0, -1.2])
        np.testing.assert_allclose(geom.gradient, normal)
        np.testing.assert_allclose(geom.hessian, 0.0, atol=1e-12)
        assert geom.is_smooth_branch is True

        # Centered finite-difference verification of gradient
        eps = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            g_plus = provider.evaluate_geometry(cand, point + delta).gap
            g_minus = provider.evaluate_geometry(cand, point - delta).gap
            grad_fd[i] = (g_plus - g_minus) / (2.0 * eps)

        np.testing.assert_allclose(grad_fd, geom.gradient, atol=1e-8)


class TestSphereGeometryProvider:
    """Verify exact analytical spherical geometry provider and curvature Hessian."""

    def test_sphere_geometry_curvature_hessian(self) -> None:
        """Verify that SphereGeometryProvider Hessian matches analytical (I - n n.T) / d and finite differences."""
        center = np.array([1.0, 2.0, 3.0])
        radius = 2.5
        provider = SphereGeometryProvider(center=center, radius=radius)

        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="sphere")
        # Point outside/inside sphere
        point = center + np.array([1.2, -1.6, 1.5])
        diff = point - center
        dist = np.linalg.norm(diff)
        expected_normal = diff / dist

        geom = provider.evaluate_geometry(cand, point)
        assert geom.gap == pytest.approx(dist - radius)
        np.testing.assert_allclose(geom.normal, expected_normal, atol=1e-12)
        np.testing.assert_allclose(geom.gradient, expected_normal, atol=1e-12)
        assert geom.is_smooth_branch is True

        # Analytical curvature Hessian formula: (I - n (x) n) / d
        expected_hess = (np.eye(3) - np.outer(expected_normal, expected_normal)) / dist
        np.testing.assert_allclose(geom.hessian, expected_hess, atol=1e-12)

        # Eigenvalue verification: one zero eigenvalue along normal, two equal to 1/d
        eigvals = np.linalg.eigvalsh(geom.hessian)
        np.testing.assert_allclose(eigvals[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(eigvals[1:], 1.0 / dist, rtol=1e-6)

        # Centered finite differences: dg / dx == gradient
        eps = 1e-6
        grad_fd = np.zeros(3)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            g_plus = provider.evaluate_geometry(cand, point + delta).gap
            g_minus = provider.evaluate_geometry(cand, point - delta).gap
            grad_fd[i] = (g_plus - g_minus) / (2.0 * eps)
        np.testing.assert_allclose(grad_fd, geom.gradient, atol=1e-7)

        # Centered finite differences: d(grad) / dx == Hessian
        hess_fd = np.zeros((3, 3))
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = eps
            grad_plus = provider.evaluate_geometry(cand, point + delta).gradient
            grad_minus = provider.evaluate_geometry(cand, point - delta).gradient
            hess_fd[:, j] = (grad_plus - grad_minus) / (2.0 * eps)
        np.testing.assert_allclose(hess_fd, geom.hessian, atol=1e-5)


class TestCylinderGeometryProvider:
    """Verify exact analytical cylinder geometry provider and circumferential curvature Hessian."""

    def test_cylinder_geometry_circumferential_hessian(self) -> None:
        """Verify cylinder gap, normal, and Hessian (I - a a.T - n n.T) / d via finite differences."""
        center = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 3.0
        provider = CylinderGeometryProvider(center=center, axis=axis, radius=radius)

        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="cylinder")
        # Point at x = 2.0, y = 1.5, z = 4.0
        point = np.array([2.0, 1.5, 4.0])
        rad_vec = np.array([2.0, 1.5, 0.0])
        dist = np.linalg.norm(rad_vec)
        expected_normal = rad_vec / dist

        geom = provider.evaluate_geometry(cand, point)
        assert geom.gap == pytest.approx(dist - radius)
        np.testing.assert_allclose(geom.normal, expected_normal, atol=1e-12)
        np.testing.assert_allclose(geom.gradient, expected_normal, atol=1e-12)

        # Analytical circumferential Hessian: (I - a a.T - n n.T) / d
        P_circ = np.eye(3) - np.outer(axis, axis) - np.outer(expected_normal, expected_normal)
        expected_hess = P_circ / dist
        np.testing.assert_allclose(geom.hessian, expected_hess, atol=1e-12)

        # Eigenvalues: zero along normal, zero along axis, 1/d along circumferential binormal
        eigvals = np.linalg.eigvalsh(geom.hessian)
        np.testing.assert_allclose(eigvals[:2], 0.0, atol=1e-12)
        assert eigvals[2] == pytest.approx(1.0 / dist, rel=1e-6)

        # Centered finite differences: d(grad) / dx == Hessian
        eps = 1e-6
        hess_fd = np.zeros((3, 3))
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = eps
            grad_plus = provider.evaluate_geometry(cand, point + delta).gradient
            grad_minus = provider.evaluate_geometry(cand, point - delta).gradient
            hess_fd[:, j] = (grad_plus - grad_minus) / (2.0 * eps)
        np.testing.assert_allclose(hess_fd, geom.hessian, atol=1e-5)


class TestTriangleGeometryProvider:
    """Verify closest-feature node-to-triangle geometry provider."""

    def test_triangle_geometry_12dof_gradient_and_hessian(self) -> None:
        """Verify node-triangle 12-DOF gap gradient and Hessian on interior face projection."""
        provider = TriangleGeometryProvider()
        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="tri_0")

        # Stencil: slave node at [0.3, 0.3, -0.05] (penetrating by 0.05 into xy triangle)
        slave_pt = np.array([0.3, 0.3, -0.05])
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        coords = np.vstack([slave_pt, v0, v1, v2])

        geom = provider.evaluate_geometry(cand, coords)
        assert geom.feature_id == "face"
        assert geom.gap == pytest.approx(-0.05)
        np.testing.assert_allclose(geom.normal, [0.0, 0.0, 1.0])
        assert geom.is_smooth_branch is True
        assert geom.gradient.shape == (12,)
        assert geom.hessian.shape == (12, 12)

        # Hessian must be symmetric
        np.testing.assert_allclose(geom.hessian, geom.hessian.T, atol=1e-12)

        # 12-DOF finite-difference verification of gradient:
        # q = [x_s, v0, v1, v2] in R^12
        eps = 1e-6
        grad_fd = np.zeros(12)
        q_flat = coords.ravel()
        for k in range(12):
            delta = np.zeros(12)
            delta[k] = eps
            coords_plus = (q_flat + delta).reshape(4, 3)
            coords_minus = (q_flat - delta).reshape(4, 3)
            g_plus = provider.evaluate_geometry(cand, coords_plus).gap
            g_minus = provider.evaluate_geometry(cand, coords_minus).gap
            grad_fd[k] = (g_plus - g_minus) / (2.0 * eps)

        np.testing.assert_allclose(grad_fd, geom.gradient, atol=1e-6)

        # 12-DOF finite-difference verification of Hessian:
        hess_fd = np.zeros((12, 12))
        for j in range(12):
            delta = np.zeros(12)
            delta[j] = eps
            coords_plus = (q_flat + delta).reshape(4, 3)
            coords_minus = (q_flat - delta).reshape(4, 3)
            grad_plus = provider.evaluate_geometry(cand, coords_plus).gradient
            grad_minus = provider.evaluate_geometry(cand, coords_minus).gradient
            hess_fd[:, j] = (grad_plus - grad_minus) / (2.0 * eps)

        np.testing.assert_allclose(hess_fd, geom.hessian, atol=1e-5)


# =============================================================================
# 3. Downstream Sole-Source Consumer Integration
# =============================================================================

class TestDownstreamSoleSourceIntegration:
    """Verify that canonical records act as the sole source of geometry for downstream solvers."""

    def test_assembly_and_force_evaluation_from_canonical_record(self) -> None:
        """Verify assembling global contact force and stiffness solely from DifferentiableContactGeometry."""
        provider = PlanarGeometryProvider(origin=[0, 0, 0], normal=[0, 0, 1])
        cand = ContactCandidate(candidate_index=0, slave_node=2, master_feature_id="floor")
        slave_coord = np.array([1.0, 1.0, -0.04])  # 0.04 mm penetration

        # 1. Sole geometry evaluation
        geom = provider.evaluate_geometry(cand, slave_coord)

        # 2. Constraint law evaluation consuming only canonical geometry
        kp = 5000.0
        law_state = NormalLawState.evaluate(geom, k_penalty=kp, formulation="penalty")
        assert law_state.pressure == pytest.approx(200.0)

        # 3. Assembly map consuming candidate and domain offsets
        slave_dofs = np.array([6, 7, 8], dtype=np.int64)  # Node 2 DOFs (2 * 3)
        asm_map = ContactAssemblyMap(
            slave_node=cand.slave_node,
            slave_dofs=slave_dofs,
            master_nodes=np.zeros(0, dtype=np.int64),
            master_dofs=np.zeros(0, dtype=np.int64),
            stencil_dofs=slave_dofs,
            interpolation_weights=np.array([1.0]),
            slave_offset=0,
            master_offset=0,
        )

        # 4. Global scatter using purely canonical records
        global_force = np.zeros(12, dtype=np.float64)
        local_force = law_state.pressure * geom.gradient  # f = p * grad g
        global_force[asm_map.slave_dofs] = local_force

        # Verify global force vector
        assert global_force[8] == pytest.approx(200.0)  # +z repulsive force on slave node 2
        np.testing.assert_allclose(global_force[:8], 0.0)
        np.testing.assert_allclose(global_force[9:], 0.0)
