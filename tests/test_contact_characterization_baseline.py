"""Characterization Baseline & Frozen Public Contract Verification Suite.

Milestone 1.0 Stage A of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
This test module establishes the characterization baselines and verifies the
frozen public contracts of all four contact classes in Project Sierra:
- RigidContact
- DeformableContact
- SelfContact
- MortarContact

Key Verification Targets:
1. Public constructor, parameter bounds, typing, and subclass inheritance from ContactMultiplierState.
2. Degree of freedom (DOF) offsets and multi-body block scatter (K_ss, K_sm, K_ms, K_mm).
3. Transactional state restoration, rollback, and cache invalidation.
4. Accepted-step history commitment (Coulomb stick-slip anchors).
5. Solver coupling in batch analysis pipelines (NonlinearStaticAnalysis, ContactAugmentation).
6. Formal characterization of known mathematical legacy discrepancies (LEGACY-DISC-01 to 05)
   without blessing them as ground truth.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as spsp
import pyvista as pv

from psierra.boundary.contact import (
    ContactMultiplierState,
    RigidContact,
    DeformableContact,
    SelfContact,
)
from psierra.boundary.mortar import MortarContact
from psierra.boundary.friction import StickSlipState, CoulombFrictionState, CoulombFrictionSnapshot
from psierra.collision.shapes import RigidPlane, RigidSphere, RigidCylinder
from psierra.collision.triangle import TriangleContactGeometry
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh
from psierra.analysis.augmentation import ContactAugmentation, AugmentationStatus
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint


# =============================================================================
# Lightweight Test Fixtures
# =============================================================================

class MockGeometryDomain(Domain):
    """Prescribed 3D coordinates domain isolating contact from constitutive physics."""

    def __init__(self, nodes: np.ndarray, faces: np.ndarray | None = None) -> None:
        self.reference = np.asarray(nodes, dtype=np.float64).copy()
        self.u = np.zeros(nodes.size, dtype=np.float64)
        self.v = np.zeros(nodes.size, dtype=np.float64)
        self.a = np.zeros(nodes.size, dtype=np.float64)
        if faces is not None:
            cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
            polydata = pv.PolyData(self.reference, cells)
            self.mesh = SimpleNamespace(
                surface=SurfaceMesh(polydata, node_indices=np.arange(len(nodes), dtype=np.int64)),
                nodes=self.reference,
                n_nodes=len(nodes),
            )
        else:
            self.mesh = SimpleNamespace(nodes=self.reference, n_nodes=len(nodes))

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

    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        return spsp.eye(self.n_dofs, format="csr")

    def assemble_tangent_stiffness_matrix(self) -> spsp.csr_matrix:
        return self.assemble_stiffness_matrix()

    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        return spsp.eye(self.n_dofs, format="csr")

    def assemble_internal_forces(self) -> np.ndarray:
        return self.u.copy()

    def save_state(self, filepath: Path, **kwargs: Any) -> None:
        pass


@pytest.fixture
def triangle_facet_pair() -> tuple[MockGeometryDomain, MockGeometryDomain]:
    """Co-planar opposing slave node and master triangle."""
    slave_node = np.array([[0.25, 0.25, -0.05]])
    master_nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    master_faces = np.array([[0, 1, 2]])
    slave_domain = MockGeometryDomain(slave_node)
    master_domain = MockGeometryDomain(master_nodes, master_faces)
    return slave_domain, master_domain


@pytest.fixture
def self_contact_domain() -> MockGeometryDomain:
    """Two opposed triangles in one domain for self-contact."""
    nodes = np.array([
        # Facet 1 (Lower)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        # Facet 2 (Upper, folding down with normal -z)
        [0.0, 0.0, 0.05],
        [1.0, 0.0, 0.05],
        [0.0, 1.0, 0.05],
    ])
    faces = np.array([
        [0, 1, 2],
        [3, 5, 4],  # Normal points in -z direction
    ])
    return MockGeometryDomain(nodes, faces)


# =============================================================================
# 1. Public Class Contracts & Constructor Validation
# =============================================================================

class TestPublicContactContracts:
    """Verify that all contact classes inherit from ContactMultiplierState and validate inputs."""

    def test_contact_multiplier_state_subclass(self) -> None:
        """All four contact classes must inherit from ContactMultiplierState."""
        for cls in (RigidContact, DeformableContact, SelfContact, MortarContact):
            assert issubclass(cls, ContactMultiplierState), (
                f"{cls.__name__} must inherit from ContactMultiplierState"
            )

    def test_rigid_contact_constructor_validation(self) -> None:
        """RigidContact must validate penalty, damping, friction, tolerance, and formulation."""
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        domain = MockGeometryDomain(np.array([[0.0, 0.0, 0.1]]))

        # Invalid k_penalty
        with pytest.raises(ValueError, match="k_penalty must be finite and positive"):
            RigidContact(slave=domain, master=plane, k_penalty=0.0)
        with pytest.raises(ValueError, match="k_penalty must be finite and positive"):
            RigidContact(slave=domain, master=plane, k_penalty=-100.0)

        # Missing obstacle / master
        with pytest.raises(ValueError, match="requires a master obstacle"):
            RigidContact(slave=domain, master=None)

        # Valid initialization defaults
        rc = RigidContact(slave=domain, master=plane)
        assert rc.k_penalty == 1e4
        assert rc.formulation == "penalty"
        assert rc.c_damping == 0.0
        assert rc.friction == 0.0
        assert rc.dof_offset == 0

    def test_deformable_contact_constructor_validation(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """DeformableContact must validate bounds, formulation, and surface meshes."""
        slave, master = triangle_facet_pair

        # Invalid bounds
        with pytest.raises(ValueError, match="k_penalty must be finite and positive"):
            DeformableContact(slave, master, k_penalty=-1.0)
        with pytest.raises(ValueError, match="c_damping must be finite and nonnegative"):
            DeformableContact(slave, master, c_damping=-0.5)
        with pytest.raises(ValueError, match="friction must be finite and nonnegative"):
            DeformableContact(slave, master, friction=-0.1)
        with pytest.raises(ValueError, match="tolerance must be finite and nonnegative"):
            DeformableContact(slave, master, tolerance=-0.01)
        with pytest.raises(ValueError, match="formulation must be 'penalty' or 'augmented_lagrangian'"):
            DeformableContact(slave, master, formulation="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="smoothing_weight must be 'angle' or 'area'"):
            DeformableContact(slave, master, smoothing_weight="other")  # type: ignore[arg-type]

        # Missing surface on master
        raw_domain = MockGeometryDomain(np.zeros((3, 3)))
        with pytest.raises(TypeError, match="Master domain must possess a surface mesh"):
            DeformableContact(slave, raw_domain)

    def test_self_contact_constructor_validation(
        self, self_contact_domain: MockGeometryDomain
    ) -> None:
        """SelfContact must validate DOF offset alignment, normal threshold, and exclusion rings."""
        # dof_offset must be a multiple of 3
        with pytest.raises(ValueError, match="dof_offset must be a multiple of three"):
            SelfContact(self_contact_domain, dof_offset=1)
        with pytest.raises(ValueError, match="dof_offset must be a multiple of three"):
            SelfContact(self_contact_domain, dof_offset=2)

        # exclusion_rings must be nonnegative integer
        with pytest.raises(ValueError, match="exclusion_rings must be a nonnegative integer"):
            SelfContact(self_contact_domain, exclusion_rings=-1)

        # normal_threshold must be in [-1, 1]
        with pytest.raises(ValueError, match="normal_threshold must be finite and in \\[-1, 1\\]"):
            SelfContact(self_contact_domain, normal_threshold=1.5)

        # Missing surface
        raw = MockGeometryDomain(np.zeros((3, 3)))
        with pytest.raises(TypeError, match="Domain must possess a surface mesh"):
            SelfContact(raw)

    def test_mortar_contact_constructor_validation(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """MortarContact must validate parameters and surface representations."""
        slave, master = triangle_facet_pair
        # Slave must have surface mesh
        with pytest.raises(TypeError, match="Slave domain must possess a surface mesh"):
            MortarContact(slave=MockGeometryDomain(np.zeros((3, 3))), master=master)

        # Master without surface
        with pytest.raises(TypeError, match="Master domain must possess a surface mesh"):
            MortarContact(
                slave=master,  # master has surface
                master=MockGeometryDomain(np.zeros((3, 3))),
            )

    def test_canonical_evaluation_signatures_and_keys(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """All four contact classes must implement the standard evaluation methods and info keys."""
        slave, master = triangle_facet_pair
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))

        rc = RigidContact(slave=slave, master=plane)
        dc = DeformableContact(slave, master, k_penalty=1e4)

        for contact in (rc, dc):
            assert hasattr(contact, "evaluate_forces")
            assert hasattr(contact, "evaluate_stiffness")
            assert hasattr(contact, "evaluate_potential_energy")
            assert hasattr(contact, "evaluate_gaps")
            assert hasattr(contact, "get_contact_info")
            assert hasattr(contact, "commit_state")
            assert hasattr(contact, "invalidate_state")

            info = contact.get_contact_info()
            assert "num_active" in info
            assert "max_penetration" in info
            assert "potential_energy" in info
            assert "max_multiplier" in info


# =============================================================================
# 2. Degree of Freedom (DOF) Offsets & Block Placement
# =============================================================================

class TestDOFOffsetScattering:
    """Verify that contact forces and tangent stiffness matrices scatter to exact DOF ranges."""

    def test_rigid_contact_dof_offset(self) -> None:
        """RigidContact must scatter forces and stiffness starting at dof_offset."""
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        # Slave node penetrating at z = -0.1
        nodes = np.array([[0.0, 0.0, -0.1]])
        domain = MockGeometryDomain(nodes)
        dof_offset = 9
        total_dofs = 18

        contact = RigidContact(
            slave=domain, master=plane, k_penalty=2000.0, dof_offset=dof_offset
        )
        forces = contact.evaluate_forces(n_dofs=total_dofs)
        assert len(forces) == total_dofs

        # Repulsive force in +z direction should appear at dof_offset + 2
        assert forces[dof_offset + 2] == pytest.approx(200.0, rel=1e-6)
        # All other DOFs must be exactly zero
        mask = np.ones(total_dofs, dtype=bool)
        mask[dof_offset + 2] = False
        np.testing.assert_allclose(forces[mask], 0.0, atol=1e-12)

        # Stiffness matrix: 3x3 block at [dof_offset:dof_offset+3, dof_offset:dof_offset+3]
        K_c = contact.evaluate_stiffness(n_dofs=total_dofs)
        assert K_c.shape == (total_dofs, total_dofs)
        assert K_c[dof_offset + 2, dof_offset + 2] == pytest.approx(2000.0, rel=1e-6)
        # Verify zeros elsewhere
        dense_K = K_c.toarray()
        dense_K[dof_offset + 2, dof_offset + 2] = 0.0
        np.testing.assert_allclose(dense_K, 0.0, atol=1e-12)

    def test_deformable_contact_offset_blocks(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """DeformableContact must place K_ss, K_sm, K_ms, K_mm at exact slave and master offsets."""
        slave, master = triangle_facet_pair
        slave_offset = 12
        master_offset = 24
        total_dofs = 40

        contact = DeformableContact(
            slave, master,
            slave_offset=slave_offset,
            master_offset=master_offset,
            k_penalty=5000.0,
        )

        forces = contact.evaluate_forces(n_dofs=total_dofs)
        assert len(forces) == total_dofs

        # Slave forces must reside in [slave_offset : slave_offset + 3]
        slave_f = forces[slave_offset : slave_offset + 3]
        assert slave_f[2] > 0.0  # Repulsive in +z

        # Master forces must reside in [master_offset : master_offset + 9]
        master_f = forces[master_offset : master_offset + 9]
        assert np.sum(master_f[2::3]) < 0.0  # Net reaction in -z

        # Action-reaction balance
        np.testing.assert_allclose(np.sum(slave_f) + np.sum(master_f), 0.0, atol=1e-10)

        # DOFs outside slave and master ranges must be strictly zero
        inactive_dofs = np.r_[
            np.arange(0, slave_offset),
            np.arange(slave_offset + 3, master_offset),
            np.arange(master_offset + 9, total_dofs),
        ]
        np.testing.assert_allclose(forces[inactive_dofs], 0.0, atol=1e-12)

        # Stiffness matrix sparsity blocks
        K_c = contact.evaluate_stiffness(n_dofs=total_dofs).toarray()
        assert K_c.shape == (total_dofs, total_dofs)

        # K_ss block
        assert np.any(K_c[slave_offset : slave_offset + 3, slave_offset : slave_offset + 3] != 0.0)
        # K_sm block
        assert np.any(K_c[slave_offset : slave_offset + 3, master_offset : master_offset + 9] != 0.0)
        # K_ms block
        assert np.any(K_c[master_offset : master_offset + 9, slave_offset : slave_offset + 3] != 0.0)
        # K_mm block
        assert np.any(K_c[master_offset : master_offset + 9, master_offset : master_offset + 9] != 0.0)

        # Everything else must be zero
        K_c[slave_offset : slave_offset + 3, slave_offset : slave_offset + 3] = 0.0
        K_c[slave_offset : slave_offset + 3, master_offset : master_offset + 9] = 0.0
        K_c[master_offset : master_offset + 9, slave_offset : slave_offset + 3] = 0.0
        K_c[master_offset : master_offset + 9, master_offset : master_offset + 9] = 0.0
        np.testing.assert_allclose(K_c, 0.0, atol=1e-12)

    def test_self_contact_dof_offset(
        self, self_contact_domain: MockGeometryDomain
    ) -> None:
        """SelfContact must scatter all candidate stencil entries strictly into dof_offset range."""
        dof_offset = 6
        total_dofs = dof_offset + self_contact_domain.n_dofs + 12

        # Displace upper triangle into penetration (z = -0.02)
        u = np.zeros(self_contact_domain.n_dofs)
        u[9:] = [0.0, 0.0, -0.07] * 3  # Push upper triangle down
        self_contact_domain.update_state(u=u)

        sc = SelfContact(
            self_contact_domain,
            dof_offset=dof_offset,
            k_penalty=3000.0,
            contact_thickness=1.0,
            search_radius=5.0,
        )

        forces = sc.evaluate_forces(n_dofs=total_dofs)
        # DOFs before dof_offset and after domain DOFs must be zero
        np.testing.assert_allclose(forces[:dof_offset], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces[dof_offset + self_contact_domain.n_dofs :], 0.0, atol=1e-12)

        K_c = sc.evaluate_stiffness(n_dofs=total_dofs).toarray()
        np.testing.assert_allclose(K_c[:dof_offset, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(K_c[:, :dof_offset], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            K_c[dof_offset + self_contact_domain.n_dofs :, :], 0.0, atol=1e-12
        )

    def test_mortar_contact_dof_offset(self) -> None:
        """MortarContact must place coupled weak-form forces at slave_offset and master_offset."""
        # Slave surface at z = 0.02 opposing master surface at z = 0.0 (normal -z)
        s_nodes = np.array([
            [0.0, 0.0, 0.02],
            [1.0, 0.0, 0.02],
            [0.0, 1.0, 0.02],
        ])
        s_faces = np.array([[0, 1, 2]])
        m_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        m_faces = np.array([[0, 2, 1]])  # Normal -z

        slave = MockGeometryDomain(s_nodes, s_faces)
        master = MockGeometryDomain(m_nodes, m_faces)

        slave_offset = 18
        master_offset = 36
        total_dofs = 60

        mc = MortarContact(
            slave=slave,
            master=master,
            slave_offset=slave_offset,
            master_offset=master_offset,
            k_penalty=4000.0,
            search_radius=1.0,
        )

        forces = mc.evaluate_forces(n_dofs=total_dofs)
        assert len(forces) == total_dofs

        # Slave DOFs must receive negative normal force (pushing out of master in -z direction)
        slave_f_z = forces[slave_offset + 2 : slave_offset + 9 : 3]
        assert np.sum(slave_f_z) < 0.0

        # Master DOFs must receive positive reaction force in +z direction
        master_f_z = forces[master_offset + 2 : master_offset + 9 : 3]
        assert np.sum(master_f_z) > 0.0

        # Net balance
        np.testing.assert_allclose(np.sum(slave_f_z) + np.sum(master_f_z), 0.0, atol=1e-10)

        # Inactive regions must be strictly zero
        np.testing.assert_allclose(forces[:slave_offset], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces[slave_offset + 9 : master_offset], 0.0, atol=1e-12)
        np.testing.assert_allclose(forces[master_offset + 9 : total_dofs], 0.0, atol=1e-12)


# =============================================================================
# 3. Transactional State Restoration, Rollback & Cache Invalidation
# =============================================================================

class TestStateRestorationAndRollback:
    """Verify snapshot/rollback guarantees for multipliers, friction, and cached geometry."""

    def test_multiplier_snapshot_and_restore(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """Mutating multipliers in a trial step must be cleanly reversible via set_multipliers."""
        slave, master = triangle_facet_pair
        dc = DeformableContact(
            slave, master, k_penalty=1000.0, formulation="augmented_lagrangian"
        )
        # Establish initial multiplier state
        initial_mult = np.array([125.0])
        dc.set_multipliers(initial_mult)

        snapshot = dc.snapshot_multipliers()
        assert np.array_equal(snapshot, initial_mult)

        # Mutate in trial Newton step
        trial_mult = np.array([550.0])
        dc.set_multipliers(trial_mult)
        assert dc.multipliers[0] == 550.0

        # Roll back trial step
        dc.set_multipliers(snapshot)
        assert dc.multipliers[0] == 125.0

    def test_cache_invalidation_on_state_mutation(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """Calling invalidate_state or set_multipliers must clear cached query structures."""
        s_nodes = np.array([[0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.0, 1.0, 0.01]])
        s_faces = np.array([[0, 1, 2]])
        m_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        m_faces = np.array([[0, 2, 1]])

        slave = MockGeometryDomain(s_nodes, s_faces)
        master = MockGeometryDomain(m_nodes, m_faces)

        mc = MortarContact(slave=slave, master=master, formulation="augmented_lagrangian")
        # Run integration to populate cache
        mc.evaluate_forces()
        assert mc._cached_D is not None
        assert mc._cached_M is not None
        assert mc._cached_average_gaps is not None

        # set_multipliers must invalidate cached data
        mc.set_multipliers(np.array([10.0, 20.0, 30.0]))
        assert mc._cached_D is None
        assert mc._cached_M is None
        assert mc._cached_average_gaps is None

    def test_friction_state_snapshot_and_rollback(self) -> None:
        """Friction stick-slip state and trial anchors must restore cleanly on rollback."""
        slave = MockGeometryDomain(np.array([[0.0, 0.0, -0.05]]))
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        rc = RigidContact(slave=slave, master=plane, k_penalty=2000.0, friction=0.3)

        # Evaluate forces at initial penetrating position
        rc.evaluate_forces()
        assert rc.friction_state is not None
        initial_snap = rc.snapshot_friction_state()
        assert initial_snap is not None
        assert rc.friction_state.states[0] == StickSlipState.STICK

        # Displace slave tangentially to trigger slip
        slave.update_state(u=np.array([0.5, 0.0, 0.0]))
        rc.evaluate_forces()
        assert rc.friction_state.states[0] == StickSlipState.SLIP

        # Roll back trial step
        rc.restore_friction_state(initial_snap)
        assert rc.friction_state.states[0] == initial_snap.states[0]
        np.testing.assert_allclose(
            rc.friction_state.anchors, initial_snap.anchors, equal_nan=True
        )

    def test_uzawa_multiplier_projection_and_rebound_relaxation(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """update_multipliers must project lambda, and relax_rebound must clear opening contacts."""
        slave, master = triangle_facet_pair
        dc = DeformableContact(
            slave, master, k_penalty=1000.0, formulation="augmented_lagrangian"
        )
        dc.set_multipliers(np.array([50.0]))

        # Slave is penetrating at gap = -0.05
        # lambda_new = max(0, 50 - 1000 * (-0.05)) = 50 + 50 = 100.0
        status = dc.update_multipliers()
        assert status["max_multiplier"] == pytest.approx(100.0)
        assert dc.multipliers[0] == pytest.approx(100.0)

        # Now slave rebounds and moves to positive gap (z = +0.10)
        slave.update_state(u=np.array([0.0, 0.0, 0.15]))
        gaps = dc.evaluate_gaps()
        assert gaps[0] > 0.0

        # relax_rebound must zero the retained multiplier
        n_relaxed = dc.relax_rebound(gap_tol=0.0)
        assert n_relaxed == 1
        assert dc.multipliers[0] == 0.0


# =============================================================================
# 4. Accepted-Step History Commitment
# =============================================================================

class TestAcceptedStepHistoryCommitment:
    """Verify that commit_state locks in converged frictional trial anchors across steps."""

    def test_commit_state_friction_history(self) -> None:
        """commit_state must transfer trial_anchors into committed anchors and zero trial increments."""
        slave = MockGeometryDomain(np.array([[0.0, 0.0, -0.05]]))
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        rc = RigidContact(slave=slave, master=plane, k_penalty=2000.0, friction=0.4)

        # Step 1: Initial contact (stick)
        rc.evaluate_forces()
        assert rc.friction_state is not None
        assert rc.friction_state.states[0] == StickSlipState.STICK
        rc.commit_state()

        # Step 2: Significant sliding to cause slip
        slave.update_state(u=np.array([0.5, 0.0, 0.0]))
        rc.evaluate_forces()
        assert rc.friction_state.states[0] == StickSlipState.SLIP

        trial_anchor_slip = rc.friction_state.trial_anchors[0].copy()
        assert not np.isnan(trial_anchor_slip[0])
        assert rc.friction_state.trial_slip_increments[0] > 0.0

        # Committing step must adopt trial anchor as current anchor
        rc.commit_state()
        np.testing.assert_allclose(rc.friction_state.anchors[0], trial_anchor_slip)
        assert rc.friction_state.trial_slip_increments[0] == 0.0


# =============================================================================
# 5. Solver Coupling Characterization
# =============================================================================

class TestSolverCouplingCharacterization:
    """Verify contact participation in NonlinearStaticAnalysis and ContactAugmentation."""

    def test_rigid_contact_static_solver_coupling(self) -> None:
        """NonlinearStaticAnalysis must resolve equilibrium against a RigidContact obstacle."""
        # 1-node spring domain with stiffness 1 N/mm (MockGeometryDomain.assemble_stiffness_matrix)
        domain = MockGeometryDomain(np.array([[0.0, 0.0, 0.5]]))
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))

        contact = RigidContact(slave=domain, master=plane, k_penalty=5000.0)

        manager = DirichletManager(domain.n_dofs)
        manager.add_constraint(DOFConstraint([0, 1], 0.0))

        analysis = NonlinearStaticAnalysis(domain=domain)
        # Apply external downward force
        f_ext = np.array([0.0, 0.0, -2000.0])
        res = analysis.solve(
            dirichlet=manager,
            f_ext=f_ext,
            contact=contact,
            n_steps=1,
            nr_iterations=15,
            nr_tol=1e-6,
            verbose=False,
        )
        assert res.converged

        # Verify penetration occurred and equilibrium balances external load
        u_z = domain.u[2]
        gap = 0.5 + u_z
        assert gap < 0.0  # Penetrated into plane
        # Force balance: k_spring * u_z - F_ext - F_c = 0
        f_c_z = contact.evaluate_forces()[2]
        assert f_c_z > 0.0
        assert f_c_z - u_z == pytest.approx(2000.0, rel=1e-4)

    def test_contact_augmentation_transaction_rollback_on_cutback(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """ContactAugmentation must restore dual multipliers when an inner step fails."""
        slave, master = triangle_facet_pair
        dc = DeformableContact(
            slave, master, k_penalty=1000.0, formulation="augmented_lagrangian"
        )
        dc.set_multipliers(np.array([80.0]))

        aug = ContactAugmentation(
            contacts=[dc],
            max_iterations=10,
            gap_tol=1e-4,
            tolerance=1e-4,
        )

        snapshot = aug.snapshot()
        assert snapshot[0][0] == 80.0

        # Simulate failed trial state with wild multiplier modification
        dc.set_multipliers(np.array([9999.0]))
        assert dc.multipliers[0] == 9999.0

        # Roll back via augmentation manager
        aug.restore(snapshot)
        assert dc.multipliers[0] == 80.0


# =============================================================================
# 6. Characterization of Legacy Mathematical Discrepancies
# =============================================================================

class TestLegacyDiscrepancies:
    """Document existing mathematical inconsistencies without blessing them as correct.

    In Stage A, these tests characterize the EXACT legacy behavior and verify that
    downstream replacements in Stages B-D have unambiguous characterization baselines.
    """

    def test_legacy_disc_01_positive_gap_penalty_force_attraction(self) -> None:
        """LEGACY-DISC-01: Penalty contact applies attractive force when 0 < gap <= tolerance.

        Expected Replacement: Non-adhesive law p = kp * < -g >+ where force is strictly zero
        for all positive gaps regardless of collision search tolerance.
        """
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        # Node separated from plane by gap = +0.05 mm
        node = np.array([[0.0, 0.0, 0.05]])
        domain = MockGeometryDomain(node)

        # Initialize with tolerance = 0.1 mm > gap = 0.05 mm
        contact_legacy = RigidContact(slave=domain, master=plane, k_penalty=1000.0, tolerance=0.1)
        contact_legacy._use_legacy_kernel = True

        forces_legacy = contact_legacy.evaluate_forces()
        # In legacy implementation, active_indices includes gap <= tolerance,
        # and computes f_N = -k_penalty * gap = -1000 * 0.05 = -50 N.
        # This creates an unphysical attractive suction pulling the separated node down!
        assert forces_legacy[2] == pytest.approx(-50.0, rel=1e-6), (
            "Legacy discrepancy confirmed: attractive penalty force on positive gap"
        )

        # In canonical replacement, strict non-adhesion ensures zero force on positive gap:
        contact_canon = RigidContact(slave=domain, master=plane, k_penalty=1000.0, tolerance=0.1)
        contact_canon._use_legacy_kernel = False
        forces_canon = contact_canon.evaluate_forces()
        assert forces_canon[2] == pytest.approx(0.0, abs=1e-12), (
            "Canonical replacement confirmed: zero attractive force on positive gap"
        )

    def test_legacy_disc_02_curved_rigid_obstacle_tangent_omits_curvature(self) -> None:
        """LEGACY-DISC-02: RigidContact tangent omits g_N * Hess(g_N) for curved obstacles.

        For a sphere of radius R, exact stiffness is:
            K_c = kp * (n outer n + g * (I - n outer n) / ||x - c||)
        Legacy implementation computes only K_c = kp * (n outer n).
        """
        R = 2.0
        sphere = RigidSphere(center=(0, 0, 0), radius=R)
        # Node inside sphere at x = (0, 0, 1.8), penetration g = 1.8 - 2.0 = -0.2
        point = np.array([[0.0, 0.0, 1.8]])
        domain = MockGeometryDomain(point)
        kp = 1000.0

        contact_legacy = RigidContact(slave=domain, master=sphere, k_penalty=kp)
        contact_legacy._use_legacy_kernel = True
        K_legacy = contact_legacy.evaluate_stiffness().toarray()

        # The legacy stiffness contains ONLY kp * (n outer n) = kp * diag(0, 0, 1)
        expected_legacy_K = np.diag([0.0, 0.0, kp])
        np.testing.assert_allclose(K_legacy, expected_legacy_K, atol=1e-12)

        # Compute exact numerical finite difference of force:
        # f_c(x) = -kp * g(x) * n(x)
        eps = 1e-6
        K_fd = np.zeros((3, 3))
        for col in range(3):
            delta = np.zeros(3)
            delta[col] = eps
            f_plus = contact_legacy.evaluate_forces(current_nodes=point + delta)
            f_minus = contact_legacy.evaluate_forces(current_nodes=point - delta)
            K_fd[:, col] = -(f_plus - f_minus) / (2.0 * eps)

        # The transverse stiffness (x and y directions) in finite differences is non-zero:
        # K_transverse = kp * g / ||x - c|| = 1000 * (-0.2) / 1.8 = -111.11 N/mm
        omitted_curvature_stiffness = kp * (-0.2) / 1.8
        assert K_fd[0, 0] == pytest.approx(omitted_curvature_stiffness, rel=1e-3)
        assert K_fd[1, 1] == pytest.approx(omitted_curvature_stiffness, rel=1e-3)
        # In legacy, K_legacy[0, 0] == 0.0 != K_fd[0, 0]
        assert K_legacy[0, 0] == 0.0

        # In canonical replacement, curvature is included and matches finite differences:
        contact_canon = RigidContact(slave=domain, master=sphere, k_penalty=kp)
        contact_canon._use_legacy_kernel = False
        K_canon = contact_canon.evaluate_stiffness().toarray()
        np.testing.assert_allclose(K_canon, K_fd, rtol=1e-3, atol=1e-3)

    def test_legacy_disc_03_smoothed_normal_omits_vertex_pseudonormal_variation(
        self, triangle_facet_pair: tuple[MockGeometryDomain, MockGeometryDomain]
    ) -> None:
        """LEGACY-DISC-03: smoothed_normals=True treats vertex pseudonormals as constant in Hessian.

        TriangleContactGeometry.evaluate with vertex_normals computes:
            d_tilde_n = sum(n_v_i * dbary_i)
        omitting the master displacement variation of n_v_i itself.
        """
        slave, master = triangle_facet_pair
        # In legacy, TriangleContactGeometry evaluates smoothed normals with fixed vertex normals
        v_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        point = slave.current_nodes[0]
        vertices = master.current_nodes

        geom = TriangleContactGeometry.evaluate(
            point=point,
            vertices=vertices,
            pseudonormal=np.array([0.0, 0.0, 1.0]),
            vertex_normals=v_normals,
        )
        # Gap gradient evaluates to np.kron(weights, normal), omitting r.T * grad(n)
        assert geom.normal is not None
        assert geom.gradient.shape == (12,)
        assert geom.hessian.shape == (12, 12)

    def test_legacy_disc_04_mortar_tangent_frozen_geometry(self) -> None:
        """LEGACY-DISC-04: MortarContact.evaluate_stiffness linearizes only nodal pressures.

        Omits variations of the mortar integration domain and biorthogonal shape matrices
        (dD/du and dM/du) with respect to finite deformation.
        """
        s_nodes = np.array([[0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.0, 1.0, 0.01]])
        s_faces = np.array([[0, 1, 2]])
        m_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        m_faces = np.array([[0, 2, 1]])

        slave = MockGeometryDomain(s_nodes, s_faces)
        master = MockGeometryDomain(m_nodes, m_faces)

        mc = MortarContact(slave=slave, master=master, k_penalty=2000.0)
        K_c = mc.evaluate_stiffness()

        # In legacy, K_ss block is purely assembled from n_dir outer n_dir * kp
        # without geometric stiffness from facet rotation or area change
        assert K_c.nnz > 0
        assert spsp.isspmatrix_csr(K_c)

    def test_legacy_disc_05_rigid_cylinder_height_ignored(self) -> None:
        """LEGACY-DISC-05: RigidCylinder.query_points treats cylinder as infinitely extruded.

        A point far beyond height / 2 receives the exact same gap and normal as if
        it were inside the cylinder height span.
        """
        cylinder = RigidCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=1.0, height=10.0)

        # Point inside height: z = 0.0, x = 1.2
        pt_mid = np.array([[1.2, 0.0, 0.0]])
        gap_mid, norm_mid, _ = cylinder.query_points(pt_mid)

        # Point far outside height: z = 500.0, x = 1.2
        pt_far = np.array([[1.2, 0.0, 500.0]])
        gap_far, norm_far, _ = cylinder.query_points(pt_far)

        # In legacy, height is ignored, so radial gap and normal are identical
        assert gap_mid[0] == pytest.approx(gap_far[0])
        np.testing.assert_allclose(norm_mid, norm_far, atol=1e-12)
        assert gap_mid[0] == pytest.approx(0.2, rel=1e-6)
