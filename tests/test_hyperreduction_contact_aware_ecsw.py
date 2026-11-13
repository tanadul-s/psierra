"""
Unit and Integration Tests for Contact-Aware ECSW & Enriched POD Basis (Method 2).
==================================================================================
Validates:
1. EnrichedPODBasis block-orthonormality, Krylov compliance modes, and partition properties.
2. ContactAwareECSWTrainingMatrix joint virtual work assembly and exact contact conservation.
3. ContactAwareECSWOptimizer contact-zone clustering and positive cubature weights.
4. ContactAwareReducedIntegrationDomain serialization roundtrip.
5. ContactAwareHyperReducedModel tangent stiffness symmetry and high-frequency dynamic solve.
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import scipy.sparse as spsp
import pytest

from psierra.mesh.volumetric import VolumetricMesh
from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.neohookean import NeoHookean
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.domain.solid import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.boundary.selector import select_nodes_by_plane
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.basis import EnrichedPODBasis, PODBasis
from psierra.hyperreduction.contact_ecsw import (
    ContactAwareReducedIntegrationDomain,
    ContactAwareECSWTrainingMatrix,
    ContactAwareECSWOptimizer,
    ContactAwareHyperReducedModel,
)
from psierra.hyperreduction.contact import HyperReducedRigidContact
from psierra.hyperreduction.solver import HyperReducedDynamicSolver
from psierra.collision.shapes import RigidPlane


@pytest.fixture
def beam_contact_setup() -> tuple[SolidDomain, DirichletManager, VolumetricMesh, np.ndarray]:
    """Sets up a cantilever beam with an identified tip contact interface."""
    mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip("beam1.vtk not found")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    # E = 21 kPa = 0.021 MPa, nu = 0.45, rho = 1.1e-9 tonne/mm³
    params.set_from_youngs_poisson(0.021, 0.45, 1.1e-9)
    material = NeoHookean(params)
    elem = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=elem)

    # Clamped root at Z = min(Z)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    # Candidate contact zone: bottom surface near the free tip (Z > 150 mm, Y < -10 mm)
    tip_bottom_mask = (mesh.nodes[:, 2] > 150.0) & (mesh.nodes[:, 1] < -10.0)
    contact_nodes = np.flatnonzero(tip_bottom_mask)
    assert len(contact_nodes) > 0, "No contact nodes found in tip region"

    return domain, dirichlet, mesh, contact_nodes


class TestEnrichedPODBasis:
    """Tests for EnrichedPODBasis construction, orthonormality, and subspace partitioning."""

    def test_orthonormality_and_partitioning(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(123)

        # Generate synthetic bulk displacement snapshots (e.g. bending modes)
        m_bulk = 8
        S_bulk = rng.standard_normal((domain.n_dofs, m_bulk))
        S_bulk[dirichlet.prescribed_dofs, :] = 0.0

        # Generate synthetic contact localized response snapshots
        m_contact = 4
        S_contact = rng.standard_normal((domain.n_dofs, m_contact)) * 0.1
        # Concentrate deformation near contact nodes
        contact_dofs = (3 * contact_nodes[:, None] + np.array([0, 1, 2])).ravel()
        S_contact[contact_dofs, :] += 2.0
        S_contact[dirichlet.prescribed_dofs, :] = 0.0

        basis = EnrichedPODBasis.from_snapshots_and_contact(
            bulk_snapshots=S_bulk,
            contact_snapshots=S_contact,
            energy_threshold_bulk=0.99,
            max_bulk_modes=4,
            energy_threshold_contact=0.95,
            max_contact_modes=2,
        )

        assert basis.n_bulk_modes > 0
        assert basis.n_contact_modes > 0
        assert basis.n_modes == basis.n_bulk_modes + basis.n_contact_modes

        # 1. Exact orthonormality: Phi^T @ Phi = I_r
        I_r = np.eye(basis.n_modes, dtype=np.float64)
        ortho_err = float(np.linalg.norm(basis.basis_matrix.T @ basis.basis_matrix - I_r, ord="fro"))
        assert ortho_err < 1e-12, f"Enriched basis orthonormality error {ortho_err:.2e} exceeded tolerance"

        # 2. Strict mutual orthogonality between bulk and contact sub-bases:
        # Phi_bulk^T @ Phi_contact == 0 to machine precision
        cross_block = basis.bulk_basis.T @ basis.contact_basis
        cross_err = float(np.linalg.norm(cross_block, ord="fro"))
        assert cross_err < 1e-12, f"Bulk-contact cross-orthogonality error {cross_err:.2e} exceeded tolerance"

    def test_krylov_compliance_mode_generation(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(456)

        S_bulk = rng.standard_normal((domain.n_dofs, 5))
        S_bulk[dirichlet.prescribed_dofs, :] = 0.0

        # Synthesize contact forces on contact nodes
        contact_dofs = (3 * contact_nodes[:, None] + np.array([0, 1, 2])).ravel()
        F_c = np.zeros((domain.n_dofs, 3), dtype=np.float64)
        F_c[contact_dofs[:10], 0] = -50.0  # Normal contact force
        F_c[contact_dofs[:10], 1] = 20.0
        F_c[contact_dofs[:10], 2] = -10.0

        # Linear stiffness matrix for compliance solves
        K_0 = domain.assemble_tangent_stiffness_matrix()

        basis = EnrichedPODBasis.from_snapshots_and_contact(
            bulk_snapshots=S_bulk,
            contact_forces=F_c,
            K_0=K_0,
            max_bulk_modes=3,
            max_contact_modes=2,
        )

        assert basis.n_bulk_modes == 3
        assert basis.n_contact_modes > 0
        assert basis.orthonormality_error() < 1e-12


class TestContactAwareECSWTrainingMatrix:
    """Tests for joint virtual work training matrix assembly and exact conservation."""

    def test_joint_assembly_and_exact_contact_conservation(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(789)

        r = 4
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        basis = PODBasis(
            basis=Phi,
            singular_values=np.ones(r),
            cumulative_energy=1.0,
            total_energy=float(r),
        )

        # Displacement snapshots
        u_snaps = [Phi @ np.array([0.1, -0.05, 0.05, 0.02]), Phi @ np.array([0.2, -0.1, 0.1, 0.04])]

        # Contact forces concentrated on contact nodes
        contact_dofs = (3 * contact_nodes[:, None] + np.array([0, 1, 2])).ravel()
        fc_1 = np.zeros(domain.n_dofs, dtype=np.float64)
        fc_1[contact_dofs[:15]] = -25.0
        fc_2 = np.zeros(domain.n_dofs, dtype=np.float64)
        fc_2[contact_dofs[5:20]] = -40.0
        contact_forces = [fc_1, fc_2]

        A_joint, b_joint, info = ContactAwareECSWTrainingMatrix.build_joint(
            domain=domain,
            basis=basis,
            snapshots=u_snaps,
            contact_forces=contact_forces,
            contact_nodes=contact_nodes,
            include_stiffness=False,
        )

        n_elems = domain.mesh.n_elements
        assert A_joint.shape[1] == n_elems
        assert info["n_contact_elements"] > 0

        # EXACT CONSERVATION: A_joint @ 1_{N_e} == b_joint identically
        b_recomputed = A_joint.sum(axis=1)
        np.testing.assert_allclose(
            b_joint,
            b_recomputed,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Joint virtual work vector does not match column sum of training matrix",
        )

        # INTERIOR ELEMENTS HAVE ZERO CONTACT VIRTUAL WORK:
        # Check that elements with no nodes on contact_nodes have zero in the contact rows
        c_elems_set = set(info["contact_elements"])
        bulk_elems = [e for e in range(n_elems) if e not in c_elems_set]
        assert len(bulk_elems) > 0

        n_vol_rows = info["n_vol_rows"]
        n_contact_rows = info["n_contact_rows"]
        A_contact_block = A_joint[n_vol_rows : n_vol_rows + n_contact_rows, :]

        # All bulk columns in A_contact_block must be identically 0.0
        bulk_contact_norm = np.linalg.norm(A_contact_block[:, bulk_elems])
        assert bulk_contact_norm == 0.0, f"Bulk elements have non-zero contact work: {bulk_contact_norm}"


class TestContactAwareECSWOptimizer:
    """Tests for contact boundary clustering and RID optimization."""

    def test_contact_zone_clustering_and_positive_weights(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(101)

        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        basis = PODBasis(basis=Phi, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=3.0)

        u_snap = Phi @ np.array([0.15, -0.08, 0.12])
        contact_dofs = (3 * contact_nodes[:, None] + np.array([0, 1, 2])).ravel()
        fc_snap = np.zeros(domain.n_dofs, dtype=np.float64)
        fc_snap[contact_dofs[:12]] = -35.0

        A_joint, b_joint, info = ContactAwareECSWTrainingMatrix.build_joint(
            domain=domain,
            basis=basis,
            snapshots=[u_snap],
            contact_forces=[fc_snap],
            contact_nodes=contact_nodes,
            include_stiffness=True,
        )

        optimizer = ContactAwareECSWOptimizer(
            tolerance=0.05,
            contact_priority_weight=2.0,
            min_contact_elements=2,
            verbose=False,
        )

        rid = optimizer.optimize(
            A=A_joint,
            b=b_joint,
            contact_elements=info["contact_elements"],
        )

        assert isinstance(rid, ContactAwareReducedIntegrationDomain)
        assert rid.n_selected > 0
        assert np.all(rid.weights > 0.0), "All cubature weights must be strictly positive"
        assert rid.relative_error <= 0.05

        # Verify contact clustering
        assert rid.n_contact_selected >= 2, f"Expected >= 2 contact elements, got {rid.n_contact_selected}"
        assert rid.contact_cluster_ratio > 0.0
        assert len(rid.contact_element_indices) + len(rid.bulk_element_indices) == rid.n_selected

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_contact_ecsw_rid.npz"

            rid = ContactAwareReducedIntegrationDomain(
                element_indices=np.array([1, 4, 10, 25, 40], dtype=np.int64),
                weights=np.array([1.2, 0.8, 2.0, 1.5, 0.5], dtype=np.float64),
                n_total_elements=100,
                relative_error=0.012,
                contact_element_indices=np.array([1, 4], dtype=np.int64),
                bulk_element_indices=np.array([10, 25, 40], dtype=np.int64),
                n_contact_elements_in_full_mesh=15,
                metadata={"test": "persistence"},
            )

            saved = rid.save(archive_path)
            assert saved.is_file()

            loaded = ContactAwareReducedIntegrationDomain.load(archive_path)
            assert loaded.n_selected == 5
            assert loaded.n_contact_selected == 2
            assert loaded.n_bulk_selected == 3
            assert loaded.n_contact_elements_in_full_mesh == 15
            assert loaded.contact_cluster_ratio == 2 / 15
            np.testing.assert_array_equal(loaded.contact_element_indices, np.array([1, 4]))
            np.testing.assert_array_equal(loaded.weights, rid.weights)


class TestContactAwareHyperReducedModel:
    """Tests for ContactAwareHyperReducedModel evaluation, symmetry, and dynamic solve."""

    def test_tangent_stiffness_symmetry_under_contact(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(202)

        # Enriched POD Basis
        m_bulk = 6
        S_bulk = rng.standard_normal((domain.n_dofs, m_bulk))
        S_bulk[dirichlet.prescribed_dofs, :] = 0.0
        contact_dofs = (3 * contact_nodes[:, None] + np.array([0, 1, 2])).ravel()
        S_contact = rng.standard_normal((domain.n_dofs, 3)) * 0.1
        S_contact[contact_dofs, :] += 1.5
        S_contact[dirichlet.prescribed_dofs, :] = 0.0

        basis = EnrichedPODBasis.from_snapshots_and_contact(
            bulk_snapshots=S_bulk,
            contact_snapshots=S_contact,
            max_bulk_modes=3,
            max_contact_modes=2,
        )

        # Contact-Aware RID
        u_snap = basis.basis_matrix @ np.array([0.1, -0.05, 0.08, 0.02, -0.01])
        fc_snap = np.zeros(domain.n_dofs, dtype=np.float64)
        fc_snap[contact_dofs[:10]] = -30.0

        A_joint, b_joint, info = ContactAwareECSWTrainingMatrix.build_joint(
            domain=domain,
            basis=basis,
            snapshots=[u_snap],
            contact_forces=[fc_snap],
            contact_nodes=contact_nodes,
            include_stiffness=True,
        )

        rid = ContactAwareECSWOptimizer(
            tolerance=0.05,
            contact_priority_weight=2.0,
            min_contact_elements=2,
            verbose=False,
        ).optimize(A_joint, b_joint, contact_elements=info["contact_elements"])

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        # Rigid floor located under the beam
        obstacle = RigidPlane(origin=(0.0, -20.0, 0.0), normal=(0.0, 1.0, 0.0))

        # Build contact-aware hyper-reduced model
        model = ContactAwareHyperReducedModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            damping=RayleighDamping(0.5, 0.001),
            f_ext_base=f_grav,
            surface_nodes=mesh.surface.node_indices,
        )

        contact = HyperReducedRigidContact(
            model=model,
            obstacle=obstacle,
            k_penalty=50.0,
            formulation="augmented_lagrangian",
        )
        model.contact = contact

        assert model.is_enriched

        # Evaluate at state with active contact penetration
        u_r_penetrating = np.array([0.5, -2.0, 0.5, 0.2, -0.1])
        R_r, K_r = model.evaluate_total_residual_and_tangent(u_r_penetrating)

        assert R_r.shape == (basis.n_modes,)
        assert K_r.shape == (basis.n_modes, basis.n_modes)

        # STRICT TANGENT STIFFNESS SYMMETRY: K_r = K_r^T to machine precision
        np.testing.assert_allclose(
            K_r,
            K_r.T,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Hyper-reduced tangent stiffness matrix is not symmetric",
        )

        # Positive semi-definiteness
        eigenvals = np.linalg.eigvalsh(K_r)
        assert np.all(eigenvals >= -1e-10), f"Negative eigenvalue detected in tangent stiffness: {eigenvals.min()}"

    def test_dynamic_solve_with_contact_awareness(self, beam_contact_setup) -> None:
        domain, dirichlet, mesh, contact_nodes = beam_contact_setup
        rng = np.random.default_rng(303)

        S_bulk = rng.standard_normal((domain.n_dofs, 5))
        S_bulk[dirichlet.prescribed_dofs, :] = 0.0

        basis = EnrichedPODBasis.from_snapshots_and_contact(
            bulk_snapshots=S_bulk,
            max_bulk_modes=3,
        )

        u_snap = basis.basis_matrix @ np.array([0.1, -0.05, 0.05])
        A_joint, b_joint, info = ContactAwareECSWTrainingMatrix.build_joint(
            domain=domain,
            basis=basis,
            snapshots=[u_snap],
            contact_nodes=contact_nodes,
            include_stiffness=True,
        )

        rid = ContactAwareECSWOptimizer(
            tolerance=0.08,
            contact_priority_weight=1.5,
            verbose=False,
        ).optimize(A_joint, b_joint, contact_elements=info["contact_elements"])

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        model = ContactAwareHyperReducedModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            damping=RayleighDamping(1.0, 0.002),
            f_ext_base=f_grav,
            surface_nodes=mesh.surface.node_indices,
        )

        obstacle = RigidPlane(origin=(0.0, -40.0, 0.0), normal=(0.0, 1.0, 0.0))
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=obstacle,
            k_penalty=20.0,
            formulation="penalty",
        )

        solver = HyperReducedDynamicSolver(
            model=model,
            contact=contact,
            dt=0.002,
            spectral_radius=0.8,
            nr_max_iter=15,
            nr_tol=1e-4,
        )

        for _ in range(10):
            u_r_step = solver.step()
            assert np.all(np.isfinite(u_r_step))

        stats = solver.get_stats()
        assert stats.steps_completed == 10
        assert stats.effective_hz > 100.0, f"Expected > 100 Hz solve frequency, got {stats.effective_hz:.1f} Hz"
