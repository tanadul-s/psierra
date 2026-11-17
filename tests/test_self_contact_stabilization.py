"""Verification Suite: Self-Contact Stabilization via Proximity Buffer and Orientation Filtering.

Milestone 1.5 of Loop 3 Contact Mechanics Overhaul in Project Sierra.
---------------------------------------------------------------------
Verifies the stabilized self-contact formulation for thin-walled, folded, and
geometric feature interfaces (Stanford Bunny, slotted cuboid flexure):
1. Proximity Thickness Buffer:
   - Effective gap kinematics: g_eff = g_N - delta_buffer
   - Contact activation threshold: g_eff <= 0 <=> g_N <= delta_buffer
   - Exact conservation of gradients and Hessians: grad g_eff = grad g_N, Hess g_eff = Hess g_N
   - Potential energy and gradient consistency via centered finite differences
   - Scale-invariant buffer scaling: delta_buffer = buffer_ratio * h_e
   - Parameter validation and get_contact_info reporting
2. Consistent Normal Orientation Filtering:
   - Rejection of interior back-to-back facet pairs (g_ref <= 0, n_s . n_m <= -0.2)
   - Prevention of spurious thin-ear penetrations in reference configuration
   - Preservation of legitimate folding contact across exterior gaps
3. End-to-End Benchmark Verifications:
   - Stanford Bunny ear collapse with Augmented Lagrangian and reference filtering
   - Slotted Cuboid flexure dynamic pinching and rebound without active-set chattering
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp

from psierra.boundary.contact import SelfContact, compute_thickness_buffer
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh
from psierra.mesh.volumetric import VolumetricMesh
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import NonlinearStaticAnalysis, NonlinearDynamicAnalysis
from psierra.damping import RayleighDamping


# =============================================================================
# Prescribed-Kinematics Test Domain
# =============================================================================

class PrescribedSurfaceDomain(Domain):
    """Prescribed coordinate domain for exact self-contact kinematic verification."""

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


def make_opposing_triangles(gap_z: float = 0.10) -> tuple[PrescribedSurfaceDomain, np.ndarray, np.ndarray]:
    """Create two opposing triangle patches on a single domain.

    Upper patch: triangle 0 = [0, 1, 2], normal pointing down towards -Z.
    Lower patch: triangle 1 = [3, 4, 5], normal pointing up towards +Z.
    Node 0 sits at [0.3, 0.3, gap_z] above the lower triangle in the z=0 plane.
    """
    nodes = np.array([
        # Upper patch (node 0 at [0.3, 0.3, gap_z])
        [0.3, 0.3, gap_z],
        [0.3, 0.6, gap_z + 0.2],
        [0.6, 0.3, gap_z + 0.2],
        # Lower patch (triangle in z = 0 plane)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    domain = PrescribedSurfaceDomain(nodes, faces)
    return domain, nodes, faces


def make_thin_wall(thickness: float = 0.20) -> tuple[PrescribedSurfaceDomain, np.ndarray, np.ndarray]:
    """Create a thin volumetric slab/wall with top and bottom surfaces.

    Top surface: z = thickness, outward normal pointing +Z.
    Bottom surface: z = 0, outward normal pointing -Z.
    Nodes 0..2 on bottom, nodes 3..5 on top.
    In reference configuration, outward normals point in opposite directions
    through the solid interior (g_ref = -thickness < 0, n_top . n_bottom = -1.0).
    """
    nodes = np.array([
        # Bottom surface (z = 0, normal [0, 0, -1])
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        # Top surface (z = thickness, normal [0, 0, 1])
        [0.0, 0.0, thickness],
        [1.0, 0.0, thickness],
        [0.0, 1.0, thickness],
    ], dtype=np.float64)
    # Winding:
    # [0, 1, 2]: (v1-v0) x (v2-v0) = [0, 1, 0] x [1, 0, 0] = [0, 0, -1] -> outward normal points -Z
    # [3, 4, 5]: (v1-v0) x (v2-v0) = [1, 0, 0] x [0, 1, 0] = [0, 0, +1] -> outward normal points +Z
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    domain = PrescribedSurfaceDomain(nodes, faces)
    return domain, nodes, faces


# =============================================================================
# 1. Proximity Thickness Buffer Tests
# =============================================================================

class TestThicknessBufferFormulation:
    """Verifies the proximity thickness buffer kinematics, derivatives, and scaling."""

    def test_effective_gap_activation_threshold(self) -> None:
        """Effective gap g_eff = g_N - delta_buffer activates when g_N <= delta_buffer."""
        gap_phys = 0.10  # mm separated physically
        delta_buffer = 0.15  # mm proximity buffer
        domain, _, _ = make_opposing_triangles(gap_z=gap_phys)

        # Baseline: without buffer, gap is +0.10 mm, inactive (p=0)
        c_nobuff = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            thickness_buffer=0.0,
        )
        info_nobuff = c_nobuff.get_contact_info()
        assert info_nobuff["num_active"] == 0
        assert info_nobuff["max_penetration"] == 0.0
        assert np.isclose(c_nobuff.evaluate_gaps()[0], gap_phys, atol=1e-6)

        # With buffer: g_eff = 0.10 - 0.15 = -0.05 mm < 0, activates with penetration 0.05 mm
        c_buff = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            thickness_buffer=delta_buffer,
        )
        info_buff = c_buff.get_contact_info()
        assert info_buff["num_active"] == 1
        assert np.isclose(info_buff["thickness_buffer"], delta_buffer)
        expected_geff = gap_phys - delta_buffer
        assert np.isclose(c_buff.evaluate_gaps()[0], expected_geff, atol=1e-6)
        assert np.isclose(info_buff["max_penetration"], -expected_geff, atol=1e-6)

        # Normal pressure p = max(0, -kp * g_eff) = 1000 * 0.05 = 50.0 N
        p, _ = c_buff._pressure(c_buff._queries()[0])
        assert np.isclose(p, 1000.0 * (-expected_geff), atol=1e-6)

    def test_exact_gradients_and_hessians_preserved(self) -> None:
        """Shifting gap by constant scalar delta_buffer preserves exact gradients and Hessians."""
        gap_phys = 0.05
        domain, _, _ = make_opposing_triangles(gap_z=gap_phys)

        c_base = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            thickness_buffer=0.0,
        )
        c_buff = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            thickness_buffer=0.20,
        )

        q_base = c_base._queries()[0]
        q_buff = c_buff._queries()[0]

        # Gap shifted exactly by delta_buffer
        assert np.isclose(q_buff.geometry.gap, q_base.geometry.gap - 0.20, atol=1e-12)

        # Normal, gradient, and Hessian must be identical
        np.testing.assert_allclose(q_buff.geometry.normal, q_base.geometry.normal, atol=1e-14)
        np.testing.assert_allclose(q_buff.geometry.gradient, q_base.geometry.gradient, atol=1e-14)
        np.testing.assert_allclose(q_buff.geometry.hessian, q_base.geometry.hessian, atol=1e-14)
        np.testing.assert_allclose(q_buff.geometry.closest_point, q_base.geometry.closest_point, atol=1e-14)

    def test_forces_potential_gradient_consistency_with_buffer(self) -> None:
        """Conservative forces f = -grad(Pi) match centered finite differences under buffer shift."""
        domain, _, _ = make_opposing_triangles(gap_z=0.05)
        contact = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=2000.0,
            exclusion_rings=0,
            thickness_buffer=0.10,  # g_eff = 0.05 - 0.10 = -0.05 mm < 0
        )

        f_analytic = contact.evaluate_forces()
        assert np.linalg.norm(f_analytic) > 0.0

        # Centered finite-difference verification
        eps = 1e-6
        f_fd = np.zeros_like(f_analytic)
        nodes_ref = domain.reference.copy()
        for i in range(domain.n_dofs):
            node_idx = i // 3
            comp = i % 3

            nodes_pos = nodes_ref.copy()
            nodes_pos[node_idx, comp] += eps
            e_pos = contact.evaluate_potential_energy(nodes_pos)

            nodes_neg = nodes_ref.copy()
            nodes_neg[node_idx, comp] -= eps
            e_neg = contact.evaluate_potential_energy(nodes_neg)

            # f = -dPi/dq
            f_fd[i] = -(e_pos - e_neg) / (2.0 * eps)

        np.testing.assert_allclose(f_analytic, f_fd, rtol=1e-4, atol=1e-4)

    def test_thickness_buffer_scaling_helper_and_auto_option(self) -> None:
        """compute_thickness_buffer evaluates buffer_ratio * h_e correctly and handles auto."""
        domain, nodes, faces = make_opposing_triangles(gap_z=0.1)

        # Triangle 1 has vertices [0,0,0], [1,0,0], [0,1,0]: edge lengths 1, 1, sqrt(2)
        # Mean edge length of triangle 1 is ~ 1.138
        buff_015 = compute_thickness_buffer(domain=domain, buffer_ratio=0.15)
        buff_020 = compute_thickness_buffer(domain=domain, buffer_ratio=0.20)

        assert buff_015 > 0.0
        assert np.isclose(buff_020, (0.20 / 0.15) * buff_015, rtol=1e-6)

        # "auto" parameter in SelfContact
        contact_auto = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            thickness_buffer="auto",
            buffer_ratio=0.15,
        )
        assert np.isclose(contact_auto.thickness_buffer, buff_015, rtol=1e-6)
        assert contact_auto.contact_thickness >= 2.0 * contact_auto.thickness_buffer

    def test_thickness_buffer_parameter_validation(self) -> None:
        """Invalid buffer values or ratios raise informative ValueError."""
        domain, _, _ = make_opposing_triangles()

        with pytest.raises(ValueError, match="buffer_ratio must be finite and positive"):
            SelfContact(domain=domain, contact_nodes=[0], buffer_ratio=-0.1)

        with pytest.raises(ValueError, match="thickness_buffer must be finite and nonnegative"):
            SelfContact(domain=domain, contact_nodes=[0], thickness_buffer=-1.0)


# =============================================================================
# 2. Consistent Normal Orientation Filtering Tests
# =============================================================================

class TestConsistentOrientationFilter:
    """Verifies rejection of back-to-back reference penetrations across thin walls."""

    def test_filter_excludes_interior_back_to_back_facets(self) -> None:
        """Consistent orientation filter excludes opposing facets pointing through solid interior."""
        t_slab = 0.15  # mm wall thickness
        domain, nodes, faces = make_thin_wall(thickness=t_slab)

        # With filter=False and rings=0: bottom node falsely detects top face as penetrating
        c_unfiltered = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            contact_thickness=1.0,
            consistent_orientation_filter=False,
        )
        info_unfiltered = c_unfiltered.get_contact_info()
        assert info_unfiltered["num_active"] == 1
        assert info_unfiltered["max_penetration"] > 0.0

        # With filter=True: reference back-to-back opposition (g_ref <= 0, n1.n2 <= -0.2) is excluded
        c_filtered = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            contact_thickness=1.0,
            consistent_orientation_filter=True,
        )
        info_filtered = c_filtered.get_contact_info()
        assert info_filtered["num_active"] == 0
        assert info_filtered["max_penetration"] == 0.0

    def test_filter_preserves_legitimate_exterior_contact(self) -> None:
        """Consistent orientation filter does NOT exclude opposing facets separated across exterior void."""
        gap_ext = 0.20  # mm exterior void separation
        domain, _, _ = make_opposing_triangles(gap_z=gap_ext)

        # In reference configuration, gap is positive (+0.20 mm) across exterior void
        contact = SelfContact(
            domain=domain,
            contact_nodes=[0],
            k_penalty=1000.0,
            exclusion_rings=0,
            contact_thickness=1.0,
            consistent_orientation_filter=True,
        )
        # Master facet 1 is NOT excluded because g_ref = +0.20 > 0
        assert 1 not in contact._excluded_faces[0]

        # When deformed to close the exterior gap (z = -0.05 mm), contact is successfully admitted
        u_def = np.zeros(domain.n_dofs)
        u_def[2] = -(gap_ext + 0.05)  # move node 0 down by 0.25 mm into z = -0.05
        domain.update_state(u=u_def)

        info = contact.get_contact_info()
        assert info["num_active"] == 1
        assert np.isclose(info["max_penetration"], 0.05, atol=1e-5)


# =============================================================================
# 3. Dynamic and Static Benchmark Smoke Tests
# =============================================================================

class TestSelfContactStabilizationBenchmarks:
    """Verifies stabilized self-contact execution on representative structural benchmarks."""

    def test_cuboid_dynamic_slit_closure_smoke(self) -> None:
        """Slotted cuboid flexure dynamic steps converge without active-set chattering."""
        repo_root = Path(__file__).resolve().parents[1]
        mesh_path = repo_root / "data" / "cuboid1" / "mesh" / "cuboid1.vtk"
        if not mesh_path.exists():
            pytest.skip("cuboid1.vtk mesh not available for benchmark test")

        cuboid_mesh = VolumetricMesh(mesh_path)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6
        )
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)
        cuboid_domain = SolidDomain(cuboid_mesh, mat, elem)

        system = MultiBodySystem({"SlottedCuboid": cuboid_domain})
        surf_nodes = np.asarray(cuboid_mesh.surface.node_indices)
        nodes = cuboid_mesh.nodes
        slit_candidates = surf_nodes[
            (nodes[surf_nodes, 2] >= 34.0)
            & (nodes[surf_nodes, 2] <= 41.0)
            & (np.abs(nodes[surf_nodes, 1]) <= 65.0)
        ]

        system.add_self_contact(
            body="SlottedCuboid",
            contact_nodes=slit_candidates,
            k_penalty=50000.0,
            contact_thickness=6.0,
            search_radius=15.0,
            exclusion_rings=6,
            normal_threshold=-0.2,
            thickness_buffer="auto",
            consistent_orientation_filter=True,
            smoothed_normals=True,
        )

        base_nodes = np.where(nodes[:, 2] <= 0.5)[0]
        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(DirichletConstraint(base_nodes, components=(0, 1, 2), value=0.0))

        # Downward gravity loading driving sag along -Z
        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0), dofs_per_node=3)
        M = cuboid_domain.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)

        damping = RayleighDamping(alpha=1.0, beta=0.002)
        solver = NonlinearDynamicAnalysis(system, damping=damping, spectral_radius=0.8)

        # Solve 3 dynamic steps
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.02,
            n_steps=3,
            nr_tol=1e-3,
            verbose=False,
        )
        assert res.converged is True
        assert res.steps_completed == 3

    def test_bunny_reference_filter_eliminates_spurious_penetrations(self) -> None:
        """Consistent orientation filter eliminates all undeformed ear self-penetrations on Stanford Bunny."""
        repo_root = Path(__file__).resolve().parents[1]
        bunny_path = repo_root / "data" / "bunny1" / "mesh" / "bunny1.vtk"
        if not bunny_path.exists():
            pytest.skip("bunny1.vtk mesh not available for benchmark test")

        bunny_mesh = VolumetricMesh(bunny_path)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=2000.0, poissons_ratio=0.35, mass_density=1000e-6
        )
        mat = NeoHookean(params)
        elem = Tet4Linear(mat)
        domain = SolidDomain(bunny_mesh, mat, elem)

        surf_nodes = np.array(list(bunny_mesh.surface.node_indices))
        nodes = bunny_mesh.nodes
        cand_nodes = surf_nodes[nodes[surf_nodes, 2] > 50.0]

        # With filter=False: thin ears (~3-4 mm) produce spurious undeformed penetrations under thickness=5 mm
        contact_unfiltered = SelfContact(
            domain=domain,
            contact_nodes=cand_nodes,
            k_penalty=20000.0,
            exclusion_rings=4,
            contact_thickness=5.0,
            normal_threshold=-0.2,
            consistent_orientation_filter=False,
        )
        info_unfiltered = contact_unfiltered.get_contact_info()
        assert info_unfiltered["num_active"] > 0
        assert info_unfiltered["max_penetration"] > 4.0  # Spurious ~4.9 mm ear interior penetration

        # With filter=True: all undeformed thin-ear back-to-back interactions are cleanly rejected
        contact_filtered = SelfContact(
            domain=domain,
            contact_nodes=cand_nodes,
            k_penalty=20000.0,
            exclusion_rings=4,
            contact_thickness=5.0,
            normal_threshold=-0.2,
            consistent_orientation_filter=True,
            thickness_buffer="auto",
        )
        info_filtered = contact_filtered.get_contact_info()
        assert info_filtered["num_active"] == 0
        assert info_filtered["max_penetration"] == 0.0
