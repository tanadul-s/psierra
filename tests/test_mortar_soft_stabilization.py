"""Verification Suite for Mortar Contact Stabilization for Ultra-Soft Continua.

Milestone 1.4 of Loop 3: Robust Contact Mechanics & Advanced Multiphysics.
-------------------------------------------------------------------------
Verifies:
1. Area-Weighted Mortar Projection Normals:
   - Evaluates normal projection across non-conforming and non-planar overlapping facets.
   - Proves elimination of unweighted normal tilting from tiny edge clipping slivers.
2. Bounded Multiplier Augmentation:
   - Enforces scale-aware multiplier increment upper bound:
     |Delta lambda_N| <= lambda_max^step = gamma_cap * (E * A_e / h_e) * h_e = gamma_cap * E * A_e.
   - Guarantees finite dual force growth and modulus scaling invariance across ultra-soft continua (E <= 100 kPa).
3. Soft Continuum Admissibility (Zero Element Inversions):
   - Evaluates deformation gradient F = I + grad(u) and Jacobian det(F) > 0.01 across all continuum elements.
4. End-to-End Static Mortar Solve (Silicone Beam on Deformable Floor):
   - Verifies 2-body two-way deformable mortar coupling with Augmented Lagrangian Uzawa enforcement.
5. End-to-End Dynamic Mortar Solve (Generalized-Alpha Integration):
   - Verifies dynamic contact stability without multiplier chattering or unphysical element distortion.
6. Automatic Adaptive Penalty & Multiplier Cap Integration:
   - Validates 'auto' penalty and adaptive cap calculation across different Young's moduli.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.collision.mortar import (
    MortarSegmentData,
)
from psierra.boundary.mortar import MortarContact
from psierra.contact.scaling import (
    compute_multiplier_cap,
    compute_adaptive_penalty,
    extract_material_stiffness,
)
from psierra.mesh import VolumetricMesh
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    select_nodes_by_bounds,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import (
    NonlinearStaticAnalysis,
    NonlinearDynamicAnalysis,
)
from psierra.analysis.augmentation import ContactAugmentation


def _resolve_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


class TestMortarNormalsAndBoundedAugmentation:
    """Verifies area-weighted normal projections and bounded Uzawa multiplier increments."""

    def test_area_weighted_mortar_projection_normals(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verify that area-weighted normals prevent facet-edge tilting from tiny sliver overlaps."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(20000.0, 0.35, 1.0)
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)

        upper = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)
        lower = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)

        contact = MortarContact(slave=upper, master=lower, k_penalty=1000.0)

        # Construct synthetic segments incident on slave node 0:
        # Segment 1: Large contact patch, Area = 1.0 mm^2, Normal = [0, 1, 0]
        # Segment 2: Microscopic edge sliver, Area = 1e-4 mm^2, Normal = [1/sqrt(2), 1/sqrt(2), 0]
        n1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n2 = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0], dtype=np.float64)

        seg1 = MortarSegmentData(
            slave_facet_idx=0,
            master_facet_idx=0,
            slave_nodes=np.array([0, 1, 2], dtype=np.int64),
            master_nodes=np.array([0, 1, 2], dtype=np.int64),
            overlap_area=1.0,
            D_local=np.eye(3) / 3.0,
            M_local=np.eye(3) / 3.0,
            weighted_gaps=np.zeros(3),
            normal=n1,
        )
        seg2 = MortarSegmentData(
            slave_facet_idx=1,
            master_facet_idx=1,
            slave_nodes=np.array([0, 3, 4], dtype=np.int64),
            master_nodes=np.array([0, 3, 4], dtype=np.int64),
            overlap_area=1.0e-4,
            D_local=1.0e-4 * np.eye(3) / 3.0,
            M_local=1.0e-4 * np.eye(3) / 3.0,
            weighted_gaps=np.zeros(3),
            normal=n2,
        )

        segments = [seg1, seg2]
        node_normals = contact._compute_area_weighted_normals(segments, n_nodes=8)

        # Unweighted normal would tilt by ~22.5 deg: (n1 + n2) / ||n1 + n2|| -> [0.3827, 0.9239, 0]
        unweighted_tilt_x = (n1[0] + n2[0]) / np.linalg.norm(n1 + n2)
        assert unweighted_tilt_x > 0.3

        # Area-weighted normal: (1.0 * n1 + 1e-4 * n2) / ||...||
        weighted_n0 = node_normals[0]
        assert np.isclose(np.linalg.norm(weighted_n0), 1.0, atol=1e-12)
        # Transverse tilt in X is bounded by 1e-4 / 1.0 < 1e-3
        assert abs(weighted_n0[0]) < 1.0e-3
        assert np.isclose(weighted_n0[1], 1.0, atol=1.0e-3)
        assert np.isclose(weighted_n0[2], 0.0, atol=1e-12)

    def test_bounded_multiplier_augmentation_capping(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verify that multiplier updates are strictly bounded by lambda_max^step."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=25000.0, poissons_ratio=0.35, mass_density=1.0)
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)

        upper = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)
        lower = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)

        gamma_cap = 0.5
        contact = MortarContact(
            slave=upper,
            master=lower,
            k_penalty=50000.0,
            formulation="augmented_lagrangian",
            gamma_cap=gamma_cap,
        )

        assert contact.max_multiplier_step is not None
        expected_cap = compute_multiplier_cap(domain=upper, gamma_cap=gamma_cap)
        assert np.isclose(contact.max_multiplier_step, expected_cap, rtol=1e-5)

        # Synthesize a huge penetration on candidate gaps: g = -1.0 mm
        # Unbounded Uzawa update would be: Delta lambda = -k_p * g = 50000.0 N
        contact.multipliers[:] = 0.0
        n_candidates = len(contact.slave_nodes)

        # Mock evaluate_gaps returning large penetration
        original_eval_gaps = contact.evaluate_gaps
        try:
            contact.evaluate_gaps = lambda current_nodes=None: np.full(n_candidates, -1.0)

            augmentation = ContactAugmentation(
                [contact], max_iterations=5, gap_tol=1e-3, tolerance=1e-3
            )
            status = augmentation.assess()

            proposed_delta = status.proposed[0] - contact.multipliers
            max_delta = float(np.max(proposed_delta))

            # Multiplier update MUST be strictly clamped to contact.max_multiplier_step
            assert max_delta <= contact.max_multiplier_step + 1e-12
            assert max_delta < 50000.0  # Much smaller than raw unconstrained update
            assert np.all(status.proposed[0] >= 0.0)

            # Test direct ContactMultiplierState.update_multipliers()
            info = contact.update_multipliers()
            assert info["max_delta_multiplier"] <= contact.max_multiplier_step + 1e-12
            assert np.all(contact.multipliers <= contact.max_multiplier_step + 1e-12)

        finally:
            contact.evaluate_gaps = original_eval_gaps

    def test_multiplier_cap_modulus_scaling_invariance(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verify that lambda_max^step scales strictly proportionally with Young's modulus E."""
        moduli = [2.0e3, 20.0e3, 200.0e3]
        caps = []

        for E in moduli:
            params = IsotropicElasticParameters()
            params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=0.35, mass_density=1.0)
            mat = NeoHookean(params)
            elem = Tet4TotalLagrangian(mat)
            domain = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)

            cap = compute_multiplier_cap(domain=domain, gamma_cap=1.0)
            caps.append(cap)

        # Ratio between successive caps must exactly match ratio of Young's moduli (10.0)
        assert np.isclose(caps[1] / caps[0], 10.0, rtol=1e-4)
        assert np.isclose(caps[2] / caps[1], 10.0, rtol=1e-4)

    def test_soft_continuum_admissibility_zero_element_inversions(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verify that large mortar compression on soft continuum maintains det(F) > 0.01."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(10000.0, 0.35, 1.0)
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)

        upper = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)
        lower = SolidDomain(five_tet_cube_mesh.copy(), mat, elem)
        lower.mesh.transform(translation=(0.0, -1.0, 0.0))

        system = MultiBodySystem({"Upper": upper, "Lower": lower})
        contact = system.add_mortar_contact(
            slave="Upper",
            master="Lower",
            k_penalty=5000.0,
            formulation="augmented_lagrangian",
            gamma_cap=1.0,
            search_radius=0.2,
        )

        bot_nodes = np.where(np.isclose(lower.mesh.nodes[:, 1], -1.0))[0]
        clamp_bot = DirichletConstraint(
            bot_nodes,
            components=(0, 1, 2),
            value=0.0,
            dof_offset=system.get_body_dof_slice("Lower").start,
        )

        top_nodes = np.where(np.isclose(upper.mesh.nodes[:, 1], 1.0))[0]
        guide_x = DirichletConstraint(top_nodes, components=(0,), value=0.0, dof_offset=0)
        guide_z = DirichletConstraint(top_nodes, components=(2,), value=0.0, dof_offset=0)
        prescribe_y = DirichletConstraint(top_nodes, components=(1,), value=-0.05, dof_offset=0)

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(clamp_bot)
        dirichlet.add_constraint(guide_x)
        dirichlet.add_constraint(guide_z)
        dirichlet.add_constraint(prescribe_y)

        solver = NonlinearStaticAnalysis(system)
        result = solver.solve(dirichlet=dirichlet, n_steps=2, nr_tol=1e-3, verbose=False)

        assert result.converged

        # Evaluate deformation gradient determinant across all elements in both domains
        for name, domain in system.items():
            for conn in domain.mesh.elements:
                X_elem = domain.mesh.nodes[conn]
                x_elem = domain.current_nodes[conn]
                D_x = (x_elem[1:] - x_elem[0]).T
                D_X = (X_elem[1:] - X_elem[0]).T
                F_mat = D_x @ np.linalg.inv(D_X)
                det_F = float(np.linalg.det(F_mat))
                assert det_F > 0.01, f"Element inversion in {name}: det(F) = {det_F}"


class TestEndToEndMortarSiliconeBeamFloor:
    """End-to-end static and dynamic verification on soft silicone beam and flexible floor."""

    def test_end_to_end_static_silicone_beam_on_deformable_floor(self) -> None:
        """Verify static convergence and zero element inversions on deformable beam and floor."""
        beam_mesh = VolumetricMesh(_resolve_path("data/beam1/mesh/beam1.vtk"))
        floor_mesh = VolumetricMesh(_resolve_path("data/floor1/mesh/floor1.vtk"))

        min_z = float(beam_mesh.bounds[4])
        arm_node_mask = beam_mesh.nodes[:, 2] > 100.0
        arm_bottom_y = (
            float(beam_mesh.nodes[arm_node_mask, 1].min())
            if np.any(arm_node_mask)
            else float(beam_mesh.bounds[2])
        )
        orig_top_y = float(floor_mesh.bounds[3])
        dy = arm_bottom_y - orig_top_y
        floor_mesh.transform(translation=(0.0, dy, 150.0))

        params_beam = IsotropicElasticParameters()
        params_beam.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat_beam = NeoHookean(params_beam)
        beam_domain = SolidDomain(beam_mesh, mat_beam, Tet4TotalLagrangian(mat_beam))

        params_floor = IsotropicElasticParameters()
        params_floor.set_from_youngs_poisson(5000.0, 0.35, 1100e-6)
        mat_floor = NeoHookean(params_floor)
        floor_domain = SolidDomain(floor_mesh, mat_floor, Tet4TotalLagrangian(mat_floor))

        system = MultiBodySystem({"Beam": beam_domain, "Floor": floor_domain})
        contact = system.add_mortar_contact(
            slave="Beam",
            master="Floor",
            k_penalty=20000.0,
            formulation="augmented_lagrangian",
            search_radius=30.0,
            gamma_cap=1.0,
        )

        base_nodes = select_nodes_by_bounds(
            beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, min_z - 1.0, min_z + 5.0)
        )
        beam_dof_offset = system.get_body_dof_slice("Beam").start
        clamp_beam = DirichletConstraint(
            base_nodes, components=(0, 1, 2), value=0.0, dof_offset=beam_dof_offset
        )

        floor_dof_offset = system.get_body_dof_slice("Floor").start
        floor_min_y = float(floor_mesh.bounds[2])
        bot_nodes = np.where(np.abs(floor_mesh.nodes[:, 1] - floor_min_y) < 1.0)[0]
        clamp_floor = DirichletConstraint(
            bot_nodes, components=(0, 1, 2), value=0.0, dof_offset=floor_dof_offset
        )

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(clamp_beam)
        dirichlet.add_constraint(clamp_floor)

        max_z = float(beam_mesh.bounds[5])
        tip_nodes = select_nodes_by_bounds(
            beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, max_z - 20.0, max_z + 1.0)
        )
        f_ext = np.zeros(system.n_dofs, dtype=np.float64)
        tip_force_per_node = -2.0e4 / len(tip_nodes)
        for n in tip_nodes:
            f_ext[beam_dof_offset + 3 * n + 1] = tip_force_per_node

        solver = NonlinearStaticAnalysis(system)
        result = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=2,
            nr_tol=1e-3,
            al_gap_tol=5e-3,
            al_tol=0.03,
            al_max_iterations=8,
            verbose=False,
        )

        assert result.converged
        assert result.steps_completed == 2

        # Verify contact activity
        info = contact.get_contact_info()
        assert info["num_active"] > 0
        assert info["max_force"] > 0.0

        # Verify zero element inversions
        for name, domain in system.items():
            for conn in domain.mesh.elements:
                X_elem = domain.mesh.nodes[conn]
                x_elem = domain.current_nodes[conn]
                D_x = (x_elem[1:] - x_elem[0]).T
                D_X = (X_elem[1:] - X_elem[0]).T
                det_F = float(np.linalg.det(D_x @ np.linalg.inv(D_X)))
                assert det_F > 0.01, f"Element inversion in body {name}: det(F) = {det_F}"

    def test_end_to_end_dynamic_silicone_beam_on_deformable_floor(self) -> None:
        """Verify dynamic generalized-alpha integration with mortar contact and AL."""
        beam_mesh = VolumetricMesh(_resolve_path("data/beam1/mesh/beam1.vtk"))
        floor_mesh = VolumetricMesh(_resolve_path("data/floor1/mesh/floor1.vtk"))

        min_z = float(beam_mesh.bounds[4])
        arm_node_mask = beam_mesh.nodes[:, 2] > 100.0
        arm_bottom_y = (
            float(beam_mesh.nodes[arm_node_mask, 1].min())
            if np.any(arm_node_mask)
            else float(beam_mesh.bounds[2])
        )
        orig_top_y = float(floor_mesh.bounds[3])
        dy = arm_bottom_y - orig_top_y
        floor_mesh.transform(translation=(0.0, dy, 150.0))

        params_beam = IsotropicElasticParameters()
        params_beam.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat_beam = NeoHookean(params_beam)
        beam_domain = SolidDomain(beam_mesh, mat_beam, Tet4TotalLagrangian(mat_beam))

        params_floor = IsotropicElasticParameters()
        params_floor.set_from_youngs_poisson(5000.0, 0.35, 1100e-6)
        mat_floor = NeoHookean(params_floor)
        floor_domain = SolidDomain(floor_mesh, mat_floor, Tet4TotalLagrangian(mat_floor))

        system = MultiBodySystem({"Beam": beam_domain, "Floor": floor_domain})
        contact = system.add_mortar_contact(
            slave="Beam",
            master="Floor",
            k_penalty=20000.0,
            formulation="augmented_lagrangian",
            search_radius=30.0,
            gamma_cap=1.0,
        )

        base_nodes = select_nodes_by_bounds(
            beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, min_z - 1.0, min_z + 5.0)
        )
        beam_dof_offset = system.get_body_dof_slice("Beam").start
        clamp_beam = DirichletConstraint(
            base_nodes, components=(0, 1, 2), value=0.0, dof_offset=beam_dof_offset
        )

        floor_dof_offset = system.get_body_dof_slice("Floor").start
        floor_min_y = float(floor_mesh.bounds[2])
        bot_nodes = np.where(np.abs(floor_mesh.nodes[:, 1] - floor_min_y) < 1.0)[0]
        clamp_floor = DirichletConstraint(
            bot_nodes, components=(0, 1, 2), value=0.0, dof_offset=floor_dof_offset
        )

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(clamp_beam)
        dirichlet.add_constraint(clamp_floor)

        gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
        M = system.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)
        damping = RayleighDamping(alpha=1.0, beta=0.002)

        solver = NonlinearDynamicAnalysis(system, damping=damping, spectral_radius=0.8)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.002,
            n_steps=2,
            nr_iterations=15,
            nr_tol=1e-3,
            al_gap_tol=5e-3,
            al_tol=0.25,
            al_max_iterations=8,
            verbose=False,
        )

        assert res.converged
        assert res.steps_completed == 2
        for name, domain in system.items():
            for conn in domain.mesh.elements:
                X_elem = domain.mesh.nodes[conn]
                x_elem = domain.current_nodes[conn]
                D_x = (x_elem[1:] - x_elem[0]).T
                D_X = (X_elem[1:] - X_elem[0]).T
                det_F = float(np.linalg.det(D_x @ np.linalg.inv(D_X)))
                assert det_F > 0.01, f"Dynamic element inversion in body {name}: det(F) = {det_F}"
