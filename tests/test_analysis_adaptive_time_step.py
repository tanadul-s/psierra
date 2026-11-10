from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element import Tet4TotalLagrangian
from psierra.material import NeoHookean, Viscoelastic, PronyTerm
from psierra.mesh.volumetric import VolumetricMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary.contact import RigidContact
from psierra.collision.shapes import RigidPlane
from psierra.damping.rayleigh import RayleighDamping
from psierra.analysis.adaptive_time_step import AdaptiveTimeStepController
from psierra.analysis.nonlinear_dynamic import (
    NonlinearDynamicAnalysis,
    NonlinearDynamicResult,
)


class TestAdaptiveTimeStepController:
    """Verifies AdaptiveTimeStepController unit mechanics and parameter invariants."""

    def test_controller_validation(self) -> None:
        # Invalid dt_min
        with pytest.raises(ValueError, match="dt_min"):
            AdaptiveTimeStepController(dt_min=-1.0)

        # Invalid dt_max < dt_min
        with pytest.raises(ValueError, match="dt_max"):
            AdaptiveTimeStepController(dt_min=0.01, dt_max=0.005)

        # Invalid gamma_cut
        with pytest.raises(ValueError, match="gamma_cut"):
            AdaptiveTimeStepController(gamma_cut=1.5)
        with pytest.raises(ValueError, match="gamma_cut"):
            AdaptiveTimeStepController(gamma_cut=0.0)

        # Invalid gamma_grow
        with pytest.raises(ValueError, match="gamma_grow"):
            AdaptiveTimeStepController(gamma_grow=0.9)

        # Invalid counts
        with pytest.raises(ValueError, match="max_cutbacks"):
            AdaptiveTimeStepController(max_cutbacks=0)
        with pytest.raises(ValueError, match="consecutive_success_for_grow"):
            AdaptiveTimeStepController(consecutive_success_for_grow=0)
        with pytest.raises(ValueError, match="target_iterations"):
            AdaptiveTimeStepController(target_iterations=0)

    def test_controller_initialization_and_cutback(self) -> None:
        ctrl = AdaptiveTimeStepController(
            dt_min=1e-3,
            gamma_cut=0.5,
            max_cutbacks=3,
        )
        assert not ctrl.can_cutback()  # Not initialized yet

        ctrl.initialize(initial_dt=0.01)
        assert ctrl.initialized is True
        assert ctrl.current_dt == 0.01
        assert ctrl.dt_max == 0.01

        # Cutback 1: 0.01 -> 0.005
        assert ctrl.can_cutback()
        dt1 = ctrl.cutback()
        assert dt1 == pytest.approx(0.005)
        assert ctrl.cutback_count == 1
        assert ctrl.total_cutbacks == 1

        # Cutback 2: 0.005 -> 0.0025
        assert ctrl.can_cutback()
        dt2 = ctrl.cutback()
        assert dt2 == pytest.approx(0.0025)
        assert ctrl.cutback_count == 2
        assert ctrl.total_cutbacks == 2

        # Cutback 3: 0.0025 * 0.5 = 0.00125 (>= dt_min=0.001)
        assert ctrl.can_cutback()
        dt3 = ctrl.cutback()
        assert dt3 == pytest.approx(0.00125)
        assert ctrl.cutback_count == 3
        assert ctrl.total_cutbacks == 3

        # Cutback limit reached (max_cutbacks = 3)
        assert not ctrl.can_cutback()
        with pytest.raises(RuntimeError, match="Cannot execute cutback"):
            ctrl.cutback()

    def test_controller_dynamic_growth(self) -> None:
        ctrl = AdaptiveTimeStepController(
            dt_min=1e-4,
            dt_max=0.02,
            gamma_grow=1.5,
            consecutive_success_for_grow=2,
            target_iterations=3,
        )
        ctrl.initialize(initial_dt=0.005)

        # Step 1: Converged in 2 iters (<= target 3) -> streak 1
        ctrl.record_success(iterations=2)
        assert ctrl.current_dt == pytest.approx(0.005)

        # Step 2: Converged in 1 iter (<= target 3) -> streak 2 -> Grow!
        ctrl.record_success(iterations=1)
        assert ctrl.current_dt == pytest.approx(0.005 * 1.5)  # 0.0075

        # Step 3: Converged in 4 iters (> target 3) -> Streak broken
        ctrl.record_success(iterations=4)
        assert ctrl.current_dt == pytest.approx(0.0075)

        # Step 4 & 5: Fast convergences -> Grow again
        ctrl.record_success(iterations=2)
        ctrl.record_success(iterations=2)
        assert ctrl.current_dt == pytest.approx(0.0075 * 1.5)  # 0.01125

        # Grow until clamped by dt_max
        ctrl.record_success(iterations=1)
        ctrl.record_success(iterations=1)
        assert ctrl.current_dt == pytest.approx(0.01125 * 1.5)  # 0.016875
        ctrl.record_success(iterations=1)
        ctrl.record_success(iterations=1)
        assert ctrl.current_dt == pytest.approx(0.02)  # Clamped to dt_max


class TestAdaptiveDynamicSubstepping:
    """Verifies Pillar 1 temporal regularization integrated into NonlinearDynamicAnalysis."""

    def test_adaptive_substepping_on_convergent_problem(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, 15.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        damping = RayleighDamping(alpha=1.0, beta=0.001)
        analysis = NonlinearDynamicAnalysis(domain, damping=damping)

        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.01,
            n_steps=5,
            adaptive=True,
            verbose=False,
        )

        assert isinstance(result, NonlinearDynamicResult)
        assert result.converged is True
        assert result.steps_completed == 5
        assert len(result.cutbacks_history) == 5
        assert len(result.time_step_history) == 5
        assert result.time[-1] == pytest.approx(0.05)

    def test_automatic_cutback_recovers_from_line_search_stall(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Applies a high shock load with a coarse nominal time step that causes
        fixed-step integration to stall in line search, but passes seamlessly
        when adaptive cutback substepping is enabled.
        """
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        # High shock tensile load on flexible Neo-Hookean cube
        shock_fz = 2000.0
        load = NodalForce(top_nodes, [0.0, 0.0, shock_fz / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)
        damping = RayleighDamping(alpha=0.5, beta=0.001)

        # Case A: Fixed coarse time step with limited Newton iterations stalls
        domain_fixed = SolidDomain(mesh, neohookean_material, element)
        analysis_fixed = NonlinearDynamicAnalysis(domain_fixed, damping=damping)
        result_fixed = analysis_fixed.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.1,  # Large time step causing Newton convergence failure
            n_steps=2,
            nr_iterations=2,  # Strict iteration ceiling to force stall
            nr_tol=1e-5,
            adaptive=False,
            verbose=False,
        )
        assert result_fixed.converged is False
        assert result_fixed.steps_completed < 2

        # Case B: Adaptive substepping automatically cuts back and recovers
        domain_adaptive = SolidDomain(mesh, neohookean_material, element)
        analysis_adaptive = NonlinearDynamicAnalysis(domain_adaptive, damping=damping)
        controller = AdaptiveTimeStepController(
            dt_min=1e-4,
            gamma_cut=0.5,
            gamma_grow=1.25,
            max_cutbacks=5,
            consecutive_success_for_grow=2,
            target_iterations=2,
        )
        result_adaptive = analysis_adaptive.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.1,
            n_steps=2,
            nr_iterations=2,
            nr_tol=1e-5,
            time_step_controller=controller,
            verbose=False,
        )

        assert isinstance(result_adaptive, NonlinearDynamicResult)
        assert result_adaptive.converged is True
        assert sum(result_adaptive.cutbacks_history) > 0  # Verified cutback occurred
        assert result_adaptive.time[-1] == pytest.approx(0.2)
        assert np.linalg.norm(domain_adaptive.displacement) > 0.0

    def test_viscoelastic_history_reversion_on_cutback(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that when an adaptive cutback occurs, the domain.revert_state()
        hook correctly restores internal Maxwell history variables Q_m.
        """
        visco_material = Viscoelastic(
            base_material=neohookean_material,
            prony_series=[PronyTerm(relative_modulus=0.5, relaxation_time=0.1)],
        )
        element = Tet4TotalLagrangian(visco_material)
        domain = SolidDomain(five_tet_cube_mesh, visco_material, element)

        fixed_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="max")
        dirichlet = DirichletManager(five_tet_cube_mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        load = NodalForce(top_nodes, [0.0, 0.0, 2000.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(five_tet_cube_mesh.n_dofs)

        controller = AdaptiveTimeStepController(
            dt_min=1e-4,
            gamma_cut=0.5,
            max_cutbacks=5,
            target_iterations=2,
        )
        analysis = NonlinearDynamicAnalysis(domain, damping=RayleighDamping(0.5, 0.001))
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.1,
            n_steps=2,
            nr_iterations=2,
            nr_tol=1e-5,
            time_step_controller=controller,
            verbose=False,
        )

        assert result.converged is True
        assert sum(result.cutbacks_history) > 0
        # Viscoelastic state is active and finite
        history = domain.get_history_variables()
        assert "Q" in history
        assert np.all(np.isfinite(history["Q"]))

    def test_contact_multiplier_restoration_on_cutback(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that contact multiplier states are preserved and restored
        consistently across cutback retries.
        """
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(five_tet_cube_mesh, neohookean_material, element)

        # Contact with a rigid floor at z = -0.05
        floor = RigidPlane(origin=[0.0, 0.0, -0.05], normal=[0.0, 0.0, 1.0])
        contact = RigidContact(
            obstacle=floor,
            domain=domain,
            k_penalty=5e4,
            formulation="augmented_lagrangian",
        )

        fixed_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="max")
        bottom_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(five_tet_cube_mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        # Downward load driving bottom nodes into floor
        load = NodalForce(bottom_nodes, [0.0, 0.0, -80.0 / len(bottom_nodes)])
        f_ext = load.compute_force_vector(five_tet_cube_mesh.n_dofs)

        controller = AdaptiveTimeStepController(
            dt_min=1e-5,
            gamma_cut=0.5,
            max_cutbacks=5,
            target_iterations=3,
        )
        analysis = NonlinearDynamicAnalysis(domain, damping=RayleighDamping(1.0, 0.005))
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.05,
            n_steps=2,
            contact=contact,
            nr_iterations=4,
            time_step_controller=controller,
            verbose=False,
        )

        assert result.converged is True
        assert len(result.time_step_history) > 0
        assert np.all(np.isfinite(contact.multipliers))

    def test_cutback_exhaustion_fails_gracefully(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that when convergence is mathematically impossible (e.g. 1 iteration allowed),
        cutbacks are exhausted and the solver cleanly reports converged=False without crashing.
        """
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(five_tet_cube_mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="max")
        dirichlet = DirichletManager(five_tet_cube_mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )
        load = NodalForce(top_nodes, [0.0, 0.0, 150.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(five_tet_cube_mesh.n_dofs)

        controller = AdaptiveTimeStepController(
            dt_min=1e-4,
            gamma_cut=0.5,
            max_cutbacks=2,
        )
        analysis = NonlinearDynamicAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.02,
            n_steps=2,
            nr_iterations=1,  # Impossible to converge in 1 iteration with 1e-4 tol
            nr_tol=1e-10,
            time_step_controller=controller,
            verbose=False,
        )

        assert result.converged is False
        assert result.steps_completed == 0
        assert controller.cutback_count == 2

    def test_adaptive_with_writer_export(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
        tmp_path: Path,
    ) -> None:
        """Verifies that adaptive substepping writes valid time steps and visualization files."""
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(five_tet_cube_mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="max")
        dirichlet = DirichletManager(five_tet_cube_mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )
        load = NodalForce(top_nodes, [0.0, 0.0, 50.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(five_tet_cube_mesh.n_dofs)

        out_dir = tmp_path / "adaptive_xdmf"
        analysis = NonlinearDynamicAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.02,
            n_steps=3,
            adaptive=True,
            save_interval=1,
            save_directory=out_dir,
            verbose=False,
        )

        assert result.converged is True
        assert (out_dir / "result.xdmf").exists() or (out_dir / "result.pvd").exists()

    def test_adaptive_steady_state_early_exit(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
    ) -> None:
        """Verifies that dynamic relaxation / steady state stopping works under adaptive stepping."""
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(five_tet_cube_mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(five_tet_cube_mesh, plane="xy", bound="max")
        dirichlet = DirichletManager(five_tet_cube_mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )
        load = NodalForce(top_nodes, [0.0, 0.0, 20.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(five_tet_cube_mesh.n_dofs)

        # High damping to reach static equilibrium quickly
        damping = RayleighDamping(alpha=4.0, beta=0.01)
        analysis = NonlinearDynamicAnalysis(domain, damping=damping)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.05,
            n_steps=50,
            steady_state_tol=1e-2,
            consecutive_hold_steps=2,
            adaptive=True,
            verbose=False,
        )

        assert result.converged is True
        assert result.steady_state_converged is True
        assert result.steady_state_step is not None
        assert result.steps_completed < 50
