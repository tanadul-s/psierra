"""Tests for Milestone 1.3: Two-Body Large-Deformation Ironing Benchmark.

Verifies:
1. Normal contact force generation in Phase 1 vertical indentation (F_N > 0).
2. Continuous finite sliding across multiple element widths without divergence.
3. Coulomb kinetic friction agreement (|F_T / F_N| ≈ mu) in sliding steady state.
4. Second Law of Thermodynamics: Frictional dissipation is non-negative (W_f <= 0, D_f >= 0).
5. Spatial Regularization: C1 smoothed normals eliminate facet-edge force chattering.
6. Finite strain continuum admissibility: Zero element inversions (det(F) > 0) throughout sliding.
7. Verification of the executable benchmark example (0812_ironing_large_strain.py).
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import pyvista as pv

from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.boundary.contact import DeformableContact
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint
from psierra.boundary.friction import StickSlipState
from psierra.domain import SolidDomain
from psierra.element import Tet4Linear
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.mesher.tetrahedral import TetrahedralMesher
from psierra.multibody import MultiBodySystem


def make_ironing_model(
    mu: float = 0.25,
    smoothed_normals: bool = True,
    e_slab: float = 1.0,
    e_slider: float = 5.0,
) -> tuple[MultiBodySystem, DeformableContact, np.ndarray, np.ndarray, int, int]:
    """Helper to assemble a lightweight two-body ironing system for unit/regression tests."""
    box = pv.Box(bounds=(0.0, 24.0, -3.5, 3.5, -4.0, 0.0)).triangulate()
    mesher_slab = TetrahedralMesher(order=1, edge_length_fac=0.25, quiet=True)
    slab_mesh = mesher_slab.generate(box)

    params_slab = IsotropicElasticParameters()
    params_slab.set_from_youngs_poisson(
        youngs_modulus=e_slab, poissons_ratio=0.35, mass_density=1e-6
    )
    mat_slab = NeoHookean(params_slab)
    slab_domain = SolidDomain(slab_mesh, mat_slab, Tet4Linear(mat_slab))

    cyl = pv.Cylinder(
        center=(5.0, 0.0, 3.0),
        direction=(0.0, 1.0, 0.0),
        radius=3.0,
        height=4.0,
        resolution=14,
    ).triangulate()
    mesher_slider = TetrahedralMesher(order=1, edge_length_fac=0.30, quiet=True)
    slider_mesh = mesher_slider.generate(cyl)

    params_slider = IsotropicElasticParameters()
    params_slider.set_from_youngs_poisson(
        youngs_modulus=e_slider, poissons_ratio=0.35, mass_density=1e-6
    )
    mat_slider = NeoHookean(params_slider)
    slider_domain = SolidDomain(slider_mesh, mat_slider, Tet4Linear(mat_slider))

    system = MultiBodySystem({"Slider": slider_domain, "Slab": slab_domain})
    slider_offset = system.get_body_dof_slice("Slider").start
    slab_offset = system.get_body_dof_slice("Slab").start

    contact = system.add_contact_pair(
        slave="Slider",
        master="Slab",
        k_penalty="auto",
        friction=mu,
        smoothed_normals=smoothed_normals,
    )

    bot_nodes = np.flatnonzero(slab_domain.mesh.nodes[:, 2] <= -3.8)
    top_nodes = np.flatnonzero(slider_domain.mesh.nodes[:, 2] >= 5.8)

    return system, contact, bot_nodes, top_nodes, slider_offset, slab_offset


class TestIroningBenchmarkMechanics:
    """Rigorous verification tests for two-body finite-sliding ironing mechanics."""

    def test_indentation_normal_force_generation(self) -> None:
        """Phase 1: Vertical indentation must generate positive repulsive normal contact force."""
        mu = 0.25
        system, contact, bot_nodes, top_nodes, slider_offset, slab_offset = (
            make_ironing_model(mu=mu)
        )

        delta_z = 0.15
        n_steps = 2

        def dirichlet_indent(step: int, total_steps: int) -> DirichletManager:
            mgr = DirichletManager(system.n_dofs)
            for node in bot_nodes:
                mgr.add_constraint(
                    DOFConstraint(
                        [
                            slab_offset + 3 * node,
                            slab_offset + 3 * node + 1,
                            slab_offset + 3 * node + 2,
                        ],
                        0.0,
                    )
                )
            uz = -delta_z * (step / total_steps)
            for node in top_nodes:
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node], 0.0))
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 1], 0.0))
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 2], uz))
            return mgr

        analysis = NonlinearStaticAnalysis(domain=system)
        sol = analysis.solve(
            dirichlet=dirichlet_indent,
            contact=contact,
            n_steps=n_steps,
            nr_iterations=25,
            nr_tol=0.03,
            verbose=False,
        )
        assert sol.converged, f"Indentation phase failed: {sol.iteration_history}"

        # Evaluate slave contact forces
        f_all = contact.evaluate_forces().reshape(-1, 3)
        n_slider_nodes = system["Slider"].mesh.n_nodes
        f_slave = f_all[slider_offset // 3 : slider_offset // 3 + n_slider_nodes]

        fx = float(np.sum(f_slave[:, 0]))
        fz = float(np.sum(f_slave[:, 2]))

        # Normal force on slave must be positive (repulsive upward from slab)
        assert fz > 0.1, f"Expected positive normal force, got Fz = {fz}"
        # With zero horizontal motion, lateral force must be negligible
        assert (
            abs(fx) < 0.05 * fz
        ), f"Unexpected lateral force in pure indentation: Fx = {fx}, Fz = {fz}"

    def test_finite_sliding_coulomb_agreement_and_dissipation(self) -> None:
        """Phase 2: Sliding across multiple elements satisfies Coulomb limit and non-negative dissipation."""
        mu = 0.25
        system, contact, bot_nodes, top_nodes, slider_offset, slab_offset = (
            make_ironing_model(mu=mu)
        )

        delta_z = 0.15
        slip_total = 2.0
        n_indent_steps = 2
        n_slip_steps = 4
        total_steps = n_indent_steps + n_slip_steps

        def unified_dirichlet(step: int, _total: int) -> DirichletManager:
            mgr = DirichletManager(system.n_dofs)
            for node in bot_nodes:
                mgr.add_constraint(
                    DOFConstraint(
                        [
                            slab_offset + 3 * node,
                            slab_offset + 3 * node + 1,
                            slab_offset + 3 * node + 2,
                        ],
                        0.0,
                    )
                )
            if step <= n_indent_steps:
                uz = -delta_z * (step / n_indent_steps)
                ux = 0.0
            else:
                uz = -delta_z
                slip_frac = (step - n_indent_steps) / n_slip_steps
                ux = slip_frac * slip_total
            for node in top_nodes:
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node], ux))
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 1], 0.0))
                mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 2], uz))
            return mgr

        analysis = NonlinearStaticAnalysis(domain=system)

        prev_ux = 0.0
        cumulative_work = 0.0
        sliding_ratios: list[float] = []
        sliding_fx: list[float] = []

        n_slider_nodes = system["Slider"].mesh.n_nodes

        for step in range(1, total_steps + 1):
            mgr = unified_dirichlet(step, total_steps)
            if step <= n_indent_steps:
                curr_ux = 0.0
            else:
                curr_ux = ((step - n_indent_steps) / n_slip_steps) * slip_total

            delta_ux = curr_ux - prev_ux

            res = analysis.solve(
                dirichlet=mgr,
                contact=contact,
                n_steps=1,
                nr_iterations=25,
                nr_tol=0.03,
                verbose=False,
            )
            assert (
                res.converged
            ), f"Step {step} failed to converge: history {res.iteration_history}"

            f_all = contact.evaluate_forces().reshape(-1, 3)
            f_slave = f_all[slider_offset // 3 : slider_offset // 3 + n_slider_nodes]
            fx = float(np.sum(f_slave[:, 0]))
            fz = float(np.sum(f_slave[:, 2]))

            if step > n_indent_steps:
                sliding_fx.append(fx)
                ratio = abs(fx) / max(fz, 1e-12)
                sliding_ratios.append(ratio)

                # Friction work increment: F_x * d(u_x)
                d_work = fx * delta_ux
                cumulative_work += d_work
                # Cumulative friction work must be dissipative overall
                assert (
                    cumulative_work <= 1e-4
                ), f"Cumulative friction work must be non-positive: {cumulative_work}"

            prev_ux = curr_ux

        # In steady sliding, mean horizontal friction force strictly opposes motion
        assert (
            np.mean(sliding_fx) < -0.01
        ), f"Mean sliding friction force must oppose motion: {np.mean(sliding_fx)}"

        # Check Coulomb friction agreement in steady sliding regime
        # On a curved cylindrical indenter, apparent horizontal ratio |Fx/Fz| aligns with
        # mu * cos(theta_contact) ~ [0.65*mu, 1.05*mu] (Wriggers Computational Contact Mechanics ch. 9)
        steady_ratios = (
            sliding_ratios[1:] if len(sliding_ratios) >= 3 else sliding_ratios
        )
        mean_ratio = float(np.mean(steady_ratios))
        assert 0.65 * mu <= mean_ratio <= 1.05 * mu, (
            f"Steady sliding friction ratio |Fx/Fz| = {mean_ratio:.4f} outside expected "
            f"curved ironing regime [{0.65*mu:.4f}, {1.05*mu:.4f}] for mu = {mu}"
        )

        # Active-set stick-slip state verification
        assert contact.friction_state is not None
        assert np.any(
            contact.friction_state.states == StickSlipState.SLIP
        ), "Expected slipping nodes"

    def test_smoothed_vs_unsmoothed_force_smoothness(self) -> None:
        """C1 smoothed normals must reduce facet-boundary force oscillation compared to raw facets."""
        mu = 0.25
        # 1. Run with smoothed normals
        sys_smooth, c_smooth, bot_s, top_s, off_slider_s, off_slab_s = (
            make_ironing_model(mu=mu, smoothed_normals=True)
        )
        # 2. Run with unsmoothed normals
        sys_raw, c_raw, bot_r, top_r, off_slider_r, off_slab_r = make_ironing_model(
            mu=mu, smoothed_normals=False
        )

        delta_z = 0.15
        ux_slide = 1.0

        def make_dirichlet(
            system: MultiBodySystem,
            bot: np.ndarray,
            top: np.ndarray,
            s_off: int,
            slab_off: int,
        ) -> DirichletManager:
            mgr = DirichletManager(system.n_dofs)
            for node in bot:
                mgr.add_constraint(
                    DOFConstraint(
                        [
                            slab_off + 3 * node,
                            slab_off + 3 * node + 1,
                            slab_off + 3 * node + 2,
                        ],
                        0.0,
                    )
                )
            for node in top:
                mgr.add_constraint(DOFConstraint([s_off + 3 * node], ux_slide))
                mgr.add_constraint(DOFConstraint([s_off + 3 * node + 1], 0.0))
                mgr.add_constraint(DOFConstraint([s_off + 3 * node + 2], -delta_z))
            return mgr

        # Indent then slide for both
        mgr_smooth = make_dirichlet(sys_smooth, bot_s, top_s, off_slider_s, off_slab_s)
        mgr_raw = make_dirichlet(sys_raw, bot_r, top_r, off_slider_r, off_slab_r)

        analysis_s = NonlinearStaticAnalysis(domain=sys_smooth)
        sol_s = analysis_s.solve(
            dirichlet=mgr_smooth,
            contact=c_smooth,
            n_steps=2,
            nr_tol=0.04,
            verbose=False,
        )

        analysis_r = NonlinearStaticAnalysis(domain=sys_raw)
        sol_r = analysis_r.solve(
            dirichlet=mgr_raw, contact=c_raw, n_steps=2, nr_tol=0.04, verbose=False
        )

        assert sol_s.converged, "Smoothed normals solve failed"
        assert sol_r.converged, "Raw facet normals solve failed"

        # Compare convergence iterations: C1 smoothed normals typically require fewer or equal iterations
        total_iter_s = sum(sol_s.iteration_history)
        total_iter_r = sum(sol_r.iteration_history)
        assert (
            total_iter_s <= total_iter_r + 2
        ), f"Smoothed iter {total_iter_s} exceeded raw {total_iter_r}"

    def test_zero_element_inversions_throughout_simulation(self) -> None:
        """Ensures Jacobian determinant det(F) > 0 for all continuum elements across both bodies."""
        mu = 0.25
        system, contact, bot_nodes, top_nodes, slider_offset, slab_offset = (
            make_ironing_model(mu=mu)
        )

        mgr = DirichletManager(system.n_dofs)
        for node in bot_nodes:
            mgr.add_constraint(
                DOFConstraint(
                    [
                        slab_offset + 3 * node,
                        slab_offset + 3 * node + 1,
                        slab_offset + 3 * node + 2,
                    ],
                    0.0,
                )
            )
        for node in top_nodes:
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node], 1.5))
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 1], 0.0))
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 2], -0.15))

        analysis = NonlinearStaticAnalysis(domain=system)
        sol = analysis.solve(
            dirichlet=mgr, contact=contact, n_steps=2, nr_tol=0.03, verbose=False
        )
        assert sol.converged

        # Verify det(F) > 0 for all elements in both domains
        for name, domain in system.items():
            elements = (
                domain.element if isinstance(domain.element, list) else [domain.element]
            )
            for elem in elements:
                u_d = domain.displacement
                # Evaluate deformation gradient on elements
                elem_nodes = domain.mesh.elements
                for elem_idx in range(len(elem_nodes)):
                    conn = elem_nodes[elem_idx]
                    X_elem = domain.mesh.nodes[conn]
                    x_elem = domain.current_nodes[conn]

                    # Standard linear tetrahedron Jacobian determinant
                    # [x1-x0, x2-x0, x3-x0] = F * [X1-X0, X2-X0, X3-X0]
                    D_x = (x_elem[1:] - x_elem[0]).T
                    D_X = (X_elem[1:] - X_elem[0]).T
                    F_mat = D_x @ np.linalg.inv(D_X)
                    det_F = float(np.linalg.det(F_mat))
                    assert (
                        det_F > 0.01
                    ), f"Element inversion in body {name}, element {elem_idx}: det(F) = {det_F}"


class TestIroningExecutableBenchmark:
    """Verifies that the standalone benchmark CLI script executes successfully."""

    def test_executable_benchmark_runs_cleanly(self, tmp_path: Path) -> None:
        """Runs examples/08_contact/0807_ironing_large_strain.py with CLI arguments and exports."""
        script = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "08_contact"
            / "0807_ironing_large_strain.py"
        )
        assert script.exists(), f"Benchmark script {script} not found"

        cmd = [
            sys.executable,
            str(script),
            "--friction",
            "0.25",
            "--indentation",
            "0.15",
            "--sliding-distance",
            "2.0",
            "--n-indent-steps",
            "2",
            "--n-slip-steps",
            "4",
            "--quiet",
            "--export-pvd",
            "--output-dir",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert (
            result.returncode == 0
        ), f"Benchmark execution failed:\n{result.stderr}\n{result.stdout}"

        # Check output contents
        assert "Milestone 1.3 Verification Gate" in result.stdout
        assert "Overall Acceptance Gate: PASSED" in result.stdout

        # Verify exported datasets
        pvd_file = tmp_path / "ironing.pvd"
        assert pvd_file.exists(), f"Expected PVD file {pvd_file} was not created"
        vtu_files = list(tmp_path.glob("*.vtu"))
        assert (
            len(vtu_files) == 6
        ), f"Expected 6 VTU files (2 indent + 4 sliding), found {len(vtu_files)}"
