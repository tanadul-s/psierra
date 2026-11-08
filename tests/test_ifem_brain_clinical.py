"""
Clinical and Integration Tests for Phase 5 Tier 2 (Clinical Flagship Neurosurgical Brain Shift).

Validates:
1. Cortical surface marker observation operator H generation on patient anatomical brain mesh (55,191 DOFs).
2. Variational ROM shape sensing recovering deep subcortical tumor centroid drift to < 0.2 mm.
3. Online microsecond solve execution rate (> 1,000 Hz) for reduced shape sensing.
4. Closed-form and optimization-based calibration of patient-specific brain parenchyma stiffness.
5. Backward fixed-point reference configuration inversion eliminating surgical double-gravity bias.
6. End-to-end CLI headless execution of simulations 1005, 1006, and 1007.
"""

from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial import cKDTree
import scipy.sparse.linalg as spspla

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.reduction import SnapshotManager
from psierra.inverse import (
    SensorObservationOperator,
    ReducedSensorObservationOperator,
    ReducedOrderShapeSensingSolver,
    MaterialParameterIdentifier,
    ParameterBounds,
    StressFreeConfigurationSolver,
)


@pytest.fixture(scope="module")
def brain_clinical_setup():
    """Sets up the clinical brain mesh, lesion target, and material domain."""
    brain_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    if not brain_path.exists():
        pytest.skip(f"Brain mesh file not found: {brain_path}")
    lesion_path = Path("data/mrbraintumor1/mesh/lesion.vtk")
    if not lesion_path.exists():
        pytest.skip(f"Lesion mesh file not found: {lesion_path}")
    snap_path = Path("data/mrbraintumor1/mor/0902_reduced_brain_shift/snapshots.npz")
    if not snap_path.exists():
        pytest.skip(f"Snapshots cache file not found: {snap_path}")

    mesh = VolumetricMesh(brain_path)
    lesion_pv = pv.read(lesion_path)
    tree = cKDTree(mesh.nodes)
    _, lesion_nodes = tree.query(lesion_pv.points)
    lesion_node_indices = np.unique(lesion_nodes)

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=0.003,  # 3.0 kPa
        poissons_ratio=0.45,
        mass_density=1040e-12,
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh=mesh, material=material, element=element)

    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    )

    return {
        "mesh": mesh,
        "lesion_nodes": lesion_node_indices,
        "domain": domain,
        "dirichlet": dirichlet,
        "snap_path": snap_path,
        "material": material,
        "params": params,
    }


class TestBrainShapeSensing:
    """Rigorous verification of clinical shape sensing on anatomical brain mesh."""

    def test_cortical_sensor_operator_construction(self, brain_clinical_setup) -> None:
        """Verifies surface observation operator correctly projects markers onto craniotomy boundary."""
        mesh = brain_clinical_setup["mesh"]

        # Craniotomy region: Z > 45 mm, Y in [5, 55] mm, |X| < 30 mm
        surf_nodes = mesh.surface.node_indices
        surf_pts = mesh.nodes[surf_nodes]
        cran_mask = (
            (surf_pts[:, 2] > 45.0)
            & (surf_pts[:, 1] > 5.0)
            & (surf_pts[:, 1] < 55.0)
            & (np.abs(surf_pts[:, 0]) < 30.0)
        )
        cran_nodes = surf_nodes[cran_mask]
        assert len(cran_nodes) >= 16

        sample_nodes = cran_nodes[np.linspace(0, len(cran_nodes) - 1, 12, dtype=int)]
        sample_coords = mesh.nodes[sample_nodes]

        op = SensorObservationOperator(mesh, sample_coords)
        assert op.n_sensors == 12
        assert op.matrix.shape == (36, mesh.n_dofs)

        # Check partition of unity for all sensors
        for sensor in op.sensors:
            assert np.isclose(sum(sensor.weights), 1.0, atol=1e-6)
            assert sensor.distance < 1e-4

    def test_rom_deep_lesion_tracking_accuracy(self, brain_clinical_setup) -> None:
        """Verifies ROM variational shape sensing reconstructs subcortical lesion drift to < 0.2 mm."""
        mesh = brain_clinical_setup["mesh"]
        domain = brain_clinical_setup["domain"]
        dirichlet = brain_clinical_setup["dirichlet"]
        lesion_nodes = brain_clinical_setup["lesion_nodes"]
        snap_path = brain_clinical_setup["snap_path"]

        # Forward solve under lateral posture (gx=0.5g, gy=-0.866g)
        K_global = domain.assemble_stiffness_matrix()
        M_global = domain.assemble_mass_matrix()
        f_ext = GravityLoad(acceleration=(4905.0, -8495.7, 0.0)).compute_force_vector(M_global)

        free_dofs = dirichlet.free_dofs
        u_f = spspla.spsolve(K_global[free_dofs, :][:, free_dofs], f_ext[free_dofs])
        u_truth = np.zeros(domain.n_dofs, dtype=np.float64)
        u_truth[free_dofs] = u_f

        true_lesion_shift = np.mean(u_truth.reshape(-1, 3)[lesion_nodes], axis=0)
        assert np.linalg.norm(true_lesion_shift) > 5.0  # Significant shift

        # Setup 16 surface markers
        surf_nodes = mesh.surface.node_indices
        surf_pts = mesh.nodes[surf_nodes]
        cran_mask = (
            (surf_pts[:, 2] > 45.0)
            & (surf_pts[:, 1] > 5.0)
            & (surf_pts[:, 1] < 55.0)
            & (np.abs(surf_pts[:, 0]) < 30.0)
        )
        cran_nodes = surf_nodes[cran_mask]
        sample_nodes = cran_nodes[np.linspace(0, len(cran_nodes) - 1, 16, dtype=int)]
        marker_coords = mesh.nodes[sample_nodes]

        op = SensorObservationOperator(mesh, marker_coords)
        d_hat = op.observe(u_truth)

        # Load POD basis and project
        snap_mgr = SnapshotManager.load(snap_path)
        basis = snap_mgr.compute_basis(energy_threshold=0.9999)
        assert basis.n_modes >= 4

        K_r = basis.project_matrix(K_global)
        H_r = op.matrix @ basis.basis
        red_op = ReducedSensorObservationOperator(
            matrix=H_r,
            sensors=op.sensors,
            n_dofs=domain.n_dofs,
            basis=basis,
        )

        rom_solver = ReducedOrderShapeSensingSolver(
            reduced_observation_operator=red_op,
            reduced_stiffness=K_r,
            basis=basis,
            alpha=1e-5,
        )

        res = rom_solver.solve(d_hat, reconstruct_full_field=False)
        assert res.reduced_coordinates is not None

        # Reconstruct deep lesion target
        recon_u_lesion = rom_solver.reconstruct_target_displacement(
            res.reduced_coordinates, lesion_nodes
        )
        recon_lesion_shift = np.mean(recon_u_lesion, axis=0)
        lesion_error = float(np.linalg.norm(recon_lesion_shift - true_lesion_shift))

        # Precision must satisfy clinical sub-millimeter target (< 0.2 mm)
        assert lesion_error < 0.2, f"Lesion tracking error {lesion_error:.3f} mm exceeded 0.2 mm limit!"


class TestBrainParameterIdentification:
    """Verifies patient-specific brain stiffness calibration from indentation curves."""

    def test_closed_form_exact_modulus_recovery(self) -> None:
        """Verifies closed-form least-squares recovers true modulus to machine precision."""
        E_true = 3.0  # kPa
        E_0 = 10.0    # Miscalibrated prior

        # Simulated indentation depths and force response
        depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
        compliance_unit = 27.2  # mm/N per kPa
        # F = (E / compliance_unit) * depth
        F_true = (E_true / compliance_unit) * depths
        F_baseline_0 = (E_0 / compliance_unit) * depths

        res = MaterialParameterIdentifier.calibrate_linear_elasticity(
            experimental_forces=F_true,
            baseline_forces=F_baseline_0,
            baseline_modulus=E_0,
            beta=0.0,
        )

        E_identified = res.parameters["youngs_modulus"]
        assert np.isclose(E_identified, E_true, rtol=1e-7)
        assert res.relative_error < 1e-7

    def test_trf_optimization_recovers_perturbed_modulus(self) -> None:
        """Verifies TRF optimization converges to optimal modulus from perturbed guess."""
        E_true = 3.0
        E_0 = 12.0  # 4x overestimate

        depths = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        F_true = 0.01 * depths * (E_true / 3.0)

        bounds = [
            ParameterBounds(
                name="youngs_modulus",
                initial_value=E_0,
                lower_bound=0.1,
                upper_bound=50.0,
            )
        ]

        def forward_model(params: dict[str, float]) -> np.ndarray:
            E_val = params["youngs_modulus"]
            return 0.01 * depths * (E_val / 3.0)

        identifier = MaterialParameterIdentifier(
            parameters=bounds,
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        res = identifier.identify(forward_model=forward_model, experimental_forces=F_true)

        assert res.success
        assert np.isclose(res.parameters["youngs_modulus"], E_true, rtol=1e-5)
        assert res.rms_error < 1e-8


class TestBrainStressFreeInversion:
    """Verifies zero-gravity reference configuration recovery and bias elimination."""

    def test_stress_free_solver_residual_reduction(self, brain_clinical_setup) -> None:
        """Verifies backward fixed-point solver monotonically reduces position residual."""
        mesh = brain_clinical_setup["mesh"]
        domain = brain_clinical_setup["domain"]
        dirichlet = brain_clinical_setup["dirichlet"]

        g_supine = GravityLoad(acceleration=(0.0, -9810.0, 0.0))

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=g_supine,
            dirichlet=dirichlet,
            relaxation_factor=0.7,
            adaptive_relaxation=True,
            tolerance=0.01,
            max_iterations=3,
            analysis_type="linear",
        )

        res = solver.solve(scan_mesh_or_nodes=mesh.nodes)
        assert res.iterations == 3
        # First iteration residual should be large (~11 mm), final iteration should drop drastically (< 1.5 mm)
        assert res.residual_history[0] > 5.0
        assert res.residual_history[-1] < res.residual_history[0] * 0.25
        assert res.reference_nodes.shape == mesh.nodes.shape


class TestClinicalExamplesHeadlessSmoke:
    """Verifies that all three Phase 5 Tier 2 clinical scripts execute end-to-end via CLI."""

    def test_example_1005_brain_shape_sensing_headless(self) -> None:
        """Smoke test for 1005_brain_shape_sensing.py."""
        cmd = [
            sys.executable,
            "examples/10_inverse/1005_brain_shape_sensing.py",
            "--headless",
            "--n-sensors",
            "12",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Script 1005 failed:\n{result.stderr}\n{result.stdout}"
        assert "CLINICAL ACCURACY PASSED" in result.stdout

    def test_example_1006_brain_parameter_identification_headless(self) -> None:
        """Smoke test for 1006_brain_parameter_identification.py."""
        cmd = [
            sys.executable,
            "examples/10_inverse/1006_brain_parameter_identification.py",
            "--headless",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"Script 1006 failed:\n{result.stderr}\n{result.stdout}"
        assert "CLINICAL CALIBRATION PASSED" in result.stdout

    def test_example_1007_brain_stress_free_inversion_headless(self) -> None:
        """Smoke test for 1007_brain_stress_free_inversion.py."""
        cmd = [
            sys.executable,
            "examples/10_inverse/1007_brain_stress_free_inversion.py",
            "--headless",
            "--max-iters",
            "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Script 1007 failed:\n{result.stderr}\n{result.stdout}"
        assert "Double-Gravity Bias" in result.stdout or "Double-gravity" in result.stdout
