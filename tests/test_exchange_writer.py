from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import h5py

from psierra.mesh.volumetric import VolumetricMesh
from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.viscoelastic import Viscoelastic, PronyTerm
from psierra.element.tet4_linear import Tet4Linear
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.domain.solid import SolidDomain
from psierra.exchange import (
    SimulationWriter,
    SimulationCheckpoint,
    XdmfWriter,
    PvdWriter,
)


@pytest.fixture
def elastic_domain(five_tet_cube_mesh: VolumetricMesh) -> SolidDomain:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(1000.0, 0.3, 1e-3)
    mat = Hookean(params)
    elem = Tet4Linear(mat)
    return SolidDomain(five_tet_cube_mesh, mat, elem)


def test_xdmf_writer_file_creation_and_xml_structure(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """Verifies that XdmfWriter generates valid HDF5 and standard XDMF 3.0 XML."""
    out_prefix = tmp_path / "sim_xdmf"

    with SimulationWriter(out_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        assert writer.format == "xdmf"
        # Record 2 steps
        u0 = np.zeros((five_tet_cube_mesh.n_nodes, 3))
        writer.add_timestep(time=0.0, displacement=u0)

        elastic_domain.update_state(u=np.ones(five_tet_cube_mesh.n_dofs) * 0.05)
        writer.add_timestep(time=0.01, domain=elastic_domain)

    # Check files exist
    h5_file = tmp_path / "sim_xdmf.h5"
    xdmf_file = tmp_path / "sim_xdmf.xdmf"
    assert h5_file.exists()
    assert xdmf_file.exists()

    # Verify HDF5 content
    with h5py.File(h5_file, "r") as h5:
        assert "mesh/geometry" in h5
        assert "mesh/topology" in h5
        assert h5["mesh/geometry"].shape == (8, 3)
        assert h5["mesh/topology"].shape == (5, 4)
        assert "time_series/step_000000/displacement" in h5
        assert "time_series/step_000001/displacement" in h5

    # Verify XDMF XML content
    xml_text = xdmf_file.read_text()
    assert '<Xdmf Version="3.0">' in xml_text
    assert 'TopologyType="Tetrahedron"' in xml_text
    assert 'sim_xdmf.h5:/mesh/topology' in xml_text
    assert 'sim_xdmf.h5:/mesh/geometry' in xml_text
    assert 'Attribute Name="Displacement"' in xml_text
    assert 'Time Value="0.000000e+00"' in xml_text
    assert 'Time Value="1.000000e-02"' in xml_text


def test_pvd_writer_compatibility(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """Verifies that PvdWriter generates valid .pvd and .vtu files with identical API."""
    out_dir = tmp_path / "sim_pvd"

    with SimulationWriter(out_dir, mesh=five_tet_cube_mesh, format="pvd") as writer:
        assert writer.format == "pvd"
        u0 = np.zeros((five_tet_cube_mesh.n_nodes, 3))
        writer.add_timestep(time=0.0, displacement=u0)

        elastic_domain.update_state(u=np.ones(five_tet_cube_mesh.n_dofs) * 0.02)
        writer.add_timestep(time=0.01, domain=elastic_domain)

    pvd_file = out_dir / "result.pvd"
    assert pvd_file.exists()
    assert (out_dir / "step_00000.vtu").exists()
    assert (out_dir / "step_00001.vtu").exists()

    pvd_text = pvd_file.read_text()
    assert "<VTKFile type=\"Collection\"" in pvd_text
    assert "step_00000.vtu" in pvd_text
    assert "step_00001.vtu" in pvd_text


def test_checkpoint_save_and_restore(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """Verifies full checkpoint creation, HDF5 persistence, and exact state restoration."""
    out_prefix = tmp_path / "test_ckpt"

    u_test = np.linspace(0.01, 0.24, five_tet_cube_mesh.n_dofs)
    v_test = np.linspace(0.1, 2.4, five_tet_cube_mesh.n_dofs)
    a_test = np.linspace(1.0, 24.0, five_tet_cube_mesh.n_dofs)
    elastic_domain.update_state(u=u_test, v=v_test, a=a_test)

    with SimulationWriter(out_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        writer.write_checkpoint(
            step=10,
            time=0.05,
            dt=0.005,
            domain=elastic_domain,
            metadata={"solver": "NonlinearDynamic", "converged": True},
        )

    # Read checkpoint back
    loaded_ckpt = SimulationWriter.load_checkpoint(out_prefix)
    assert loaded_ckpt.step == 10
    assert loaded_ckpt.time == 0.05
    assert loaded_ckpt.dt == 0.005
    assert loaded_ckpt.metadata["solver"] == "NonlinearDynamic"
    np.testing.assert_allclose(loaded_ckpt.u, u_test)
    np.testing.assert_allclose(loaded_ckpt.v, v_test)
    np.testing.assert_allclose(loaded_ckpt.a, a_test)

    # Apply to a fresh domain
    fresh_params = IsotropicElasticParameters()
    fresh_params.set_from_youngs_poisson(1000.0, 0.3, 1e-3)
    fresh_mat = Hookean(fresh_params)
    fresh_domain = SolidDomain(five_tet_cube_mesh, fresh_mat, Tet4Linear(fresh_mat))

    loaded_ckpt.apply_to_domain(fresh_domain)
    np.testing.assert_allclose(fresh_domain.u, u_test)
    np.testing.assert_allclose(fresh_domain.v, v_test)
    np.testing.assert_allclose(fresh_domain.a, a_test)


def test_viscoelastic_history_checkpoint_and_resume(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh
) -> None:
    """Verifies that internal viscoelastic history variables (Q_m, S0) serialize and restore exactly."""
    out_prefix = tmp_path / "test_visco_ckpt"

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(1000.0, 0.3, 1e-3)
    base = NeoHookean(params)
    prony = [PronyTerm(0.4, 0.1), PronyTerm(0.2, 0.5)]
    mat = Viscoelastic(base, prony)
    elem = Tet4TotalLagrangian(mat)
    domain = SolidDomain(five_tet_cube_mesh, mat, elem)

    dt = 0.01
    domain.set_time_step(dt)

    # Simulate some strain steps to populate history tensors (10% elongation along X)
    u_disp = np.zeros_like(five_tet_cube_mesh.nodes)
    u_disp[:, 0] = five_tet_cube_mesh.nodes[:, 0] * 0.1
    domain.update_state(u=u_disp.flatten())
    domain.assemble_internal_forces()
    domain.commit_state()

    # Check that history is non-zero
    orig_history = domain.get_history_variables()
    assert "Q" in orig_history
    assert np.any(orig_history["Q"] != 0.0)

    # Save checkpoint
    with SimulationWriter(out_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        writer.write_checkpoint(
            step=1,
            time=0.01,
            dt=dt,
            domain=domain,
        )

    # Restore in fresh domain
    mat_restored = Viscoelastic(base, prony)
    elem_restored = Tet4TotalLagrangian(mat_restored)
    domain_restored = SolidDomain(five_tet_cube_mesh, mat_restored, elem_restored)

    ckpt = SimulationWriter.load_checkpoint(out_prefix)
    ckpt.apply_to_domain(domain_restored)

    restored_history = domain_restored.get_history_variables()
    np.testing.assert_allclose(restored_history["Q"], orig_history["Q"])
    np.testing.assert_allclose(restored_history["S0_prev"], orig_history["S0_prev"])


def test_simulation_writer_resume_and_append(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """
    Verifies that SimulationWriter can close, reopen in resume mode, append new steps,
    and update checkpoints and XDMF manifests cleanly.
    """
    out_prefix = tmp_path / "sim_resume_run"

    # Part 1: Run initial 3 steps
    with SimulationWriter(out_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        for step in range(3):
            t = step * 0.01
            elastic_domain.update_state(u=np.ones(five_tet_cube_mesh.n_dofs) * (0.01 * (step + 1)))
            writer.add_timestep(time=t, domain=elastic_domain)
        writer.write_checkpoint(step=2, time=0.02, dt=0.01, domain=elastic_domain)

    # Inspect initial run
    summary1 = SimulationWriter.inspect(out_prefix)
    assert summary1["steps_recorded"] == 3
    assert summary1["checkpoint_step"] == 2

    # Part 2: Resume and run additional 3 steps
    with SimulationWriter(out_prefix, mesh=five_tet_cube_mesh, format="xdmf", resume=True) as writer:
        assert writer.step_count == 3
        for step in range(3, 6):
            t = step * 0.01
            elastic_domain.update_state(u=np.ones(five_tet_cube_mesh.n_dofs) * (0.01 * (step + 1)))
            writer.add_timestep(time=t, domain=elastic_domain)
        writer.write_checkpoint(step=5, time=0.05, dt=0.01, domain=elastic_domain)

    # Inspect resumed run
    summary2 = SimulationWriter.inspect(out_prefix)
    assert summary2["steps_recorded"] == 6
    assert summary2["checkpoint_step"] == 5

    # Check XDMF XML has all 6 steps
    xdmf_text = (tmp_path / "sim_resume_run.xdmf").read_text()
    for s in range(6):
        assert f"Step_{s:06d}" in xdmf_text


def test_pvd_and_xdmf_dynamic_fields_parity(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """Verifies that both PVD/VTU and XDMF record Velocity and Acceleration point data."""
    import pyvista as pv

    u = np.ones(five_tet_cube_mesh.n_dofs) * 0.01
    v = np.ones(five_tet_cube_mesh.n_dofs) * 0.5
    a = np.ones(five_tet_cube_mesh.n_dofs) * 2.0
    elastic_domain.update_state(u=u, v=v, a=a)

    # 1. PVD
    pvd_dir = tmp_path / "parity_pvd"
    with SimulationWriter(pvd_dir, mesh=five_tet_cube_mesh, format="pvd") as writer:
        writer.add_timestep(time=0.01, domain=elastic_domain)

    vtu_file = pvd_dir / "step_00000.vtu"
    assert vtu_file.exists()
    grid = pv.read(str(vtu_file))
    assert "Displacement" in grid.point_data
    assert "Velocity" in grid.point_data
    assert "Acceleration" in grid.point_data
    np.testing.assert_allclose(grid.point_data["Velocity"][0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(grid.point_data["Acceleration"][0], [2.0, 2.0, 2.0])

    # 2. XDMF
    xdmf_prefix = tmp_path / "parity_xdmf"
    with SimulationWriter(xdmf_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        writer.add_timestep(time=0.01, domain=elastic_domain)

    with h5py.File(xdmf_prefix.with_suffix(".h5"), "r") as h5:
        assert "time_series/step_000000/displacement" in h5
        assert "time_series/step_000000/velocity" in h5
        assert "time_series/step_000000/acceleration" in h5
        np.testing.assert_allclose(h5["time_series/step_000000/velocity"][0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(h5["time_series/step_000000/acceleration"][0], [2.0, 2.0, 2.0])


def test_step_0_zero_velocity_saved_when_dynamic(
    tmp_path: Path, five_tet_cube_mesh: VolumetricMesh, elastic_domain: SolidDomain
) -> None:
    """Verifies that at step 0 with zero velocity (rest) and active acceleration, Velocity is saved in both backends."""
    import pyvista as pv

    u0 = np.zeros(five_tet_cube_mesh.n_dofs)
    v0 = np.zeros(five_tet_cube_mesh.n_dofs)  # Zero velocity at rest!
    a0 = np.ones(five_tet_cube_mesh.n_dofs) * -9810.0  # Dynamic initial acceleration
    elastic_domain.update_state(u=u0, v=v0, a=a0)

    # 1. PVD backend
    pvd_dir = tmp_path / "zero_v_pvd"
    with SimulationWriter(pvd_dir, mesh=five_tet_cube_mesh, format="pvd") as writer:
        writer.add_timestep(time=0.0, domain=elastic_domain)

    vtu_0 = pvd_dir / "step_00000.vtu"
    assert vtu_0.exists()
    grid = pv.read(str(vtu_0))
    assert "Displacement" in grid.point_data
    assert "Velocity" in grid.point_data
    assert "Acceleration" in grid.point_data
    np.testing.assert_allclose(grid.point_data["Velocity"], np.zeros((five_tet_cube_mesh.n_nodes, 3)))
    np.testing.assert_allclose(grid.point_data["Acceleration"][0], [-9810.0, -9810.0, -9810.0])

    # 2. XDMF backend
    xdmf_prefix = tmp_path / "zero_v_xdmf"
    with SimulationWriter(xdmf_prefix, mesh=five_tet_cube_mesh, format="xdmf") as writer:
        writer.add_timestep(time=0.0, domain=elastic_domain)

    with h5py.File(xdmf_prefix.with_suffix(".h5"), "r") as h5:
        assert "time_series/step_000000/displacement" in h5
        assert "time_series/step_000000/velocity" in h5
        assert "time_series/step_000000/acceleration" in h5
        np.testing.assert_allclose(h5["time_series/step_000000/velocity"][:], np.zeros((five_tet_cube_mesh.n_nodes, 3)))

    xdmf_text = (tmp_path / "zero_v_xdmf.xdmf").read_text()
    assert 'Attribute Name="Velocity"' in xdmf_text
    assert 'Attribute Name="Acceleration"' in xdmf_text



