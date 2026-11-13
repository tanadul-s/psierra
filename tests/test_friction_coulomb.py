"""Automated Verification Suite for Exact Coulomb Stick-Slip Friction.
===================================================================
Tests 3D active-set Coulomb friction return-mapping, algorithmic consistent
tangents, stick-slip state transitions, hysteresis loops, action-reaction
linear momentum conservation, and solver integration with Rigid, Deformable,
and Mortar contact.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from psierra.boundary.friction import (
    StickSlipState,
    CoulombFrictionState,
    CoulombFrictionModel,
)
from psierra.boundary.contact import RigidContact, DeformableContact
from psierra.boundary.mortar import MortarContact
from psierra.collision.shapes import RigidPlane
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.material import NeoHookean, IsotropicElasticParameters
from psierra.element import Tet4TotalLagrangian
from psierra.analysis import NonlinearStaticAnalysis, NonlinearDynamicAnalysis
from psierra.multibody import MultiBodySystem
from psierra.boundary.dirichlet import DirichletConstraint, DOFConstraint, DirichletManager


def make_neohookean(E: float = 1e4, nu: float = 0.3) -> NeoHookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=nu, mass_density=1e-3)
    return NeoHookean(params)


def make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0)) -> VolumetricMesh:
    """Creates a 5-tet box mesh positioned at origin with given size."""
    o = np.array(origin, dtype=np.float64)
    s = np.array(size, dtype=np.float64)
    unit_nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    nodes = o + unit_nodes * s
    tets = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ]
    cells = []
    for tet in tets:
        cells.extend([4] + tet)
    cells_array = np.array(cells, dtype=np.int64)
    cell_types = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    ugrid = pv.UnstructuredGrid(cells_array, cell_types, nodes)
    return VolumetricMesh(ugrid)


# =============================================================================
# 1. Return-Mapping & Algorithmic Tangent Unit Tests
# =============================================================================

class TestCoulombFrictionReturnMapping:
    """Verifies the core active-set Coulomb friction return mapping algorithm."""

    def test_friction_inactive_when_no_pressure(self) -> None:
        """Friction force and stiffness must be zero when normal pressure <= 0 or mu = 0."""
        normal = np.array([0.0, 1.0, 0.0])
        delta_u = np.array([0.5, 0.0, 0.2])

        # p_N = 0
        f_T, K_TT, K_TN, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=0.0,
            normal=normal,
            delta_u_rel=delta_u,
            mu=0.5,
            k_T=1000.0,
            k_N=1000.0,
        )
        assert np.allclose(f_T, 0.0)
        assert np.allclose(K_TT, 0.0)
        assert np.allclose(K_TN, 0.0)
        assert state == StickSlipState.INACTIVE

        # mu = 0 (frictionless)
        f_T, K_TT, K_TN, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=100.0,
            normal=normal,
            delta_u_rel=delta_u,
            mu=0.0,
            k_T=1000.0,
            k_N=1000.0,
        )
        assert np.allclose(f_T, 0.0)
        assert np.allclose(K_TT, 0.0)
        assert np.allclose(K_TN, 0.0)
        assert state == StickSlipState.STICK

    def test_coulomb_stick_condition(self) -> None:
        """When ||f_T^trial|| <= mu * p_N, state must be STICK with f_T = -k_T * delta_g_T."""
        normal = np.array([0.0, 0.0, 1.0])
        p_N = 100.0
        mu = 0.5
        coulomb_limit = mu * p_N  # 50.0 N
        k_T = 1000.0  # N/mm
        k_N = 1000.0

        # Small tangential displacement: delta_u = [0.02, 0.01, 0.0]
        # ||delta_g_T|| = sqrt(0.02^2 + 0.01^2) = 0.02236 mm
        # ||f_T^trial|| = 1000 * 0.02236 = 22.36 N < 50.0 N (STICK)
        delta_u = np.array([0.02, 0.01, 0.0])

        f_T, K_TT, K_TN, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )

        assert state == StickSlipState.STICK
        expected_f_T = -k_T * np.array([0.02, 0.01, 0.0])
        assert np.allclose(f_T, expected_f_T, atol=1e-12)
        assert np.linalg.norm(f_T) < coulomb_limit

        # Tangential stiffness must be k_T * P_T
        P_T = np.eye(3) - np.outer(normal, normal)
        assert np.allclose(K_TT, k_T * P_T, atol=1e-12)
        # Normal cross-coupling is zero in stick
        assert np.allclose(K_TN, 0.0, atol=1e-12)

    def test_coulomb_slip_condition(self) -> None:
        """When ||f_T^trial|| > mu * p_N, state must be SLIP with ||f_T|| = mu * p_N."""
        normal = np.array([0.0, 0.0, 1.0])
        p_N = 100.0
        mu = 0.4
        coulomb_limit = mu * p_N  # 40.0 N
        k_T = 1000.0
        k_N = 1000.0

        # Large tangential displacement: delta_u = [0.1, 0.05, 0.0]
        # ||f_T^trial|| = 1000 * sqrt(0.01 + 0.0025) = 111.8 N > 40.0 N (SLIP)
        delta_u = np.array([0.1, 0.05, 0.0])

        f_T, K_TT, K_TN, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )

        assert state == StickSlipState.SLIP
        # Magnitude must exactly equal Coulomb limit
        assert np.isclose(np.linalg.norm(f_T), coulomb_limit, atol=1e-12)

        # Force direction must be collinear with opposing relative motion
        expected_dir = -delta_u[:2] / np.linalg.norm(delta_u[:2])
        assert np.isclose(f_T[0] / coulomb_limit, expected_dir[0], atol=1e-12)
        assert np.isclose(f_T[1] / coulomb_limit, expected_dir[1], atol=1e-12)
        assert np.isclose(f_T[2], 0.0, atol=1e-12)

    def test_zero_stiffness_along_sliding_direction(self) -> None:
        """In SLIP state, tangent stiffness along the sliding direction must be zero."""
        normal = np.array([0.0, 1.0, 0.0])
        p_N = 80.0
        mu = 0.3
        k_T = 500.0
        k_N = 500.0
        delta_u = np.array([0.2, 0.0, -0.15])

        f_T, K_TT, K_TN, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.SLIP

        # Verify K_TT @ s_dir == 0 (zero stiffness along sliding direction)
        stiffness_along_slide = K_TT @ s_dir
        assert np.allclose(stiffness_along_slide, 0.0, atol=1e-12)

    def test_algorithmic_tangent_finite_difference_consistency(self) -> None:
        """Verifies that K_TT matches the numerical directional derivative -df_T/d(u)."""
        normal = np.array([0.2, 0.9, 0.3])
        normal /= np.linalg.norm(normal)
        p_N = 120.0
        mu = 0.35
        k_T = 2000.0
        k_N = 1500.0
        delta_u = np.array([0.15, -0.05, 0.08])

        f_T, K_TT, K_TN, state, _ = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.SLIP

        # Perturbation directions in 3D
        eps = 1e-7
        for d in [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]:
            f_plus, _, _, _, _ = CoulombFrictionModel.return_mapping(
                p_N=p_N,
                normal=normal,
                delta_u_rel=delta_u + eps * d,
                mu=mu,
                k_T=k_T,
                k_N=k_N,
            )
            f_minus, _, _, _, _ = CoulombFrictionModel.return_mapping(
                p_N=p_N,
                normal=normal,
                delta_u_rel=delta_u - eps * d,
                mu=mu,
                k_T=k_T,
                k_N=k_N,
            )
            df_numeric = (f_plus - f_minus) / (2.0 * eps)
            # Contact tangent convention: K_TT = -df_T / du
            df_analytic = -K_TT @ d
            assert np.allclose(df_numeric, df_analytic, rtol=1e-4, atol=1e-6)

    def test_cross_coupling_tangent_finite_difference_consistency(self) -> None:
        """Verifies that K_TN matches the numerical derivative -df_T/d(u_N)."""
        normal = np.array([0.0, 1.0, 0.0])
        p_N = 100.0
        mu = 0.5
        k_T = 1000.0
        k_N = 1000.0
        delta_u = np.array([0.2, 0.0, 0.0])

        _, _, K_TN, state, _ = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.SLIP

        # Perturb normal penetration gap: g_N -> g_N + eps implies p_N -> p_N - k_N * eps
        eps = 1e-6
        dp_N = -k_N * eps  # change in p_N due to displacement delta_u_s = eps * normal
        f_plus, _, _, _, _ = CoulombFrictionModel.return_mapping(
            p_N=p_N + dp_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        f_minus, _, _, _, _ = CoulombFrictionModel.return_mapping(
            p_N=p_N - dp_N,
            normal=normal,
            delta_u_rel=delta_u,
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        df_numeric = (f_plus - f_minus) / (2.0 * eps)
        # K_TN contribution: -K_TN @ normal
        df_analytic = -K_TN @ normal
        assert np.allclose(df_numeric, df_analytic, rtol=1e-4, atol=1e-6)

    def test_friction_hysteretic_reversal_loop(self) -> None:
        """Reversing tangential displacement must produce elastic stick before reverse slip."""
        normal = np.array([0.0, 0.0, 1.0])
        p_N = 50.0
        mu = 0.4
        coulomb_limit = mu * p_N  # 20.0 N
        k_T = 500.0  # N/mm
        k_N = 500.0

        # Create friction state tracker for 1 node
        state_tracker = CoulombFrictionState(n_candidates=1)
        anchor = np.array([0.0, 0.0, 0.0])
        state_tracker.anchors[0] = anchor

        # Step 1: Forward slip to x = 0.1 mm
        x_pos = np.array([0.1, 0.0, 0.0])
        f_T, _, _, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=x_pos - state_tracker.anchors[0],
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.SLIP
        assert np.isclose(f_T[0], -coulomb_limit)
        # Update stick anchor to keep force at Coulomb limit
        elastic_slip = -(coulomb_limit / k_T) * s_dir
        state_tracker.anchors[0] = x_pos - elastic_slip

        # Step 2: Reverse motion slightly by 0.02 mm (x = 0.08 mm)
        # Displacement from updated anchor: must immediately be in STICK!
        x_pos = np.array([0.08, 0.0, 0.0])
        f_T, _, _, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=x_pos - state_tracker.anchors[0],
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.STICK
        # Resisting force magnitude has dropped below Coulomb limit (elastic unloading!)
        assert abs(f_T[0]) < coulomb_limit

        # Step 3: Continue reversing motion far in the opposite direction (x = -0.1 mm)
        # Must reach full reverse slip at +coulomb_limit (+20.0 N opposing negative velocity)
        x_pos = np.array([-0.1, 0.0, 0.0])
        f_T, _, _, state, s_dir = CoulombFrictionModel.return_mapping(
            p_N=p_N,
            normal=normal,
            delta_u_rel=x_pos - state_tracker.anchors[0],
            mu=mu,
            k_T=k_T,
            k_N=k_N,
        )
        assert state == StickSlipState.SLIP
        assert np.isclose(f_T[0], +coulomb_limit, atol=1e-12)


# =============================================================================
# 2. Linear Momentum Conservation Tests
# =============================================================================

class TestFrictionalMomentumConservation:
    """Verifies that tangential friction forces strictly conserve linear momentum."""

    def test_deformable_contact_friction_momentum_conservation(self) -> None:
        """Deformable-to-deformable friction must satisfy sum(f_s) + sum(f_m) = 0."""
        # Create two cubes, one above the other with slight overlap
        mesh_bottom = make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 5.0, 10.0))
        mesh_top = make_cube_mesh(origin=(2.0, 4.8, 2.0), size=(6.0, 5.0, 6.0))

        mat = make_neohookean(E=1e4, nu=0.3)
        slave_body = SolidDomain(mesh_top, mat, Tet4TotalLagrangian(mat))
        master_body = SolidDomain(mesh_bottom, mat, Tet4TotalLagrangian(mat))

        contact = DeformableContact(
            slave=slave_body,
            master=master_body,
            k_penalty=500.0,
            friction=0.4,
            slave_offset=0,
            master_offset=slave_body.n_dofs,
        )

        total_dofs = slave_body.n_dofs + master_body.n_dofs
        forces = contact.evaluate_forces(n_dofs=total_dofs)

        slave_f = forces[:slave_body.n_dofs].reshape(-1, 3)
        master_f = forces[slave_body.n_dofs:total_dofs].reshape(-1, 3)

        sum_slave = np.sum(slave_f, axis=0)
        sum_master = np.sum(master_f, axis=0)
        net_force = sum_slave + sum_master

        # Check total action-reaction balance
        assert np.linalg.norm(sum_slave) > 0.0  # Contact is active
        assert np.allclose(net_force, 0.0, atol=1e-12), f"Net interface force: {net_force}"

    def test_mortar_contact_friction_momentum_conservation(self) -> None:
        """Segment-to-segment mortar friction must satisfy sum(f_s) + sum(f_m) = 0."""
        # Non-conforming mesh interfaces
        mesh_bottom = make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 5.0, 10.0))
        mesh_top = make_cube_mesh(origin=(1.0, 4.8, 1.0), size=(8.0, 5.0, 8.0))

        mat = make_neohookean(E=1e4, nu=0.3)
        slave_body = SolidDomain(mesh_top, mat, Tet4TotalLagrangian(mat))
        master_body = SolidDomain(mesh_bottom, mat, Tet4TotalLagrangian(mat))

        contact = MortarContact(
            slave=slave_body,
            master=master_body,
            k_penalty=1000.0,
            friction=0.35,
            slave_offset=0,
            master_offset=slave_body.n_dofs,
        )

        total_dofs = slave_body.n_dofs + master_body.n_dofs
        forces = contact.evaluate_forces(n_dofs=total_dofs)

        slave_f = forces[:slave_body.n_dofs].reshape(-1, 3)
        master_f = forces[slave_body.n_dofs:total_dofs].reshape(-1, 3)

        sum_slave = np.sum(slave_f, axis=0)
        sum_master = np.sum(master_f, axis=0)
        net_force = sum_slave + sum_master

        assert np.linalg.norm(sum_slave) > 0.0
        assert np.allclose(net_force, 0.0, atol=1e-11), f"Net mortar interface force: {net_force}"


# =============================================================================
# 3. Rigid Obstacle Static Stick & Slip Solves
# =============================================================================

class TestRigidContactStickSlipSolves:
    """End-to-end static simulations of an elastic block on a rigid plane with Coulomb friction."""

    def test_static_shear_stick_and_slip_regimes(self) -> None:
        """Verifies elastic stick for F_T < mu * F_N, and clamping for sliding displacement."""
        # Single element or small block
        mesh = make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0))
        mat = make_neohookean(E=1e4, nu=0.2)
        domain = SolidDomain(mesh, mat, Tet4TotalLagrangian(mat))

        # Rigid floor at y = 0
        floor = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

        mu = 0.3
        k_penalty = 1e4
        contact = RigidContact(
            slave=domain,
            master=floor,
            k_penalty=k_penalty,
            friction=mu,
            formulation="penalty",
        )

        # Press block into floor by fixing top nodes with negative y-displacement
        top_nodes = np.flatnonzero(mesh.nodes[:, 1] > 9.9)
        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(DirichletConstraint(top_nodes, components=(1,), value=-0.05))

        analysis = NonlinearStaticAnalysis(domain=domain)
        sol = analysis.solve(
            dirichlet=dirichlet,
            contact=contact,
            n_steps=2,
            nr_iterations=25,
            nr_tol=1e-4,
        )
        assert sol.converged

        contact_info = contact.get_contact_info()
        assert contact_info["num_active"] > 0
        normal_f = contact.evaluate_forces()
        f_N_total = np.sum(normal_f.reshape(-1, 3)[:, 1])
        assert f_N_total > 0.0  # Upward repulsive contact force

        # Now apply a shear displacement along x on top nodes
        # Step 1: Small shear (STICK regime: F_x < mu * F_N)
        dirichlet_stick = DirichletManager(domain.n_dofs)
        dirichlet_stick.add_constraint(DirichletConstraint(top_nodes, components=(1,), value=-0.05))
        dirichlet_stick.add_constraint(DirichletConstraint(top_nodes, components=(0,), value=0.001))

        sol_stick = analysis.solve(
            dirichlet=dirichlet_stick,
            contact=contact,
            n_steps=1,
            nr_iterations=20,
            nr_tol=1e-4,
        )
        assert sol_stick.converged
        info_stick = contact.get_contact_info()
        assert info_stick["num_stick"] > 0
        assert info_stick["num_slip"] == 0

        # Step 2: Large shear (SLIP regime: friction force reaches mu * F_N limit)
        dirichlet_slip = DirichletManager(domain.n_dofs)
        dirichlet_slip.add_constraint(DirichletConstraint(top_nodes, components=(1,), value=-0.05))
        dirichlet_slip.add_constraint(DirichletConstraint(top_nodes, components=(0,), value=0.5))

        sol_slip = analysis.solve(
            dirichlet=dirichlet_slip,
            contact=contact,
            n_steps=1,
            nr_iterations=20,
            nr_tol=1e-4,
        )
        assert sol_slip.converged
        info_slip = contact.get_contact_info()
        assert info_slip["max_slip_distance"] > 0.0
        # Friction force on slipping nodes must reach exact Coulomb limit mu * f_N
        forces_slip = contact.evaluate_forces().reshape(-1, 3)
        for i in [1, 5]:
            f_T_i = np.linalg.norm(forces_slip[i, [0, 2]])
            f_N_i = forces_slip[i, 1]
            assert np.isclose(f_T_i, mu * f_N_i, rtol=1e-3)

    def test_rigid_contact_augmented_lagrangian_friction(self) -> None:
        """Verifies Augmented Lagrangian contact solve with Coulomb friction."""
        mesh = make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0))
        mat = make_neohookean(E=1e4, nu=0.2)
        domain = SolidDomain(mesh, mat, Tet4TotalLagrangian(mat))

        floor = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        mu = 0.35
        contact = RigidContact(
            slave=domain,
            master=floor,
            k_penalty=5e4,
            friction=mu,
            formulation="augmented_lagrangian",
        )

        top_nodes = np.flatnonzero(mesh.nodes[:, 1] > 9.9)
        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(DirichletConstraint(top_nodes, components=(1,), value=-0.04))
        dirichlet.add_constraint(DirichletConstraint(top_nodes, components=(0,), value=0.1))

        analysis = NonlinearStaticAnalysis(domain=domain)
        sol = analysis.solve(
            dirichlet=dirichlet,
            contact=contact,
            n_steps=1,
            al_max_iterations=15,
            al_gap_tol=1e-4,
            al_tol=1e-3,
            nr_iterations=20,
            nr_tol=1e-4,
        )
        assert sol.converged
        info = contact.get_contact_info()
        assert info["num_active"] > 0
        assert info["num_slip"] > 0
        assert info["max_friction_force"] > 0.0

    def test_dynamic_sliding_frictional_dissipation(self) -> None:
        """Dynamic sliding block on rigid floor must dissipate horizontal kinetic energy."""
        mesh = make_cube_mesh(origin=(0.0, 0.1, 0.0), size=(10.0, 10.0, 10.0))
        mat = make_neohookean(E=1e4, nu=0.2)
        domain = SolidDomain(mesh, mat, Tet4TotalLagrangian(mat))

        floor = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        mu = 0.4
        contact = RigidContact(
            slave=domain,
            master=floor,
            k_penalty=1e4,
            c_damping=50.0,
            friction=mu,
            formulation="penalty",
        )

        # Initialize horizontal velocity v_x = 200 mm/s and downward velocity v_y = -50 mm/s
        v_init = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
        v_init[:, 0] = 200.0  # Horizontal velocity
        v_init[:, 1] = -50.0  # Downward velocity towards floor
        domain.update_state(v=v_init.ravel())

        dirichlet = DirichletManager(domain.n_dofs)
        f_ext = np.zeros(domain.n_dofs)
        dynamic_solver = NonlinearDynamicAnalysis(
            domain=domain,
            spectral_radius=0.8,
        )
        sol = dynamic_solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.002,
            n_steps=10,
            contact=contact,
            nr_iterations=20,
            nr_tol=1e-4,
            verbose=False,
        )
        assert sol.converged

        # Verify that horizontal velocity decreased significantly due to frictional braking
        v_final = domain.v.reshape(-1, 3)
        mean_vx_final = float(np.mean(v_final[:, 0]))
        assert mean_vx_final < 200.0, f"Expected deceleration from 200, got {mean_vx_final}"
        info = contact.get_contact_info()
        assert info["max_slip_distance"] > 0.0

    def test_multibody_two_body_frictional_sliding(self) -> None:
        """Two deformable blocks in MultiBodySystem sliding with friction."""
        mesh_bottom = make_cube_mesh(origin=(0.0, 0.0, 0.0), size=(10.0, 5.0, 10.0))
        mesh_top = make_cube_mesh(origin=(2.0, 5.0, 2.0), size=(6.0, 5.0, 6.0))

        mat = make_neohookean(E=1e4, nu=0.2)
        body_bottom = SolidDomain(mesh_bottom, mat, Tet4TotalLagrangian(mat))
        body_top = SolidDomain(mesh_top, mat, Tet4TotalLagrangian(mat))

        system = MultiBodySystem({"Bottom": body_bottom, "Top": body_top})
        contact = system.add_contact_pair(
            slave="Top",
            master="Bottom",
            k_penalty=5000.0,
            friction=0.3,
            formulation="penalty",
        )

        # Fix bottom block base completely
        dirichlet = DirichletManager(system.n_dofs)
        base_nodes = np.flatnonzero(mesh_bottom.nodes[:, 1] < 0.1)
        bottom_offset = int(system._dof_offsets[0])
        dirichlet.add_constraint(DirichletConstraint(base_nodes, components=(0, 1, 2), dof_offset=bottom_offset))

        # Push top block down by 0.02 mm and pull horizontally by 0.05 mm
        top_offset = int(system._dof_offsets[1])
        top_upper_nodes = np.flatnonzero(mesh_top.nodes[:, 1] > 9.9)
        dirichlet.add_constraint(DirichletConstraint(top_upper_nodes, components=(1,), value=-0.02, dof_offset=top_offset))
        dirichlet.add_constraint(DirichletConstraint(top_upper_nodes, components=(0,), value=0.05, dof_offset=top_offset))

        analysis = NonlinearStaticAnalysis(domain=system)
        sol = analysis.solve(
            dirichlet=dirichlet,
            contact=system.contacts,
            n_steps=1,
            nr_iterations=20,
            nr_tol=1e-4,
        )
        assert sol.converged
        info = contact.get_contact_info()
        assert info["num_active"] > 0
        assert info["max_friction_force"] > 0.0
