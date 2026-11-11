"""
Unit & Integration Tests for Two-Way Deformable-to-Deformable Contact Mechanics.
================================================================================
Verifies:
1. Exact barycentric projection and interpolation on triangular master surface facets.
2. Exact Newtonian action-reaction interface force balance (sum(f_c) == 0).
3. Exact symmetry of coupled tangent stiffness matrix (K_c == K_c^T) and cross-body coupling.
4. Rigid translation invariance of the contact tangent operator.
5. MultiBodySystem automatic routing to DeformableContact.
6. Augmented Lagrangian multiplier update driving penetration error g_N -> 0.
7. End-to-end static two-body flexible contact equilibrium.
"""

from __future__ import annotations
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.mesh import VolumetricMesh
from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    NodalForce,
)
from psierra.boundary.contact import DeformableContact
from psierra.analysis import NonlinearStaticAnalysis


class TestBarycentricGeometry:
    """Tests triangle projection and barycentric coordinates."""

    def test_barycentric_interior(self) -> None:
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([10.0, 0.0, 0.0])
        v3 = np.array([0.0, 10.0, 0.0])
        p = np.array([2.0, 3.0, 0.0])

        xi1, xi2, xi3 = DeformableContact.compute_barycentric(p, v1, v2, v3)
        assert abs((xi1 + xi2 + xi3) - 1.0) < 1e-12
        assert 0.0 <= xi1 <= 1.0
        assert 0.0 <= xi2 <= 1.0
        assert 0.0 <= xi3 <= 1.0

        p_reconstructed = xi1 * v1 + xi2 * v2 + xi3 * v3
        assert np.allclose(p, p_reconstructed)

    def test_barycentric_vertices_and_edges(self) -> None:
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([5.0, 1.0, 2.0])
        v3 = np.array([2.0, 6.0, 1.0])

        # Test vertex 1
        xi = DeformableContact.compute_barycentric(v1, v1, v2, v3)
        assert np.allclose(xi, [1.0, 0.0, 0.0], atol=1e-10)

        # Test midpoint of edge 1-2
        m12 = 0.5 * (v1 + v2)
        xi = DeformableContact.compute_barycentric(m12, v1, v2, v3)
        assert np.allclose(xi, [0.5, 0.5, 0.0], atol=1e-10)


class TestDeformableContactMechanics:
    """Tests action-reaction, stiffness symmetry, and cross-coupling on two cubes."""

    @pytest.fixture
    def touching_cubes(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> tuple[SolidDomain, SolidDomain]:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        # Upper cube [0.25, 0.75] x [0, 1] x [0.25, 0.75]
        mesh_upper = five_tet_cube_mesh.copy()
        mesh_upper.transform(scale=(0.5, 1.0, 0.5), translation=(0.25, 0.0, 0.25))
        # Lower cube [0, 1] x [-1, 0] x [0, 1]
        mesh_lower = five_tet_cube_mesh.copy()
        mesh_lower.transform(translation=(0.0, -1.0, 0.0))

        body_upper = SolidDomain(mesh_upper, mat, elem)
        body_lower = SolidDomain(mesh_lower, mat, elem)
        return body_upper, body_lower

    def test_action_reaction_force_balance(
        self, touching_cubes: tuple[SolidDomain, SolidDomain]
    ) -> None:
        upper, lower = touching_cubes

        # Penetrate upper cube 0.1 mm down into lower cube
        u_upper = np.zeros(upper.n_dofs)
        u_upper[1::3] = -0.1
        upper.update_state(u=u_upper)

        contact = DeformableContact(
            slave=upper,
            master=lower,
            k_penalty=5000.0,
            slave_offset=0,
            master_offset=upper.n_dofs,
        )

        n_total = upper.n_dofs + lower.n_dofs
        f_c = contact.evaluate_forces(n_dofs=n_total)

        # Total contact force vector must strictly sum to zero in 3D (momentum conservation)
        f_x_total = np.sum(f_c[0::3])
        f_y_total = np.sum(f_c[1::3])
        f_z_total = np.sum(f_c[2::3])

        assert abs(f_x_total) < 1e-10
        assert abs(f_y_total) < 1e-10
        assert abs(f_z_total) < 1e-10

        # Slave upper body must feel upward repulsive force (+Y)
        f_slave_y = np.sum(f_c[: upper.n_dofs][1::3])
        assert f_slave_y > 0.0

        # Master lower body must feel downward reaction force (-Y)
        f_master_y = np.sum(f_c[upper.n_dofs :][1::3])
        assert f_master_y < 0.0
        assert np.isclose(f_slave_y, -f_master_y, rtol=1e-12)

    def test_tangent_stiffness_symmetry_and_cross_coupling(
        self, touching_cubes: tuple[SolidDomain, SolidDomain]
    ) -> None:
        upper, lower = touching_cubes

        u_upper = np.zeros(upper.n_dofs)
        u_upper[1::3] = -0.05
        upper.update_state(u=u_upper)

        contact = DeformableContact(
            slave=upper,
            master=lower,
            k_penalty=1e4,
            slave_offset=0,
            master_offset=upper.n_dofs,
        )

        n_total = upper.n_dofs + lower.n_dofs
        K_c = contact.evaluate_stiffness(n_dofs=n_total)

        # 1. Symmetry check: K_c == K_c^T
        diff = K_c - K_c.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12

        # 2. Cross-coupling block existence: K_AB and K_BA must be non-zero
        K_dense = K_c.toarray()
        K_AB = K_dense[: upper.n_dofs, upper.n_dofs :]
        K_BA = K_dense[upper.n_dofs :, : upper.n_dofs]

        assert np.linalg.norm(K_AB) > 0.0
        assert np.allclose(K_AB, K_BA.T)

        # 3. Rigid translations leave all relative gaps/forces unchanged.
        # The full consistent Hessian includes geometric curvature and can be
        # indefinite; test_deformable_contact_consistency verifies that curvature
        # against independent energy differences instead of imposing PSD.
        for component in range(3):
            translation = np.zeros(n_total)
            translation[component::3] = 1.0
            np.testing.assert_allclose(K_c @ translation, 0.0, atol=1e-10)

    def test_multibody_add_contact_pair_integration(
        self, touching_cubes: tuple[SolidDomain, SolidDomain]
    ) -> None:
        upper, lower = touching_cubes
        system = MultiBodySystem({"Upper": upper, "Lower": lower})

        # Automatic detection of deformable master -> DeformableContact
        contact = system.add_contact_pair("Upper", "Lower", k_penalty=2e4)
        assert isinstance(contact, DeformableContact)
        assert contact.slave is upper
        assert contact.master is lower
        assert contact.slave_offset == 0
        assert contact.master_offset == upper.n_dofs
        assert len(system.contacts) == 1


class TestAugmentedLagrangianFormulation:
    """Tests multiplier updates and zero penetration enforcement."""

    def test_multiplier_convergence(self, five_tet_cube_mesh: VolumetricMesh) -> None:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        mesh_upper = five_tet_cube_mesh.copy()
        mesh_upper.transform(scale=(0.5, 1.0, 0.5), translation=(0.25, 0.0, 0.25))
        upper = SolidDomain(mesh_upper, mat, elem)

        mesh_lower = five_tet_cube_mesh.copy()
        mesh_lower.transform(translation=(0.0, -1.0, 0.0))
        lower = SolidDomain(mesh_lower, mat, elem)

        # Initial penetration of 0.1 mm
        u_upper = np.zeros(upper.n_dofs)
        u_upper[1::3] = -0.1
        upper.update_state(u=u_upper)

        contact = DeformableContact(
            slave=upper,
            master=lower,
            k_penalty=5000.0,
            slave_offset=0,
            master_offset=upper.n_dofs,
            formulation="augmented_lagrangian",
        )

        assert np.all(contact.multipliers == 0.0)

        # First query: forces are purely penalty
        f1 = contact.evaluate_forces()
        f1_mag = np.linalg.norm(f1)
        assert f1_mag > 0.0

        # Update multipliers: lambda += kp * delta
        stats = contact.update_multipliers()
        assert stats["num_active"] > 0
        assert stats["max_multiplier"] > 0.0
        assert np.any(contact.multipliers > 0.0)

        # Subsequent force evaluation now incorporates multipliers
        f2 = contact.evaluate_forces()
        f2_mag = np.linalg.norm(f2)
        assert f2_mag > f1_mag


class TestTwoBodyStaticSolve:
    """Verifies end-to-end two-body flexible contact in static equilibrium."""

    def test_two_cube_compression_equilibrium(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(5000.0, 0.3, 1.0)
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)

        # Upper cube at [0.25, 0.75] x [0, 1] x [0.25, 0.75]
        mesh_upper = five_tet_cube_mesh.copy()
        mesh_upper.transform(scale=(0.5, 1.0, 0.5), translation=(0.25, 0.0, 0.25))
        upper = SolidDomain(mesh_upper, mat, elem)

        # Lower cube at [0, 1] x [-1, 0] x [0, 1]
        mesh_lower = five_tet_cube_mesh.copy()
        mesh_lower.transform(translation=(0.0, -1.0, 0.0))
        lower = SolidDomain(mesh_lower, mat, elem)

        system = MultiBodySystem({"Upper": upper, "Lower": lower})
        contact = system.add_contact_pair(
            slave="Upper",
            master="Lower",
            k_penalty=2e4,
            formulation="augmented_lagrangian",
        )

        # Boundary conditions:
        # - Fix bottom face of lower cube (Y = -1.0)
        bot_nodes = np.where(np.isclose(lower.mesh.nodes[:, 1], -1.0))[0]
        clamp_bot = DirichletConstraint(
            bot_nodes,
            components=(0, 1, 2),
            value=0.0,
            dof_offset=system.get_body_dof_slice("Lower").start,
        )

        # - Lateral guide for upper cube to prevent rigid slip in X and Z
        top_nodes = np.where(np.isclose(upper.mesh.nodes[:, 1], 1.0))[0]
        guide_x = DirichletConstraint(
            top_nodes, components=(0,), value=0.0, dof_offset=0
        )
        guide_z = DirichletConstraint(
            top_nodes, components=(2,), value=0.0, dof_offset=0
        )
        prescribe_top = DirichletConstraint(
            top_nodes, components=(1,), value=-0.1, dof_offset=0
        )

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(clamp_bot)
        dirichlet.add_constraint(guide_x)
        dirichlet.add_constraint(guide_z)
        dirichlet.add_constraint(prescribe_top)

        solver = NonlinearStaticAnalysis(system)
        result = solver.solve(
            dirichlet=dirichlet,
            n_steps=2,
            nr_tol=1e-3,
            verbose=True,
        )

        assert result.converged
        c_info = contact.get_contact_info()
        assert c_info["num_active"] > 0
        # Both cubes must experience downward deformation
        assert upper.displacement[:, 1].min() < -0.05
        assert lower.displacement[:, 1].min() < 0.0
        assert lower.displacement[:, 1].min() < 0.0  # lower cube compressed

        # Acceptance must satisfy equilibrium and the AL fixed-point constraint
        # together, using the stored multipliers without an unchecked update.
        contact_force = contact.evaluate_forces(n_dofs=system.n_dofs)
        internal_force = system.assemble_internal_forces()
        free = dirichlet.free_dofs
        scale = max(
            np.linalg.norm(contact_force[free]),
            np.linalg.norm(internal_force[free]),
            1.0,
        )
        assert np.linalg.norm((contact_force - internal_force)[free]) / scale < 1e-3
        gaps = contact.evaluate_gaps()
        assert np.max(np.maximum(-gaps, 0.0)) <= 1e-6
        proposed = np.maximum(0.0, contact.multipliers - contact.k_penalty * gaps)
        multiplier_scale = max(
            np.max(np.abs(proposed)), np.max(np.abs(contact.multipliers))
        )
        assert multiplier_scale > 0.0
        assert np.max(np.abs(proposed - contact.multipliers)) / multiplier_scale <= 1e-6
        assert len(result.al_iterations_history) == 2
