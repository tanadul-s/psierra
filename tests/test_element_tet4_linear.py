"""
Unit and benchmark tests for Tet4Linear finite element.

Validates:
1. Reference volume calculation (V = 1/6) and shape function gradient partition of unity.
2. Inverted element detection (negative Jacobian determinant error).
3. Mass matrix symmetry, positive-definiteness, and total mass conservation (sum of M_xx = rho * V).
4. Linear stiffness matrix symmetry, rank, and 6 rigid body modes (3 translations + 3 rotations).
5. Rigid body translation invariance (zero reaction forces).
6. Constant strain patch test (exact strain recovery to machine precision).
7. End-to-end single-element tension equilibrium.
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.element.tet4_linear import Tet4Linear
from psierra.material.hookean import Hookean
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.analysis.linear_static import LinearStaticAnalysis, LinearStaticResult
from psierra.utility.tensor import tensor_to_voigt_strain


class TestTet4Linear:
    """Rigorous physical and mathematical verification of 4-node linear tetrahedral elements."""

    def test_volume_and_gradients(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Standard unit right tetrahedron has volume 1/6 and gradients sum to zero."""
        tet = Tet4Linear(hookean_material)
        V, H = tet.compute_volume_and_gradients(regular_tet_coords)

        # Standard unit right tet has volume 1/6
        assert np.isclose(V, 1.0 / 6.0, atol=1e-14)
        assert H.shape == (4, 3)

        # Partition of unity: sum of shape function spatial gradients must be exactly zero
        assert np.allclose(np.sum(H, axis=0), np.zeros(3), atol=1e-14)

    def test_inverted_element_raises(
        self, hookean_material: Hookean, inverted_tet_coords: np.ndarray
    ) -> None:
        """Negative or zero volume elements must raise ValueError."""
        tet = Tet4Linear(hookean_material)
        with pytest.raises(ValueError, match="volume detected"):
            tet.compute_volume_and_gradients(inverted_tet_coords)

    def test_mass_matrix_properties(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Mass matrix must be symmetric, positive-definite, and conserve total mass."""
        tet = Tet4Linear(hookean_material)
        V, _ = tet.compute_volume_and_gradients(regular_tet_coords)
        M_e = tet.compute_mass_matrix(V)

        assert M_e.shape == (12, 12)
        # Symmetry
        assert np.allclose(M_e, M_e.T, atol=1e-14)

        # Positive-definiteness
        eigvals = np.linalg.eigvalsh(M_e)
        assert np.all(eigvals > 0.0)

        # Total mass conservation: sum of all coefficients in X-direction DOFs = total mass rho * V
        rho = hookean_material.mass_density
        total_mass_expected = rho * V
        x_dofs = np.arange(0, 12, 3)
        assert np.isclose(
            np.sum(M_e[np.ix_(x_dofs, x_dofs)]), total_mass_expected, rtol=1e-10
        )

    def test_linear_stiffness_matrix_rigid_body_modes(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Linear stiffness matrix must have exactly 6 zero eigenvalues and 6 positive eigenvalues."""
        tet = Tet4Linear(hookean_material)
        K_e = tet.compute_linear_stiffness_matrix(regular_tet_coords)

        assert K_e.shape == (12, 12)
        # Symmetry
        assert np.allclose(K_e, K_e.T, atol=1e-12)

        # In 3D unconstrained solid mechanics, an element has exactly 6 zero eigenvalues
        # (3 rigid translations + 3 rigid rotations) and 6 strictly positive eigenvalues.
        eigvals = np.linalg.eigvalsh(K_e)
        zero_eigvals = eigvals[:6]
        pos_eigvals = eigvals[6:]

        # Zero eigenvalues relative to max stiffness eigenvalue must be within machine epsilon
        assert np.allclose(zero_eigvals / np.max(pos_eigvals), 0.0, atol=1e-12)
        assert np.all(pos_eigvals > 1e4)

    def test_pure_translation_invariance(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Under arbitrary rigid translation, reaction forces must be zero to machine precision."""
        tet_lin = Tet4Linear(hookean_material)
        K_e = tet_lin.compute_linear_stiffness_matrix(regular_tet_coords)

        # Arbitrary rigid translation vector [tx, ty, tz]
        t = np.array([12.5, -34.2, 5.8], dtype=np.float64)
        u_rigid = np.tile(t, 4)

        # Linear reaction force K * u must be zero within machine precision relative to ||K||*||u||
        f_reaction = K_e @ u_rigid
        rel_error = np.linalg.norm(f_reaction) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error < 1e-12

        # Non-linear interface on Tet4Linear should also vanish under pure translation
        u_nodes = np.tile(t, (4, 1))
        _, f_int_lin = tet_lin.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )
        rel_error_lin = np.linalg.norm(f_int_lin) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error_lin < 1e-12

    def test_constant_strain_patch_test(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """
        Applies a linear displacement field u(X) = eps_0 @ X.
        Verifies that computed strain equals eps_0 to machine precision.
        """
        tet = Tet4Linear(hookean_material)
        _, H = tet.compute_volume_and_gradients(regular_tet_coords)
        B = tet.compute_linear_B_matrix(H)

        eps_0 = np.array(
            [
                [0.001, 0.0005, 0.0002],
                [0.0005, -0.0004, 0.0001],
                [0.0002, 0.0001, 0.0008],
            ]
        )
        # Nodal displacements: u_i = eps_0 @ X_i
        u_nodes = regular_tet_coords @ eps_0
        u_vector = u_nodes.flatten()

        eps_computed_voigt = B @ u_vector
        eps_expected_voigt = tensor_to_voigt_strain(eps_0)

        assert np.allclose(eps_computed_voigt, eps_expected_voigt, atol=1e-14)

    def test_end_to_end_single_element_tension(
        self, hookean_material: Hookean, single_element_mesh: VolumetricMesh
    ) -> None:
        """Single element under uniaxial tension verifies equilibrium and displacement."""
        elem = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh=single_element_mesh, material=hookean_material, element=elem)

        # Fix base nodes (nodes 0, 1, 2 at Z = 0)
        clamp = DirichletConstraint([0, 1, 2], components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(domain.mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Tensile load on node 3 in +Z direction
        f_applied = 500.0
        load = NodalForce([3], force_vector=[0.0, 0.0, f_applied])
        f_ext = load.compute_force_vector(domain.mesh.n_dofs)

        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

        assert isinstance(result, LinearStaticResult)
        assert result.displacement[11] > 0.0  # Apex node Z-displacement is positive
        # Reaction balance
        assert np.isclose(np.sum(result.reactions[2::3]) + f_applied, 0.0, atol=1e-8)
