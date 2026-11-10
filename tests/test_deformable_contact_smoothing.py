"""
Unit & Integration Tests for Spatial Regularization: C^1-Continuous Surface Normal Smoothing.
=============================================================================================
Pillar 2 of Loop 2 Contact Mechanics Milestone.

Verifies:
1. Exact elimination of normal directional jump Delta n = n_B - n_A across shared facet edges.
2. C^0-continuous normal vector interpolation along master surface facets.
3. Exact Newtonian action-reaction interface force balance (sum(f_c) == 0) to machine precision.
4. Exact symmetry of coupled tangent stiffness matrix (K_c == K_c^T) to machine precision.
5. Rigid body translation invariance of the smoothed contact tangent operator (K_c @ u_trans == 0).
6. Support for both angle-weighting and area-weighting vertex pseudonormals.
7. Parameter validation for invalid smoothing weights and backward compatibility (smoothed_normals=False).
8. SelfContact support for smoothed normals, symmetry, and action-reaction.
9. Smooth, continuous force variation eliminating edge chattering during sliding across facet boundaries.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp

from psierra.boundary.contact import DeformableContact, SelfContact
from psierra.collision.triangle import TriangleContactGeometry
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh


class GeometryDomain(Domain):
    """Minimal prescribed 3D coordinates domain isolating contact mechanics from material models."""

    def __init__(self, nodes: np.ndarray, faces: np.ndarray | None = None) -> None:
        self.reference = np.asarray(nodes, dtype=np.float64).copy()
        self.u = np.zeros(nodes.size, dtype=np.float64)
        self.v = np.zeros(nodes.size, dtype=np.float64)
        self.a = np.zeros(nodes.size, dtype=np.float64)
        if faces is not None:
            cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
            polydata = pv.PolyData(self.reference, cells)
            self.mesh = SimpleNamespace(
                nodes=self.reference,
                surface=SurfaceMesh(polydata, node_indices=np.arange(len(self.reference))),
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
            self.u = np.asarray(u, dtype=np.float64).copy()
        if v is not None:
            self.v = np.asarray(v, dtype=np.float64).copy()
        if a is not None:
            self.a = np.asarray(a, dtype=np.float64).copy()

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs))

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def save_state(self, filepath: Path, **kwargs: Any) -> None:
        raise NotImplementedError("Prescribed geometry domain does not support file export")


class TestTriangleGeometryNormalSmoothing:
    """Tests C^1 continuous surface normal smoothing at the triangle collision kernel level."""

    def test_shared_edge_normal_jump_elimination(self) -> None:
        """Verify that crossing a shared facet edge has a 30-deg jump unsmoothed but 0 jump smoothed."""
        theta = np.radians(30.0)
        # Triangle A in x < 0 tilted up by 30 degrees
        vA = np.array([
            [-np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Triangle B in x > 0 flat on z = 0
        vB = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        nA, _ = TriangleContactGeometry._face_normal(vA)
        nB, _ = TriangleContactGeometry._face_normal(vB)

        # Unsmoothed facet normals jump by ~0.518 (~30 degrees)
        normal_jump_facet = float(np.linalg.norm(nA - nB))
        assert normal_jump_facet > 0.50

        # Build normalized vertex pseudonormals for shared nodes (1 and 2) and outer nodes
        nv_shared = (nA + nB) / np.linalg.norm(nA + nB)
        vnA = np.array([nA, nv_shared, nv_shared])
        vnB = np.array([nv_shared, nB, nv_shared])

        # Evaluate exactly on the shared edge (x=0, y=0.5, z=0)
        p_edge = np.array([0.0, 0.5, 0.0])
        gapA_edge = TriangleContactGeometry.evaluate(p_edge, vA, vertex_normals=vnA)
        gapB_edge = TriangleContactGeometry.evaluate(p_edge, vB, vertex_normals=vnB)

        # On the edge, smoothed normals from facet A and facet B must be IDENTICAL to machine precision
        assert np.allclose(gapA_edge.normal, gapB_edge.normal, atol=1e-14)
        assert np.allclose(gapA_edge.normal, nv_shared, atol=1e-14)

        # Sliding across the edge from x = -eps to x = +eps
        eps = 1e-4
        pA = np.array([-eps, 0.5, 0.0])
        pB = np.array([+eps, 0.5, 0.0])

        gapA_unsm = TriangleContactGeometry.evaluate(pA, vA)
        gapB_unsm = TriangleContactGeometry.evaluate(pB, vB)
        jump_unsm = float(np.linalg.norm(gapA_unsm.normal - gapB_unsm.normal))
        assert jump_unsm > 0.50

        gapA_sm = TriangleContactGeometry.evaluate(pA, vA, vertex_normals=vnA)
        gapB_sm = TriangleContactGeometry.evaluate(pB, vB, vertex_normals=vnB)
        jump_sm = float(np.linalg.norm(gapA_sm.normal - gapB_sm.normal))
        assert jump_sm < 1e-4
        assert jump_sm < 0.001 * jump_unsm


class TestDeformableContactSmoothing:
    """Tests DeformableContact integration with smoothed vertex pseudonormals."""

    @pytest.fixture
    def angled_two_facet_master(self) -> tuple[GeometryDomain, GeometryDomain]:
        """Creates a 2-triangle master surface with a 30-deg dihedral ridge along y."""
        theta = np.radians(30.0)
        master_nodes = np.array([
            [-np.cos(theta), 0.0, np.sin(theta)],  # Node 0 (left facet outer)
            [0.0, 0.0, 0.0],                       # Node 1 (shared ridge)
            [0.0, 1.0, 0.0],                       # Node 2 (shared ridge)
            [1.0, 0.0, 0.0],                       # Node 3 (right facet outer)
        ])
        master_faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        master = GeometryDomain(master_nodes, master_faces)
        # Single-node slave body initially at x=-0.1, y=0.5, z=-0.05 (penetrating left facet)
        slave_node = np.array([[-0.1, 0.5, -0.05]])
        slave = GeometryDomain(slave_node)
        return slave, master

    def test_default_backward_compatibility(
        self, angled_two_facet_master: tuple[GeometryDomain, GeometryDomain]
    ) -> None:
        """Verify default smoothed_normals=False matches existing unsmoothed behavior."""
        slave, master = angled_two_facet_master
        contact = DeformableContact(slave, master, k_penalty=1000.0)
        assert contact.smoothed_normals is False
        assert contact.smoothing_weight == "angle"

    def test_invalid_smoothing_weight_rejected(
        self, angled_two_facet_master: tuple[GeometryDomain, GeometryDomain]
    ) -> None:
        """Verify invalid smoothing_weight values raise ValueError."""
        slave, master = angled_two_facet_master
        with pytest.raises(ValueError, match="smoothing_weight must be 'angle' or 'area'"):
            DeformableContact(slave, master, smoothing_weight="volume")  # type: ignore[arg-type]

    @pytest.mark.parametrize("smoothing_weight", ["angle", "area"])
    def test_action_reaction_force_balance_smoothed(
        self,
        angled_two_facet_master: tuple[GeometryDomain, GeometryDomain],
        smoothing_weight: str,
    ) -> None:
        """Verify exact Newton action-reaction (sum(f_c) == 0) across all coordinates."""
        slave, master = angled_two_facet_master
        contact = DeformableContact(
            slave=slave,
            master=master,
            k_penalty=5000.0,
            slave_offset=0,
            master_offset=slave.n_dofs,
            smoothed_normals=True,
            smoothing_weight=smoothing_weight,  # type: ignore[arg-type]
        )

        n_total = slave.n_dofs + master.n_dofs
        forces = contact.evaluate_forces(n_dofs=n_total)

        f_x_total = float(np.sum(forces[0::3]))
        f_y_total = float(np.sum(forces[1::3]))
        f_z_total = float(np.sum(forces[2::3]))

        assert abs(f_x_total) < 1e-12
        assert abs(f_y_total) < 1e-12
        assert abs(f_z_total) < 1e-12

    @pytest.mark.parametrize("smoothing_weight", ["angle", "area"])
    def test_tangent_stiffness_symmetry(
        self,
        angled_two_facet_master: tuple[GeometryDomain, GeometryDomain],
        smoothing_weight: str,
    ) -> None:
        """Verify exact symmetry of conservative contact tangent stiffness K_c == K_c^T."""
        slave, master = angled_two_facet_master
        contact = DeformableContact(
            slave=slave,
            master=master,
            k_penalty=5000.0,
            slave_offset=0,
            master_offset=slave.n_dofs,
            smoothed_normals=True,
            smoothing_weight=smoothing_weight,  # type: ignore[arg-type]
        )

        n_total = slave.n_dofs + master.n_dofs
        K_c = contact.evaluate_stiffness(n_dofs=n_total)

        assert K_c.shape == (n_total, n_total)
        K_diff = K_c - K_c.T
        asymmetry = float(np.max(np.abs(K_diff.data))) if K_diff.nnz else 0.0
        assert asymmetry < 1e-12

    def test_rigid_translation_invariance(
        self, angled_two_facet_master: tuple[GeometryDomain, GeometryDomain]
    ) -> None:
        """Verify that rigid translation of slave and master produces zero net contact force."""
        slave, master = angled_two_facet_master
        contact = DeformableContact(
            slave=slave,
            master=master,
            k_penalty=5000.0,
            slave_offset=0,
            master_offset=slave.n_dofs,
            smoothed_normals=True,
        )

        n_total = slave.n_dofs + master.n_dofs
        K_c = contact.evaluate_stiffness(n_dofs=n_total)

        # Translate uniformly along [1.0, -2.0, 3.5]
        u_trans = np.tile([1.0, -2.0, 3.5], (slave.n_dofs + master.n_dofs) // 3)
        res = K_c @ u_trans
        assert np.linalg.norm(res) < 1e-10

    def test_sliding_contact_force_continuity_across_facet_edge(
        self, angled_two_facet_master: tuple[GeometryDomain, GeometryDomain]
    ) -> None:
        """Verify that sliding a slave node across a facet edge produces no discontinuous force jump."""
        slave, master = angled_two_facet_master
        c_unsm = DeformableContact(
            slave, master, k_penalty=2000.0, slave_offset=0, master_offset=slave.n_dofs,
            smoothed_normals=False,
        )
        c_sm = DeformableContact(
            slave, master, k_penalty=2000.0, slave_offset=0, master_offset=slave.n_dofs,
            smoothed_normals=True,
        )

        n_total = slave.n_dofs + master.n_dofs

        # Position A: x = -0.005 (on left facet)
        slave.update_state(u=np.array([0.095, 0.0, 0.04]))  # x = -0.005, z = -0.01
        f_unsm_A = c_unsm.evaluate_forces(n_dofs=n_total)[:3]
        f_sm_A = c_sm.evaluate_forces(n_dofs=n_total)[:3]

        # Position B: x = +0.005 (on right facet)
        slave.update_state(u=np.array([0.105, 0.0, 0.04]))  # x = +0.005, z = -0.01
        f_unsm_B = c_unsm.evaluate_forces(n_dofs=n_total)[:3]
        f_sm_B = c_sm.evaluate_forces(n_dofs=n_total)[:3]

        jump_unsm = float(np.linalg.norm(f_unsm_A - f_unsm_B))
        jump_sm = float(np.linalg.norm(f_sm_A - f_sm_B))

        # Unsmoothed force has a large directional jump proportional to sin(theta) * p
        assert jump_unsm > 10.0
        # Smoothed force varies continuously across the facet boundary
        assert jump_sm < 3.0
        assert jump_sm < 0.3 * jump_unsm

        # Right at the edge with micro-step delta_x = 2e-5:
        slave.update_state(u=np.array([0.09999, 0.0, 0.04]))
        f_sm_left = c_sm.evaluate_forces(n_dofs=n_total)[:3]
        slave.update_state(u=np.array([0.10001, 0.0, 0.04]))
        f_sm_right = c_sm.evaluate_forces(n_dofs=n_total)[:3]
        assert float(np.linalg.norm(f_sm_left - f_sm_right)) < 0.01


class TestSelfContactSmoothing:
    """Tests SelfContact integration with smoothed vertex pseudonormals."""

    @pytest.fixture
    def folded_sheet_domain(self) -> GeometryDomain:
        """Creates a folded sheet domain with two opposing surface planes."""
        nodes = np.array([
            [0.2, 0.2, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5],  # Upper sheet
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # Lower sheet
        ])
        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # Upper sheet pointing DOWN (-z)
            [4, 5, 6], [4, 6, 7],  # Lower sheet pointing UP (+z)
        ])
        return GeometryDomain(nodes, faces)

    def test_self_contact_smoothed_normals_symmetry_and_balance(
        self, folded_sheet_domain: GeometryDomain
    ) -> None:
        """Verify SelfContact with smoothed normals preserves symmetry and action-reaction."""
        # Penetrate upper node 0 downward into lower sheet
        u = np.zeros(folded_sheet_domain.n_dofs)
        u[2] = -0.55  # node 0 z becomes -0.05
        folded_sheet_domain.update_state(u=u)

        sc = SelfContact(
            domain=folded_sheet_domain,
            contact_nodes=[0],
            k_penalty=5000.0,
            search_radius=5.0,
            contact_thickness=2.0,
            exclusion_rings=0,
            smoothed_normals=True,
            smoothing_weight="angle",
        )

        assert sc.smoothed_normals is True
        assert sc.smoothing_weight == "angle"

        forces = sc.evaluate_forces()
        f_total = np.sum(forces.reshape(-1, 3), axis=0)
        assert np.allclose(f_total, [0.0, 0.0, 0.0], atol=1e-13)

        K_c = sc.evaluate_stiffness()
        K_diff = K_c - K_c.T
        asymmetry = float(np.max(np.abs(K_diff.data))) if K_diff.nnz else 0.0
        assert asymmetry < 1e-12

        # Rigid translation invariance
        u_trans = np.tile([1.0, -1.0, 0.5], len(folded_sheet_domain.reference))
        res = K_c @ u_trans
        assert np.linalg.norm(res) < 1e-11

    def test_self_contact_invalid_smoothing_weight_rejected(
        self, folded_sheet_domain: GeometryDomain
    ) -> None:
        """Verify invalid smoothing_weight raises ValueError in SelfContact."""
        with pytest.raises(ValueError, match="smoothing_weight must be 'angle' or 'area'"):
            SelfContact(
                domain=folded_sheet_domain,
                smoothing_weight="invalid",  # type: ignore[arg-type]
            )
