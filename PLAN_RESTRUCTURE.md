# pSierra Examples Restructuring Plan

> **Status**: Active Blueprint & Execution Guide  
> **Source Directory**: `psierra/examples/` (Source of truth during authoring)  
> **Staging Directory**: `psierra/examples_v2/` (Isolated workspace for construction & verification)  
> **Target Results Directory**: `psierra/results/` (Active runs write here; legacy outputs preserved in `psierra/results_old/`)

---

## 1. Executive Summary & Architectural Goals

The goal of this restructuring is to transition the demonstration suite from an organically grown collection of 63 scripts into a high-performance, strictly parallel, and maintainable engineering benchmark suite of **30 distinct scripts across 11 canonical categories**.

### Key Imperatives
1. **Coordinate & Orientation Harmonization**:
   - **Beam Geometry (`beam1.vtk`)**: Retain native $Z$-axis length (`[-60, 60] x [-60, 60] x [-30, 200]`). Completely eliminate arbitrary Euler transforms (`Rotation.from_euler("zyz", (0, 90, 90))`). Clamping is strictly semantic:
     - Root: `select_nodes_by_plane(mesh, plane="xy", bound="min")`
     - Tip: `select_nodes_by_plane(mesh, plane="xy", bound="max")`
     - Gravity / Loading: oriented along the $-Y$ axis (`[0.0, -9810.0, 0.0]`).
   - **Brain Geometry (`brain.vtk`)**: Retain native anatomical RAS frame. Patient surgical postures (`--posture supine|prone|left_lateral|right_lateral`) reorient the gravity vector without mutating mesh nodal coordinates.
2. **Dual-Track Parallelism**:
   - Symmetrical pairs between the **Canonical Engineering Benchmark** (`beam`) and the **Clinical Digital Twin** (`brain`) across all forward solvers:
     - `[03–07]01_beam.py` $\longleftrightarrow$ `[03–07]02_brain.py`
   - Strict 1:1 paired symmetry in Model Order Reduction (`0901`–`0906`) and Inverse FEM (`1001`–`1006`).
3. **Dedicated Geometries (No Runtime Mesh Toggling)**:
   - Self-contact meshes (`bunny1.vtk` and `cuboid1.vtk`) receive dedicated standalone scripts (`0804` and `0805`), avoiding confusing runtime mesh-toggle branches.
4. **Execution & Result Output Mechanism (Option A)**:
   - **Default Execution** (`python <script>.py`): Runs the fast canonical baseline case (e.g. `--formulation tl --element tet4`) in 1–2 seconds.
   - **Targeted Case** (`python <script>.py --formulation ul --element tet10`): Runs only that configuration.
   - **Comparative Matrix** (`python <script>.py --all`): Automatically loops through all combinations and prints a Rich comparative summary table.
   - **Result Isolation**: Results are written into case-tagged subdirectories to prevent overwrites and allow simultaneous overlay in ParaView:
     `results/<category>/<script_id>_<target>/<case_tag>/` (e.g. `results/05_nonlinear_dynamic/0501_beam/ul_tet10/`).
5. **Standardized Default Time Steps**:
   - **Linear Dynamic (`03_`)**: Default to **100 steps** (`dt = 0.001 s`).
   - **Nonlinear Dynamic (`05_`)**: Default to **200 steps** (`dt = 0.001 s`).
6. **Equal Standing for Advanced Materials**:
   - Hookean, Neo-Hookean, Ogden, HGO, and Viscoelastic models are introduced side-by-side in `0101_materials.py` and selectable via `--material` flags in dynamic/static solvers.

---

## 2. Directory Hierarchy & Script Index

```
psierra/examples/
├── 01_basics/                    # Core Building Blocks, Materials, & I/O
│   ├── 0101_materials.py                 # Hookean, Hyperelastic (Neo-Hookean, Ogden), Viscoelastic, HGO
│   ├── 0102_mesh_io.py                   # VTK/XDMF I/O, inspection, checkpoint resume
│   ├── 0103_mesh_generation.py           # Unstructured tet generation for beam, toroid, and brain STL
│   ├── 0104_sparse_assembly.py           # COO triplet to CSC global stiffness assembly
│   └── 0105_solid_domain.py              # SolidDomain lifecycle & boundary condition management
│
├── 02_linear_static/             # Small-Strain Static Solvers
│   ├── 0201_beam.py                      # Cantilever beam bending (tip load & gravity load)
│   ├── 0202_plate.py                     # Thin shell plate bending (S3 / DKT triangle)
│   └── 0203_beam_shell.py                # Coupled solid-shell continuum beam
│
├── 03_linear_dynamic/            # Linear Transient Integration (Default: 100 steps)
│   ├── 0301_beam.py                      # Transient vibration with Rayleigh damping
│   └── 0302_brain.py                     # Linear transient impulse response of brain tissue
│
├── 04_nonlinear_static/          # Large Deformations & Material Nonlinearity
│   ├── 0401_beam.py                      # Finite strain beam (--formulation svk|tl|cr|ul, --fbar)
│   └── 0402_brain.py                     # Static brain gravity sag across surgical postures
│
├── 05_nonlinear_dynamic/         # Finite-Strain Dynamic Time Marching (Default: 200 steps)
│   ├── 0501_beam.py                      # Dynamic beam (--formulation, --integrator, --buoyancy)
│   └── 0502_brain.py                     # Dynamic brain shift (--posture, --buoyancy)
│
├── 06_dynamic_relaxation/        # Explicit Kinetic Peak-Damping Equilibrium Solvers
│   ├── 0601_beam.py                      # Matrix-free static equilibrium under gravity
│   └── 0602_brain.py                     # Clinical burr-hole brain shift (intracranial pressure + gravity)
│
├── 07_quasi_static/              # Rate-Dependent & History-Dependent Quasi-Static Solvers
│   ├── 0701_beam.py                      # Rate-dependent viscoelastic beam bending
│   └── 0702_brain.py                     # Viscoelastic surgical brain relaxation under sustained gravity
│
├── 08_contact/                   # Penalty, Mortar, and Self-Contact Mechanics
│   ├── 0801_beam_floor_penalty.py        # Node-to-surface penalty contact on 3D floor
│   ├── 0802_beam_floor_mortar.py         # Segment-to-segment mortar contact on 3D floor
│   ├── 0803_pole_pad.py                  # Curved frictional contact between pad and cylindrical pole
│   ├── 0804_bunny_self_contact.py        # Stanford bunny self-contact under compound inclination
│   ├── 0805_cuboid_self_contact.py       # Twisted cuboid self-contact
│   └── 0806_frictional_stick_slip.py     # Coulomb friction stick-slip limit cycles
│
├── 09_mor/                       # Model Order Reduction & Hyper-Reduction
│   ├── 0901_beam_reduced.py              # Linear POD-Galerkin ROM
│   ├── 0902_brain_reduced.py             # Linear POD-Galerkin ROM for brain shift
│   ├── 0903_beam_hyperreduced.py         # Nonlinear ECSW & DEIM hyper-reduction
│   ├── 0904_brain_hyperreduced.py        # Real-time ECSW hyper-reduced brain shift
│   ├── 0905_beam_hyperreduced_contact.py # Contact-aware ECSW for beam-floor impact
│   └── 0906_brain_hyperreduced_contact.py# ECSW brain shift with skull contact
│
├── 10_inverse/                   # Symmetrical Inverse FEM & Parameter Identification
│   ├── 1001_beam_shape_sensing.py        # Variational displacement/shape sensing from sparse markers
│   ├── 1002_brain_shape_sensing.py       # Cortical marker tracking & subcortical tumor drift
│   ├── 1003_beam_param_identification.py # Modulus calibration from force-displacement curve
│   ├── 1004_brain_param_identification.py# Patient-specific stiffness calibration via probe
│   ├── 1005_beam_stress_free.py          # Undeformed reference configuration recovery
│   └── 1006_brain_stress_free.py         # Zero-gravity pre-op MRI reference configuration recovery
│
└── 11_interactive/               # Real-Time Interactive PyVista Pipelines
    ├── 1101_beam.py                      # Interactive cantilever dynamic bending & axial torsion
    ├── 1102_brain.py                     # Interactive 3D brain shift with live posture rotation
    └── 1103_brain_twin.py                # Hyper-reduced digital twin with live multi-modal sensing
```

---

## 3. The Three Implementation Phases

```
┌────────────────────────────────┐
│  Phase 1: Categories 01 – 04   │  Basics, Linear Static, Linear Dynamic (100 steps),
│  (12 Core Scripts)             │  Nonlinear Static (Beam collapsed & Brain sag)
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  Phase 2: Categories 05 – 08   │  Nonlinear Dynamic (200 steps), Dynamic Relaxation,
│  (12 Nonlinear & Contact)      │  Quasi-Static, Contact Mechanics (Bunny/Cuboid isolated)
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  Phase 3: Categories 09 – 11   │  MOR & Hyper-reduction (0901-0906), Inverse FEM (1001-1006),
│  (15 Scripts + Cutover)        │  Interactive Digital Twins (1101-1103), Runner & Test Sync,
│                                │  Atomic Directory Swap (examples_v2 -> examples)
└────────────────────────────────┘
```

---

### Phase 1: Categories 01 to 04 (Basics, Static, & Dynamic Foundations)

#### Category 01: Basics
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0101_materials.py` | `0101`, `0102`, `0103`, `1110`, `1111` | Unify Hookean, Neo-Hookean, Ogden, Viscoelastic, and HGO parameter instantiation into a single educational showcase. |
| `0102_mesh_io.py` | `0105`, `1104` | Reading VTK mesh, inspecting geometric volume/centroid, exporting XDMF/HDF5 vs PVD/VTU, and demonstration of checkpoint resumption. |
| `0103_mesh_generation.py` | `0107`, `0108`, `0109` | Clean synthetic torus generation, beam STL tetrahedralization, and brain STL surface meshing. |
| `0104_sparse_assembly.py` | `0104`, `0106` | Global COO triplet assembly and CSR/CSC conversion with sparsity pattern visualization. |
| `0105_solid_domain.py` | `0110` | Standard `SolidDomain` container setup with Dirichlet boundary constraints and body force verification. |

#### Category 02: Linear Static
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0201_beam.py` | `0201`, `0202` | Unify tip load and self-weight gravity bending into one script with `--load {tip,gravity,both}`. Standardize beam to native $Z$-axis and plane selector `plane="xy", bound="min"`. |
| `0202_plate.py` | `0203` | Discrete Kirchhoff Triangle (DKT / S3) thin-shell bending benchmark. |
| `0203_beam_shell.py` | `0204` | Coupled 3D solid core with thin shell exterior cladding beam. |

#### Category 03: Linear Dynamic (Default: 100 steps)
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0301_beam.py` | `0301` | Linear dynamic transient beam vibration with Newmark-$\beta$ and Rayleigh damping. Standardized to native $Z$-axis. Default steps: **100**. |
| `0302_brain.py` | Derived from `0504` linear baseline | Linear transient impulse response of anatomical brain tissue under mild acceleration pulse. Default steps: **100**. |

#### Category 04: Nonlinear Static
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0401_beam.py` | `0401`, `0402`, `0403`, `0404`, `1101`, `0702` | **Master Collapsed Script**: Finite-strain beam bending under gravity. Default: Total Lagrangian Tet4. Flag `--formulation {svk,total_lagrangian,corotational,updated_lagrangian}`, `--element {tet4,tet10}`, `--fbar` locking alleviation flag, and `--all` comparative matrix. |
| `0402_brain.py` | `0504` static mode | Static neurosurgical brain shift under gravity sag across surgical postures (`--posture supine|prone|left_lateral|right_lateral`). |

**Phase 1 Verification Gate**:
- Run all 12 scripts in `examples_v2/01_basics` through `examples_v2/04_nonlinear_static`.
- Verify convergence, ParaView PVD/XDMF exports in `results/`, and plane-selector boundaries.

---

### Phase 2: Categories 05 to 08 (Nonlinear Dynamics, Solvers, & Contact)

#### Category 05: Nonlinear Dynamic (Default: 200 steps)
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0501_beam.py` | `0501`, `0502`, `0503`, `1102`, `1110` | **Master Collapsed Script**: Nonlinear dynamic beam under gravity. Default: Total Lagrangian, Newmark, 200 steps. Flags: `--formulation`, `--integrator {newmark,hht,generalized_alpha}`, `--material {neohookean,ogden,viscoelastic}`, `--buoyancy`, and `--all`. |
| `0502_brain.py` | `0504`, `1103`, `1107` | Nonlinear dynamic neurosurgical brain shift. Default: 200 steps. Flags: `--posture`, `--material {hyperelastic,viscoelastic}`, `--buoyancy` (CSF fluid immersion), and `--solver {fast,full}`. |

#### Category 06: Dynamic Relaxation
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0601_beam.py` | `1105`, `1109` | Matrix-free explicit kinetic energy peak-damping solver (`DynamicRelaxationAnalysis`) finding beam static equilibrium without tangent stiffness factorization. |
| `0602_brain.py` | `1108`, `1304` | Neurosurgical burr-hole brain shift resolving cortical bulging vs. gravitational sagging under intracranial pressure via Dynamic Relaxation. |

#### Category 07: Quasi-Static
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0701_beam.py` | `1109` | Rate-dependent viscoelastic beam bending under ramp and sustained gravitational loading (`QuasiStaticAnalysis`). |
| `0702_brain.py` | `1103` quasi-static mode | Time-dependent surgical brain tissue relaxation under sustained surgical gravity and intraoperative drainage. |

#### Category 08: Contact
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0801_beam_floor_penalty.py` | `0801`, `0802`, `0803` | Node-to-surface penalty contact of flexible beam impacting imported 3D floor mesh (`floor1.vtk`). Flags: `--floor {rigid,deformable}`, `--analysis {static,dynamic}`. |
| `0802_beam_floor_mortar.py` | `0806`, `0807` | Segment-to-segment mortar contact with exact polygon intersection clipping on rigid and deformable floors. |
| `0803_pole_pad.py` | `0809`, `0810` | Curved contact mechanics between deformable pad (`pad1.vtk`) and cylindrical pole (`pole1.vtk`). Flag: `--pole {rigid,deformable}`. |
| `0804_bunny_self_contact.py` | `0804` | Standalone Stanford bunny (`bunny1.vtk`) self-contact under compound roll/pitch gravity. |
| `0805_cuboid_self_contact.py` | `0805` | Standalone twisted cuboid (`cuboid1.vtk`) self-contact under extreme torsional folding. |
| `0806_frictional_stick_slip.py` | `0808` | Frictional contact mechanics demonstrating Coulomb stick-slip instability and velocity-weakening limit cycles. |

**Phase 2 Verification Gate**:
- Run all 12 scripts in `05_nonlinear_dynamic` through `08_contact`.
- Check that dynamic outputs run 200 steps by default and save cleanly to case subdirectories.
- Verify contact convergence without mesh-toggle branching.

---

### Phase 3: Categories 09 to 11, Automation Sync, & Cutover

#### Category 09: Model Order Reduction (MOR) & Hyper-Reduction
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `0901_beam_reduced.py` | `0901` | Linear POD-Galerkin reduced-order modeling of transient beam vibration. |
| `0902_brain_reduced.py` | `0902` | Linear POD-Galerkin reduced-order modeling of neurosurgical brain shift. |
| `0903_beam_hyperreduced.py` | `0903`, `1301` | Non-intrusive / intrusive hyper-reduction via ECSW and DEIM on nonlinear cantilever beam. |
| `0904_brain_hyperreduced.py` | `0904` | Real-time ECSW hyper-reduced nonlinear anatomical brain shift. |
| `0905_beam_hyperreduced_contact.py` | `1302` | Contact-aware ECSW with enriched contact basis vectors for beam-floor impact. |
| `0906_brain_hyperreduced_contact.py` | `0905`, `1303` | Real-time ECSW hyper-reduced brain shift coupled with skull shell contact boundaries. |

#### Category 10: Inverse FEM & Parameter Identification
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `1001_beam_shape_sensing.py` | `1001`, `1004` | Static & dynamic variational iFEM shape sensing from sparse surface strain/displacement markers. |
| `1002_brain_shape_sensing.py` | `1005` | Clinical cortical marker tracking and real-time subcortical tumor drift reconstruction ($< 0.1\text{ mm}$ error). |
| `1003_beam_param_identification.py` | `1002` | Linear least squares and trust-region reflective calibration of Young's modulus from force-displacement data. |
| `1004_brain_param_identification.py` | `1006`, `1008` | Patient-specific brain tissue stiffness calibration from simulated intraoperative cortical indentation probe. |
| `1005_beam_stress_free.py` | `1003` | Fixed-point iteration with Aitken acceleration recovering undeformed reference configuration under gravity. |
| `1006_brain_stress_free.py` | `1007` | Zero-gravity pre-operative MRI reference recovery eliminating clinical double-gravity tumor bias ($> 10\text{ mm}$). |

#### Category 11: Interactive PyVista Pipelines
| Target File | Source Script(s) | Key Refactoring & Harmonization |
| :--- | :--- | :--- |
| `1101_beam.py` | `0601`, `0602` | Real-time interactive PyVista dynamic simulation supporting bending and axial torsion under live keyboard/mouse controls. |
| `1102_brain.py` | `0603` | Real-time interactive 3D neurosurgical brain shift with continuous gravity reorientation and posture sliders. |
| `1103_brain_twin.py` | `0604`, `1009`, `1010` | Real-time clinical digital twin combining hyper-reduced simulation, multi-modal sensor fusion, and live probe interaction. |

#### Automation & Codebase Synchronization
1. **Runner Script (`psierra/scripts/run_examples.sh`)**:
   - Update regex/patterns to parse new 4-digit IDs (`0101` through `1103`).
   - Support category matching for `01` through `11`.
2. **Manifest (`psierra/scripts/examples.txt`)**:
   - Update active list to match the 30 new canonical scripts.
3. **Tests (`psierra/tests/`)**:
   - Update the 5 subprocess paths in `test_ifem_brain_clinical.py` and `test_analysis_interactive.py`.
4. **Documentation & README**:
   - Update example links and command snippets in `README.md` and `docs/`.

#### Cutover Gate
```bash
# Verify all new examples pass
./scripts/run_examples.sh --config scripts/examples.txt

# Execute atomic directory swap
mv psierra/examples psierra/examples_legacy_archive
mv psierra/examples_v2 psierra/examples
rm -rf psierra/examples_legacy_archive
```

---

## 4. Verification & Progress Tracking Checklist

- [ ] **Phase 1: Categories 01 – 04**
  - [ ] `01_basics` (0101 – 0105)
  - [ ] `02_linear_static` (0201 – 0203)
  - [ ] `03_linear_dynamic` (0301 – 0302, default 100 steps)
  - [ ] `04_nonlinear_static` (0401 – 0402)
  - [ ] Phase 1 Verification passing
- [ ] **Phase 2: Categories 05 – 08**
  - [ ] `05_nonlinear_dynamic` (0501 – 0502, default 200 steps)
  - [ ] `06_dynamic_relaxation` (0601 – 0602)
  - [ ] `07_quasi_static` (0701 – 0702)
  - [ ] `08_contact` (0801 – 0806)
  - [ ] Phase 2 Verification passing
- [ ] **Phase 3: Categories 09 – 11 & Final Cutover**
  - [ ] `09_mor` (0901 – 0906)
  - [ ] `10_inverse` (1001 – 1006)
  - [ ] `11_interactive` (1101 – 1103)
  - [ ] Runner & Test suite synchronization
  - [ ] Full regression check (`pytest -q`)
  - [ ] Atomic directory swap (`examples_v2` $\to$ `examples`)
