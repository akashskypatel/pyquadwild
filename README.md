# pyquadwild

A Python wrapper for the [QuadWild + Bi-MDF](https://github.com/cgg-bern/quadwild-bimdf) C++ quad-remeshing pipeline, with an optional [Gradio](https://www.gradio.app/) web demo.

Given any triangulated 3-D mesh, it produces a clean quad-dominant mesh by running a three-stage C++ pipeline:

1. **RemeshAndField** — optional decimation, triangulation, and geometry repair, followed by cross-field computation.
2. **Trace** — field tracing and patch decomposition.
3. **QuadPatches** — ILP-based quadrangulation with optional Laplacian smoothing.

---

## Requirements

- Python 3.9+
- Pre-compiled QuadWild shared libraries placed in `libs/` (see [Building the C++ libraries](#building-the-c-libraries))
- Python dependencies:

```
trimesh>=4.0.0
numpy>=1.24.0
scipy>=1.10.0
networkx>=3.0
gradio>=4.0.0   # only needed for the Gradio demo
```

Install them all at once:

```bash
pip install -r requirements.txt
```

Or install directly from GitHub:

```bash
pip install https://github.com/dickoah/pyquadwild
```

---

## Building the C++ Libraries

The Python wrapper loads two platform-specific shared libraries at runtime:

| Platform | Files expected in `libs/` |
|----------|--------------------------|
| Linux    | `liblib_quadwild.so`, `liblib_quadpatches.so` |
| macOS    | `liblib_quadwild.dylib`, `liblib_quadpatches.dylib` |
| Windows  | `lib_quadwild.dll`, `lib_quadpatches.dll` |

Clone and build [quadwild-bimdf](https://github.com/cgg-bern/quadwild-bimdf), then copy the resulting shared libraries into the `libs/` directory of this repository.

---

## Usage

### Python API

```python
import trimesh
from src.quadwild import QuadWild

qw = QuadWild()  # loads libs/ and config/ from the repo root by default

scene = trimesh.load("my_mesh.obj", force="scene")

# Returns a trimesh.Scene containing the quad-remeshed geometry
result = qw.remesh(
    scene,
    enable_preprocess=True,
    enable_sharp=True,
    sharp_angle=35.0,
    scale_factor=1.0,
    # target_quad_count=5000,  # override scale_factor with an approximate quad target
)

result.export("output_quad.glb")
```

To get raw vertex and face arrays instead of a scene:

```python
vertices, faces = qw.quadrangulate(scene)
```

Custom library and config paths:

```python
qw = QuadWild(
    libs_dir="/path/to/libs",
    config_dir="/path/to/config",
)
```

Both `remesh()` and `quadrangulate()` accept a file path (`str` / `Path`), a `trimesh.Trimesh`, or a `trimesh.Scene`. Scenes with multiple geometries are processed per-geometry by default; pass `merge_geometries=True` to merge them into a single mesh first.

### Gradio Web Demo

```bash
python app.py
```

Opens a local web UI where you can upload a mesh, adjust all parameters interactively, and download the resulting quad mesh.

---

## Parameters

### Basic

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_preprocess` | `bool` | `True` | Run built-in decimation, triangulation, and repair before field computation. |
| `enable_sharp` | `bool` | `True` | Detect and mark sharp feature edges to steer the cross-field. |
| `sharp_angle` | `float` | `35.0` | Dihedral-angle threshold in degrees for sharp-edge detection. |
| `enable_smoothing` | `bool` | `True` | Apply Laplacian smoothing after quadrangulation. |
| `scale_factor` | `float` | `1.0` | Controls output quad density. `> 1` → coarser; `< 1` → finer. Ignored when `target_quad_count` is set. |
| `target_quad_count` | `int \| None` | `None` | Approximate target number of output quads. Overrides `scale_factor` when set. |

### ILP Objective

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.005` | Blend between isometry (`α`) and regularity (`1 − α`). |
| `ilp_method` | `str` | `"LEASTSQUARES"` | ILP solver variant: `"LEASTSQUARES"` or `"ABS"`. |
| `time_limit` | `int` | `200` | Hard time limit in seconds for the ILP solver. |
| `gap_limit` | `float` | `0.0` | Stop early when optimality gap reaches this value (`0.0` = disabled). |
| `minimum_gap` | `float` | `0.4` | The ILP must achieve at least this optimality gap. |
| `isometry` | `bool` | `True` | Include isometry term in the ILP objective. |
| `regularity_quads` | `bool` | `True` | Include quad regularity term. |
| `regularity_non_quads` | `bool` | `True` | Include non-quad regularity term. |
| `regularity_non_quads_weight` | `float` | `0.9` | Weight for the non-quad regularity term. |
| `align_singularities` | `bool` | `True` | Align singularities in the output. |
| `align_singularities_weight` | `float` | `0.1` | Weight for singularity alignment. |
| `hard_parity_constraint` | `bool` | `True` | Enforce hard parity constraints in the ILP. |
| `fixed_chart_clusters` | `int` | `0` | Force a fixed number of chart clusters (`0` = auto). |

### Constraint Recovery

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repeat_losing_iterations` | `bool` | `True` | Retry ILP when constraints are lost over iterations. |
| `repeat_losing_quads` | `bool` | `False` | Retry when quad constraints are lost. |
| `repeat_losing_non_quads` | `bool` | `False` | Retry when non-quad constraints are lost. |
| `repeat_losing_align` | `bool` | `True` | Retry when alignment constraints are lost. |

### Solver Presets

| Parameter | Type | Default | Choices |
|-----------|------|---------|---------|
| `flow_config` | `str` | `"SIMPLE"` | `"SIMPLE"`, `"HALF"` |
| `satsuma_config` | `str` | `"DEFAULT"` | `"DEFAULT"`, `"MST"`, `"ROUND2EVEN"`, `"SYMMDC"`, `"EDGETHRU"`, `"LEMON"`, `"NODETHRU"` |

---

## Project Structure

```
pyquadwild/
├── pyproject.toml
├── MANIFEST.in
├── app.py              # Gradio web demo
├── requirements.txt
└── pyquadwild/
    ├── __init__.py
    ├── quadwild.py     # QuadWild Python class (ctypes bindings)
    ├── libs/           # Pre-compiled C++ shared libraries 
    └── config/         # JSON config files for the solvers
    ├── main_config/    # Flow solver configuration files
    └── satsuma/        # Satsuma matching preset files
```

---

## License

See [LICENSE](LICENSE) for details.

---

## Credits & Acknowledgements

This project is built on top of excellent open-source works:

- **[quadwild-bimdf](https://github.com/cgg-bern/quadwild-bimdf)** — the core C++ library for cross-field computation, quadrangulation tracing, and ILP-based quad patching. Published by the Computer Graphics Group, University of Bern.
- **[QRemeshify](https://github.com/ksami/QRemeshify)** — an inspiration and reference for bridging QuadWild's C++ pipeline into a Python-friendly interface.
- **[pyquadwild (PozzettiAndrea)](https://github.com/PozzettiAndrea/pyquadwild)** — Another helpful reference repository.
- **QuadWild** — Pietroni et al., "Reliable Feature-Line Driven Quad-Remeshing" (SIGGRAPH 2021)
