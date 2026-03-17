"""quadwild.py — Standalone Python wrapper for QuadWild + Bi-MDF quad-remesher.

Exposes a single class :class:`QuadWild` with one public method:

* ``__init__``  — configure paths to shared libraries and config files.
* ``remesh``    — run the full quad-remeshing pipeline on any trimesh-compatible
  mesh or scene.  The *output_format* parameter controls the return type:
  ``"scene"`` (default), ``"mesh"``, or ``"arrays"``.

The C++ pipeline (three stages)
--------------------------------
1. ``remeshAndField2``  — optional decimation / repair + cross-field computation.
2. ``trace2``           — field tracing and splitting into patches.
3. ``quadPatches``      — ILP-based quadrangulation + optional smoothing.

All intermediate files are written to a private ``tempfile.mkdtemp`` directory
that is cleaned up automatically, even on failure.

Dependencies
------------
* trimesh  ≥ 4.0
* numpy    ≥ 1.24
* scipy    ≥ 1.10  (used internally by trimesh for some mesh-repair operations)
"""

from __future__ import annotations

import contextlib
import enum
import logging
import math
import os
import platform
import shutil
import tempfile
from ctypes import (
    POINTER, Structure, byref, c_bool, c_char_p, c_double, c_float, c_int, cdll,
)
from pathlib import Path
from typing import List, Literal, Optional, Union, overload

import numpy as np
import trimesh

log = logging.getLogger(__name__)

__all__ = ["QuadWild", "QuadWildError", "ILPMethod", "FlowConfig", "SatsumaConfig"]


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------
class ILPMethod(enum.IntEnum):
    """ILP solver variant for stage-3 quadrangulation."""
    LEASTSQUARES = 1
    ABS = 2


class FlowConfig(str, enum.Enum):
    """Flow-solver configuration preset (path relative to *config_dir*)."""
    SIMPLE = "main_config/flow_virtual_simple.json"
    HALF   = "main_config/flow_virtual_half.json"


class SatsumaConfig(str, enum.Enum):
    """Satsuma matching configuration preset (path relative to *config_dir*)."""
    DEFAULT    = "satsuma/default.json"
    MST        = "satsuma/approx-mst.json"
    ROUND2EVEN = "satsuma/approx-round2even.json"
    SYMMDC     = "satsuma/approx-symmdc.json"
    EDGETHRU   = "satsuma/edgethru.json"
    LEMON      = "satsuma/lemon.json"
    NODETHRU   = "satsuma/nodethru.json"

# ---------------------------------------------------------------------------
# ctypes mirror of the C++ parameter structs
# ---------------------------------------------------------------------------

class _Parameters(Structure):
    """Parameters for ``remeshAndField2`` (stage 1)."""

    _fields_ = [
        ("remesh",      c_bool),    # run built-in decimation / repair
        ("sharpAngle",  c_float),   # sharp angle threshold in degrees; -1 ⇒ disabled
        ("alpha",       c_float),   # unused by the lib – kept for ABI compat
        ("scaleFact",   c_float),   # unused by the lib – kept for ABI compat
        ("hasFeature",  c_bool),    # whether a .sharp file exists
        ("hasField",    c_bool),    # whether a pre-computed .rosy field exists
    ]


class _QRParameters(Structure):
    """Parameters for ``quadPatches`` (stage 3 — ILP quadrangulation)."""

    _fields_ = [
        ("useFlowSolver",                            c_bool),
        ("flow_config_filename",                     c_char_p),
        ("satsuma_config_filename",                  c_char_p),
        ("initialRemeshing",                         c_bool),
        ("initialRemeshingEdgeFactor",               c_double),
        ("reproject",                                c_bool),
        ("splitConcaves",                            c_bool),
        ("finalSmoothing",                           c_bool),
        ("ilpMethod",                                c_int),
        ("alpha",                                    c_double),
        ("isometry",                                 c_bool),
        ("regularityQuadrilaterals",                 c_bool),
        ("regularityNonQuadrilaterals",              c_bool),
        ("regularityNonQuadrilateralsWeight",        c_double),
        ("alignSingularities",                       c_bool),
        ("alignSingularitiesWeight",                 c_double),
        ("repeatLosingConstraintsIterations",        c_bool),
        ("repeatLosingConstraintsQuads",             c_bool),
        ("repeatLosingConstraintsNonQuads",          c_bool),
        ("repeatLosingConstraintsAlign",             c_bool),
        ("feasibilityFix",                           c_bool),
        ("hardParityConstraint",                     c_bool),
        ("timeLimit",                                c_double),
        ("gapLimit",                                 c_double),
        ("minimumGap",                               c_double),
        ("callbackTimeLimit",                        POINTER(c_float)),
        ("callbackGapLimit",                         POINTER(c_float)),
        ("chartSmoothingIterations",                 c_int),
        ("quadrangulationFixedSmoothingIterations",  c_int),
        ("quadrangulationNonFixedSmoothingIterations", c_int),
        ("doubletRemoval",                           c_bool),
        ("resultSmoothingIterations",                c_int),
        ("resultSmoothingNRing",                     c_double),
        ("resultSmoothingLaplacianIterations",       c_int),
        ("resultSmoothingLaplacianNRing",            c_double),
    ]


# Minimum number of quads to allocate per geometry when distributing a global
# target_quad_count across a multi-geometry scene.
_MIN_QUADS_PER_PART: int = 50

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------
class QuadWildError(RuntimeError):
    """Raised when the QuadWild pipeline fails at any stage."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class QuadWild:
    """Standalone quad-remesher wrapping the QuadWild + Bi-MDF C++ libraries.

    Parameters
    ----------
    libs_dir:
        Directory containing the platform-appropriate shared libraries:

        * Linux   → ``liblib_quadwild.so``, ``liblib_quadpatches.so``
        * macOS   → ``liblib_quadwild.dylib``, ``liblib_quadpatches.dylib``
        * Windows → ``lib_quadwild.dll``, ``lib_quadpatches.dll``

        Defaults to a ``libs/`` folder one level above this file
        (i.e. ``<repo_root>/libs/``).

    config_dir:
        Directory containing the ``main_config/`` and ``satsuma/``
        sub-directories with the JSON solver-configuration files.
        Defaults to a ``config/`` folder one level above this file
        (i.e. ``<repo_root>/config/``).

    Raises
    ------
    QuadWildError
        If a required library file or config file is not found on disk.
    """

    def __init__(
        self,
        libs_dir: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        here = Path(__file__).parent
        self._libs_dir   = Path(libs_dir)   if libs_dir   else here.parent / "libs"
        self._config_dir = Path(config_dir) if config_dir else here.parent / "config"

        _LIB_NAMES = {
            "Windows": ("lib_quadwild.dll",       "lib_quadpatches.dll"),
            "Darwin":  ("liblib_quadwild.dylib",  "liblib_quadpatches.dylib"),
            "Linux":   ("liblib_quadwild.so",     "liblib_quadpatches.so"),
        }
        qw_name, qp_name = _LIB_NAMES.get(
            platform.system(), ("liblib_quadwild.so", "liblib_quadpatches.so"),
        )
        qw_path, qp_path = (str(self._libs_dir / n) for n in (qw_name, qp_name))

        for lib_path in (qw_path, qp_path):
            if not Path(lib_path).is_file():
                raise QuadWildError(f"Library not found: {lib_path}")

        qw = cdll.LoadLibrary(qw_path)
        qp = cdll.LoadLibrary(qp_path)

        qw.remeshAndField2.argtypes = [POINTER(_Parameters), c_char_p, c_char_p, c_char_p]
        qw.remeshAndField2.restype  = None
        qw.trace2.argtypes          = [c_char_p]
        qw.trace2.restype           = c_bool
        qp.quadPatches.argtypes     = [c_char_p, POINTER(_QRParameters), c_float, c_int, c_bool]
        qp.quadPatches.restype      = c_int

        self._qw, self._qp = qw, qp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @overload
    def remesh(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
        *,
        output_format: Literal["scene"] = ...,
        **kwargs,
    ) -> "trimesh.Scene": ...

    @overload
    def remesh(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
        *,
        output_format: Literal["mesh"],
        **kwargs,
    ) -> "trimesh.Trimesh": ...

    @overload
    def remesh(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
        *,
        output_format: Literal["arrays"],
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def remesh(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
        *,
        # ── Stage-1 (remeshAndField) ────────────────────────────────
        enable_preprocess: bool = True,
        enable_sharp: bool = True,
        sharp_angle: float = 35.0,
        # ── Stage-3 (quadPatches) ───────────────────────────────────
        enable_smoothing: bool = True,
        scale_factor: float = 1.0,
        target_quad_count: Optional[int] = None,
        fixed_chart_clusters: int = 0,
        # ILP objective
        alpha: float = 0.005,
        ilp_method: ILPMethod = ILPMethod.LEASTSQUARES,
        time_limit: int = 200,
        gap_limit: float = 0.0,
        minimum_gap: float = 0.4,
        isometry: bool = True,
        regularity_quads: bool = True,
        regularity_non_quads: bool = True,
        regularity_non_quads_weight: float = 0.9,
        align_singularities: bool = True,
        align_singularities_weight: float = 0.1,
        repeat_losing_iterations: bool = True,
        repeat_losing_quads: bool = False,
        repeat_losing_non_quads: bool = False,
        repeat_losing_align: bool = True,
        hard_parity_constraint: bool = True,
        # Solver config presets
        flow_config: FlowConfig = FlowConfig.SIMPLE,
        satsuma_config: SatsumaConfig = SatsumaConfig.LEMON,
        # Callback schedule (8-element lists; defaults are used when None)
        callback_time_limit: Optional[List[float]] = None,
        callback_gap_limit: Optional[List[float]] = None,
        # ── Processing strategy ────────────────────────────────────
        merge_geometries: bool = False,
        # Output format
        output_format: Literal["arrays", "mesh", "scene"] = "scene",
        # Diagnostics
        debug_dir: Optional[Union[str, Path]] = None,
    ) -> Union["trimesh.Scene", "trimesh.Trimesh", tuple[np.ndarray, np.ndarray]]:
        """Run the full QuadWild + Bi-MDF quad-remeshing pipeline.

        *output_format* controls the return type:

        * ``"scene"`` (default) — returns a :class:`trimesh.Scene`.  Multi-geometry
          inputs are processed per-geometry unless *merge_geometries* is ``True``.
        * ``"mesh"`` — returns a single :class:`trimesh.Trimesh`.  Multi-geometry
          inputs are always merged before processing.
        * ``"arrays"`` — returns raw ``(vertices, faces)`` numpy arrays with
          quad faces preserved as 4-element rows.  Multi-geometry inputs are
          always merged before processing.

        Parameters
        ----------
        mesh:
            Input geometry — accepts a file path (str / Path), a
            :class:`trimesh.Trimesh`, or a :class:`trimesh.Scene`.
        enable_preprocess:
            Run QuadWild's built-in decimation, triangulation, and geometry
            repair (stage 1).  Disabling this skips straight to field
            computation on the raw mesh.
        enable_sharp:
            Generate a ``.sharp`` feature file from edges whose dihedral
            angle exceeds *sharp_angle* and from boundary edges.  These
            features steer the cross-field and improve edge flow in the
            output quad mesh.
        sharp_angle:
            Dihedral-angle threshold in **degrees**.  Edges above this value
            are marked as sharp features.  Default is ``35.0``.
        enable_smoothing:
            Apply post-quadrangulation Laplacian smoothing (stage 3).
        scale_factor:
            Controls output quad density.  ``> 1`` → larger quads (lower
            poly-count); ``< 1`` → smaller quads (higher detail).  Ignored
            when *target_quad_count* is set.
        target_quad_count:
            If provided, overrides *scale_factor* with an automatically
            estimated value that targets approximately this many output quads.
            The estimate is based on the input face count and is best-effort;
            the actual count may differ due to ILP constraints and topology.
        fixed_chart_clusters:
            Force a fixed number of chart clusters.  ``0`` lets the solver
            decide automatically.
        alpha:
            Blend coefficient between isometry (``alpha``) and regularity
            (``1 − alpha``) in the ILP objective.  Default ``0.005``.
        ilp_method:
            ILP solver variant.  See :data:`ILPMethod` for accepted values.
        time_limit:
            Hard time limit in seconds for the ILP solver.
        gap_limit:
            ILP stops early when the optimality gap reaches this value.
            ``0.0`` disables early stopping.
        minimum_gap:
            The ILP must achieve at least this optimality gap.
        flow_config:
            Flow-solver preset.  See :data:`FlowConfig` for accepted values.
        satsuma_config:
            Satsuma matching preset.  See :data:`SatsumaConfig` for accepted
            values.
        callback_time_limit:
            8-element list of callback time checkpoints (seconds).  Uses a
            sensible default when *None*.
        callback_gap_limit:
            8-element list of gap thresholds corresponding to each callback
            checkpoint.  Uses a sensible default when *None*.
        debug_dir:
            Optional path to a directory where **all intermediate pipeline
            files** are copied after each stage, keeping their original names.
            The directory is created if it does not exist.

            Files saved (when they exist)::

                00_input.obj               — mesh sent to remeshAndField2
                01_sharp.sharp             — sharp-feature file
                02_remeshed.obj            — output of remeshAndField2
                03_traced.obj              — output of trace2
                04_quadrangulation.obj     — raw quadrangulation
                05_quadrangulation_smooth.obj — after Laplacian smoothing

            Known differences vs. QRemeshify (Blender addon)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Comparing these debug files with the corresponding files saved by
            QRemeshify can pinpoint why results diverge:

            * **Triangulation**: QRemeshify calls ``bmesh.ops.triangulate``
              (SHORT_EDGE / BEAUTY) before export.  This wrapper does *not*
              explicitly triangulate — it relies on the input mesh already
              being triangular.  Non-triangular input will produce different
              or broken ``remeshAndField2`` results.
            * **Sharp-edge sources**: QRemeshify marks edges as sharp when
              they are a dihedral crease, boundary, *UV seam*, *material
              boundary*, or *sculpt face-set boundary*.  This wrapper only
              uses dihedral angle and boundary.  The ``01_sharp.sharp`` file
              will therefore typically contain fewer entries than Blender's.
            * **Mesh cleanup**: vertex merging, degenerate-face removal, and
              normal fixes are currently commented out in ``_preprocess``.
              Blender's bmesh always produces a clean, consistently-wound mesh.
            * **Coordinate space**: QRemeshify strips the object's translation
              (applies only rotation + scale) so the mesh is centred near the
              origin.  This wrapper processes the mesh in whatever space it
              was loaded in; large translations can cause floating-point
              precision issues inside the C++ solver.

        Returns
        -------
        trimesh.Scene | trimesh.Trimesh | tuple[np.ndarray, np.ndarray]
            Depends on *output_format*:

            * ``"scene"`` → :class:`trimesh.Scene` (geometry key ``"mesh"``).
            * ``"mesh"``  → :class:`trimesh.Trimesh`.
            * ``"arrays"`` → ``(vertices, faces)`` numpy arrays.

        Raises
        ------
        QuadWildError
            If input validation fails or any pipeline stage fails.
        KeyError
            If an unsupported enum name is passed as a string.
        """
        # Coerce plain strings (e.g. from JSON) to enum members by name.
        if not isinstance(ilp_method, ILPMethod):
            ilp_method = ILPMethod[ilp_method]
        if not isinstance(flow_config, FlowConfig):
            flow_config = FlowConfig[flow_config]
        if not isinstance(satsuma_config, SatsumaConfig):
            satsuma_config = SatsumaConfig[satsuma_config]

        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        _CB_LEN = 8
        cb_time = callback_time_limit or [3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0]
        cb_gap  = callback_gap_limit  or [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
        if len(cb_time) != _CB_LEN:
            raise QuadWildError(
                f"callback_time_limit must have exactly {_CB_LEN} elements, got {len(cb_time)}"
            )
        if len(cb_gap) != _CB_LEN:
            raise QuadWildError(
                f"callback_gap_limit must have exactly {_CB_LEN} elements, got {len(cb_gap)}"
            )

        pipeline_kwargs = dict(
            enable_preprocess=enable_preprocess,
            enable_sharp=enable_sharp,
            sharp_angle=sharp_angle,
            enable_smoothing=enable_smoothing,
            scale_factor=scale_factor,
            fixed_chart_clusters=fixed_chart_clusters,
            alpha=alpha,
            ilp_method=ilp_method,
            time_limit=time_limit,
            gap_limit=gap_limit,
            minimum_gap=minimum_gap,
            isometry=isometry,
            regularity_quads=regularity_quads,
            regularity_non_quads=regularity_non_quads,
            regularity_non_quads_weight=regularity_non_quads_weight,
            align_singularities=align_singularities,
            align_singularities_weight=align_singularities_weight,
            repeat_losing_iterations=repeat_losing_iterations,
            repeat_losing_quads=repeat_losing_quads,
            repeat_losing_non_quads=repeat_losing_non_quads,
            repeat_losing_align=repeat_losing_align,
            hard_parity_constraint=hard_parity_constraint,
            flow_config=flow_config,
            satsuma_config=satsuma_config,
            callback_time_limit=cb_time,
            callback_gap_limit=cb_gap,
        )

        # Multi-geometry scene path (only for output_format="scene")
        if output_format == "scene":
            scene       = self._to_scene(mesh)
            geom_meshes = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.Trimesh)}
            log.info(f"[remesh] Input scene has {len(geom_meshes)} geometries")
            if not merge_geometries and len(geom_meshes) > 1:
                return self._quadrangulate_scene(
                    scene,
                    target_quad_count=target_quad_count,
                    debug_dir=debug_dir,
                    **pipeline_kwargs,
                )
            tri_mesh = self._to_mesh(scene)
        else:
            tri_mesh = self._to_mesh(mesh)

        verts, faces = self._quadrangulate_mesh(
            tri_mesh,
            target_quad_count=target_quad_count,
            debug_dir=debug_dir,
            **pipeline_kwargs,
        )
        if debug_dir is not None:
            log.info("[remesh]  intermediate files saved to %s", debug_dir)

        if output_format == "arrays":
            return verts, faces

        result_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if output_format == "scene":
            return trimesh.Scene({"mesh": result_mesh})
        return result_mesh

    # ------------------------------------------------------------------
    # Private — pipeline helpers
    # ------------------------------------------------------------------
    def _quadrangulate_scene(
        self,
        scene: "trimesh.Scene",
        *,
        target_quad_count: Optional[int],
        debug_dir: Optional[Path],
        **pipeline_kwargs,
    ) -> "trimesh.Scene":
        """Process each :class:`trimesh.Trimesh` geometry in *scene* sequentially.

        *target_quad_count* is distributed proportionally across geometries by
        face count.  All other solver parameters are forwarded via
        *pipeline_kwargs* to :meth:`_quadrangulate_mesh`.
        """
        geom_meshes = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.Trimesh)}

        total_verts = sum(len(g.vertices) for g in geom_meshes.values())
        total_faces = sum(len(g.faces) for g in geom_meshes.values())

        log.info(f"[_quadrangulate_scene]  geometries={len(geom_meshes)}  total_vertices={total_verts}  total_faces={total_faces}")

        # Skip geometries with zero faces — they can't be quad-remeshed.
        geom_meshes = {k: v for k, v in geom_meshes.items() if len(v.faces) > 0}
        if not geom_meshes:
            log.warning("[_quadrangulate_scene]  all geometries have zero faces — returning empty scene")
            return scene.copy()

        if target_quad_count is not None and target_quad_count > 0:
            per_geom_target: dict[str, Optional[int]] = {
                name: max(_MIN_QUADS_PER_PART, round(target_quad_count * len(geom.faces) / total_faces))
                for name, geom in geom_meshes.items()
            }
        else:
            per_geom_target = {name: None for name in geom_meshes}

        log.info(f"[_quadrangulate_scene] target_quad_count={target_quad_count} → per-geometry targets: {per_geom_target}")

        def _remesh_process(name: str, geom: "trimesh.Trimesh") -> tuple:
            nv, nf = len(geom.vertices), len(geom.faces)
            pv = 100.0 * nv / total_verts if total_verts > 0 else 0.0
            pf = 100.0 * nf / total_faces if total_faces > 0 else 0.0
            log.info(
                f"[_remesh_process] name={name} | vertices={nv} ({pv:.1f}%) | faces={nf} ({pf:.1f}%)"
            )
            geom_debug = (debug_dir / name) if debug_dir else None
            if geom_debug:
                geom_debug.mkdir(parents=True, exist_ok=True)
            v, f = self._quadrangulate_mesh(
                geom.copy(),
                target_quad_count=per_geom_target[name],
                debug_dir=geom_debug,
                **pipeline_kwargs,
            )
            result = trimesh.Trimesh(vertices=v, faces=f, process=False)
            log.info(f"[_remesh_process]  done  name={name}")
            if geom_debug:
                log.info(f"[_remesh_process]  name={name} files saved to {geom_debug}")
            return name, result

        remeshed = dict(_remesh_process(name, geom) for name, geom in geom_meshes.items())

        new_scene = scene.copy()
        new_scene.geometry.update(remeshed)
        return new_scene

    def _quadrangulate_mesh(
        self,
        mesh: "trimesh.Trimesh",
        *,
        target_quad_count: Optional[int],
        debug_dir: Optional[Path],
        enable_preprocess: bool,
        enable_sharp: bool,
        sharp_angle: float,
        enable_smoothing: bool,
        scale_factor: float,
        fixed_chart_clusters: int,
        alpha: float,
        ilp_method: ILPMethod,
        time_limit: int,
        gap_limit: float,
        minimum_gap: float,
        isometry: bool,
        regularity_quads: bool,
        regularity_non_quads: bool,
        regularity_non_quads_weight: float,
        align_singularities: bool,
        align_singularities_weight: float,
        repeat_losing_iterations: bool,
        repeat_losing_quads: bool,
        repeat_losing_non_quads: bool,
        repeat_losing_align: bool,
        hard_parity_constraint: bool,
        flow_config: FlowConfig,
        satsuma_config: SatsumaConfig,
        callback_time_limit: List[float],
        callback_gap_limit: List[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full three-stage C++ pipeline on a single :class:`trimesh.Trimesh`.

        Returns ``(vertices, faces)`` numpy arrays.  Quad faces remain as
        4-element rows.
        """
        # Defensive copy so the caller's mesh is never mutated.
        mesh = mesh.copy()
        if len(mesh.faces) == 0:
            raise QuadWildError("Input mesh has no faces")
        if mesh.faces.shape[1] != 3:
            raise QuadWildError("Input mesh must be strictly triangulated")
        # ── Nested helpers ────────────────────────────────────────────────────
        @contextlib.contextmanager
        def _suppress_c_output():
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_fds = {}
            try:
                for fd in (1, 2):
                    saved = os.dup(fd)
                    saved_fds[fd] = saved
                    os.dup2(devnull_fd, fd)
                yield
            finally:
                os.close(devnull_fd)
                for fd, saved in saved_fds.items():
                    os.dup2(saved, fd)
                    os.close(saved)

        def _c_ctx():
            return contextlib.nullcontext() if log.isEnabledFor(logging.DEBUG) else _suppress_c_output()

        def _get_boundary(m):
            if len(m.faces) == 0:
                return set()
            counts = np.bincount(m.edges_unique_inverse, minlength=len(m.edges_unique))
            b_edges = m.edges_unique[counts == 1]
            verts = np.round(m.vertices, decimals=6)
            return {frozenset((tuple(verts[e[0]]), tuple(verts[e[1]]))) for e in b_edges}

        def _estimate_sf(n_faces, tqc):
            if n_faces == 0 or tqc <= 0:
                return 1.0
            return max(0.01, min(10.0, math.sqrt(n_faces / (2.0 * tqc))))

        def _count_obj_elements(obj_file):
            """Count vertices and faces in an OBJ file by scanning line prefixes."""
            nv = nf = 0
            with open(obj_file) as fh:
                for ln in fh:
                    if ln.startswith("v "):
                        nv += 1
                    elif ln.startswith("f "):
                        nf += 1
            return nv, nf

        def _preprocess(m):
            """Run built-in decimation, triangulation, and geometry repair."""
            dv = 6
            m.merge_vertices(
                merge_tex=False, 
                merge_norm=False, 
                digits_vertex=dv,
                digits_norm=max(1, dv - 1), 
                digits_uv=max(1, dv - 1)
            )
            m.update_faces(m.nondegenerate_faces())
            m.update_faces(m.unique_faces())
            m.remove_unreferenced_vertices()
            
            # smooth normals
            m.vertex_normals = trimesh.geometry.weighted_vertex_normals(
                vertex_count=len(m.vertices),
                faces=m.faces,
                face_normals=m.face_normals,
                face_angles=m.face_angles,
            )
            # fix normals and winding to ensure consistent orientation (important for remeshAndField2)
            trimesh.repair.fix_normals(m)
            trimesh.repair.fix_winding(m)
            return m

        def _call_stage1(obj_path, sharp_path, field_path):
            """Call remeshAndField2 with the given parameters and file paths."""
            params = _Parameters(
                remesh=enable_preprocess,
                sharpAngle=sharp_angle if enable_sharp else -1.0,
                alpha=0.01,
                scaleFact=1.0,
                hasFeature=enable_sharp,
                hasField=False,
            )
            try:
                with _c_ctx():
                    self._qw.remeshAndField2(
                        byref(params),
                        obj_path.encode(),
                        sharp_path.encode(),
                        field_path.encode(),
                    )
            except Exception as exc:
                raise QuadWildError("remeshAndField2 failed") from exc

        def _call_stage2(remeshed_base):
            """Call trace2 with the given remeshed base path."""
            try:
                with _c_ctx():
                    ok = self._qw.trace2(remeshed_base.encode())
            except Exception as exc:
                raise QuadWildError("trace2 failed") from exc
            if not ok:
                raise QuadWildError(
                    "trace2 returned False — field tracing failed. "
                    "This often means the mesh has degenerate geometry or too few faces."
                )

        def _call_stage3(traced_path, sf):
            """Call quadPatches with the given parameters and file paths."""
            cb_time_arr = (c_float * len(callback_time_limit))(*callback_time_limit)
            cb_gap_arr  = (c_float * len(callback_gap_limit))(*callback_gap_limit)
            p = _QRParameters()
            p.useFlowSolver                              = True
            p.initialRemeshing                           = True
            p.initialRemeshingEdgeFactor                 = 1.0
            p.reproject                                  = True
            p.splitConcaves                              = False
            p.finalSmoothing                             = True
            p.doubletRemoval                             = True
            p.feasibilityFix                             = False
            p.chartSmoothingIterations                   = 0
            p.quadrangulationFixedSmoothingIterations    = 0
            p.quadrangulationNonFixedSmoothingIterations = 0
            p.resultSmoothingIterations                  = 5
            p.resultSmoothingNRing                       = 3.0
            p.resultSmoothingLaplacianIterations         = 2
            p.resultSmoothingLaplacianNRing              = 3.0

            flow_cfg_path = self._config_dir / flow_config.value
            satsuma_cfg_path = self._config_dir / satsuma_config.value

            if not flow_cfg_path.is_file():
                raise QuadWildError(f"Flow config file not found: {flow_cfg_path}")
            if not satsuma_cfg_path.is_file():
                raise QuadWildError(f"Satsuma config file not found: {satsuma_cfg_path}")

            # Keep strong references so the bytes survive until the C call returns.
            flow_cfg_bytes    = str(flow_cfg_path).encode()
            satsuma_cfg_bytes = str(satsuma_cfg_path).encode()
            p.flow_config_filename    = flow_cfg_bytes
            p.satsuma_config_filename = satsuma_cfg_bytes
            p.alpha                             = alpha
            p.ilpMethod                         = ilp_method.value
            p.timeLimit                         = float(time_limit)
            p.gapLimit                          = gap_limit
            p.minimumGap                        = minimum_gap
            p.isometry                          = isometry
            p.regularityQuadrilaterals          = regularity_quads
            p.regularityNonQuadrilaterals       = regularity_non_quads
            p.regularityNonQuadrilateralsWeight = regularity_non_quads_weight
            p.alignSingularities                = align_singularities
            p.alignSingularitiesWeight          = align_singularities_weight
            p.repeatLosingConstraintsIterations = repeat_losing_iterations
            p.repeatLosingConstraintsQuads      = repeat_losing_quads
            p.repeatLosingConstraintsNonQuads   = repeat_losing_non_quads
            p.repeatLosingConstraintsAlign      = repeat_losing_align
            p.hardParityConstraint              = hard_parity_constraint
            p.callbackTimeLimit                 = cb_time_arr
            p.callbackGapLimit                  = cb_gap_arr
            try:
                with _c_ctx():
                    return self._qp.quadPatches(
                        traced_path.encode(),
                        byref(p),
                        c_float(sf),
                        c_int(fixed_chart_clusters),
                        c_bool(enable_smoothing),
                    )
            except Exception as exc:
                raise QuadWildError("quadPatches failed") from exc

        # ── Pipeline ──────────────────────────────────────────────────────────
        pre_merge_boundary = _get_boundary(mesh) if enable_sharp else set()
        if enable_preprocess:
            mesh = _preprocess(mesh)

        log.info(
            f"[stage 0 / input]  "
            f"vertices={len(mesh.vertices)} "
            f"faces={len(mesh.faces)} "
            f"watertight={mesh.is_watertight} "
            f"scale_factor={scale_factor:.4f} "
            f"target_quad_count={target_quad_count}"
        )

        if not mesh.is_watertight:
            if enable_preprocess:
                log.debug(
                    "[stage 0 / input]  mesh is not watertight — remeshAndField2 will repair it"
                )
            else:
                log.warning(
                    "[stage 0 / input]  mesh is not watertight and Preprocess is disabled — "
                    "consider enabling Preprocess so remeshAndField2 can repair the mesh"
                )

        current_scale_factor = scale_factor
        if target_quad_count is not None and target_quad_count > 0:
            current_scale_factor = _estimate_sf(len(mesh.faces), target_quad_count)
            log.info(
                "[stage 0 / input]  target_quad_count=%d → initial scale_factor=%.4f "
                "(based on input faces=%d; will be refined after stage 1)",
                target_quad_count, current_scale_factor, len(mesh.faces),
            )

        log.info(
            "[stage 0 / params]  enable_preprocess=%s  enable_sharp=%s  "
            "sharp_angle=%.1f  enable_smoothing=%s  scale_factor=%.4f  "
            "ilp_method=%s  alpha=%.4f  time_limit=%d  flow_config=%s  satsuma_config=%s",
            enable_preprocess, enable_sharp, sharp_angle, enable_smoothing,
            current_scale_factor, ilp_method, alpha, time_limit, flow_config, satsuma_config,
        )

        tmpdir = Path(tempfile.mkdtemp(prefix="quadwild_"))
        try:
            base       = str(tmpdir / "mesh")
            obj_path   = base + ".obj"
            sharp_path = base + "_rem.sharp"
            field_path = base + "_rem.rosy"
            remeshed_base = base + "_rem"
            traced_path   = base + "_rem_p0.obj"
            out_path      = base + "_rem_p0_0_quadrangulation.obj"
            out_smooth    = base + "_rem_p0_0_quadrangulation_smooth.obj"

            self._export_obj(mesh, obj_path)
            log.info("[stage 0 / export]  wrote input OBJ → %s", obj_path)
            if debug_dir:
                shutil.copy2(obj_path, debug_dir / "00_input.obj")

            if enable_sharp:
                n_features = self._export_sharp(
                    mesh, sharp_path, sharp_angle,
                    pre_merge_boundary_positions=pre_merge_boundary,
                )
                log.info(
                    "[stage 0 / sharp]  sharp edges written = %d  (angle threshold = %.1f°)",
                    n_features, sharp_angle,
                )
                if debug_dir:
                    shutil.copy2(sharp_path, debug_dir / "01_sharp.sharp")
            else:
                n_features = 0
                log.info("[stage 0 / sharp]  sharp detection disabled")

            log.info("[stages 1–3]  running C++ pipeline …")
            _call_stage1(obj_path, sharp_path, field_path)

            n_remeshed_verts = 0
            n_remeshed_faces = 0
            remeshed_obj = Path(remeshed_base + ".obj")
            if remeshed_obj.is_file():
                n_remeshed_verts, n_remeshed_faces = _count_obj_elements(remeshed_obj)
                if target_quad_count is not None and target_quad_count > 0 and n_remeshed_faces > 0:
                    refined_sf = _estimate_sf(n_remeshed_faces, target_quad_count)
                    log.info(
                        "[stage 1]  remeshed faces=%d → refined scale_factor=%.4f "
                        "(was %.4f based on input faces=%d)",
                        n_remeshed_faces, refined_sf, current_scale_factor, len(mesh.faces),
                    )
                    current_scale_factor = refined_sf

            _call_stage2(remeshed_base)
            if (stage3_rc := _call_stage3(traced_path, current_scale_factor)) != 0:
                raise QuadWildError(
                    f"quadPatches returned non-zero exit code: {stage3_rc}"
                )
            log.info("[stages 1–3 / done]")

            if remeshed_obj.is_file():
                log.info("[stage 1]  remeshed mesh: vertices=%d  faces=%d", n_remeshed_verts, n_remeshed_faces)
                if debug_dir:
                    shutil.copy2(remeshed_obj, debug_dir / "02_remeshed.obj")
            traced_obj = Path(traced_path)
            if traced_obj.is_file():
                tv, tf = _count_obj_elements(traced_obj)
                log.info("[stage 2]  traced mesh:   vertices=%d  faces=%d", tv, tf)
                if debug_dir:
                    shutil.copy2(traced_obj, debug_dir / "03_traced.obj")
            if Path(out_path).is_file() and debug_dir:
                shutil.copy2(out_path, debug_dir / "04_quadrangulation.obj")
            if Path(out_smooth).is_file() and debug_dir:
                shutil.copy2(out_smooth, debug_dir / "05_quadrangulation_smooth.obj")

            result_path = out_smooth if (enable_smoothing and Path(out_smooth).is_file()) else out_path

            n_quads = n_tris_raw = 0
            with open(result_path) as _f:
                for _line in _f:
                    if _line.startswith("f "):
                        n_tokens = len(_line.split()) - 1
                        if n_tokens == 4:
                            n_quads += 1
                        elif n_tokens == 3:
                            n_tris_raw += 1

            verts_out, faces_out = self._import_obj(result_path)
            log.info(
                "[stage 3 / done]  smoothed=%s  "
                "quads=%d  tris=%d  (total C++ faces=%d)  vertices=%d",
                enable_smoothing and Path(out_smooth).is_file(),
                n_quads, n_tris_raw, n_quads + n_tris_raw,
                len(verts_out),
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return verts_out, faces_out

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------
    def _to_scene(
        self, 
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"]
    ) -> "trimesh.Scene":
        """Normalise any supported input type to a trimesh.Scene."""
        if isinstance(mesh, (Path, str)):
            return trimesh.load(mesh, force='scene', process=False, merge_primitives=False, skip_materials=False, maintain_order=True)
        elif isinstance(mesh, trimesh.Scene):
            return mesh
        elif isinstance(mesh, trimesh.Trimesh):
            return trimesh.Scene(mesh)
        else:
            raise TypeError(f"Input must be a file path, Trimesh, or Scene; got {type(mesh)}")

    def _to_mesh(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
    ) -> "trimesh.Trimesh":
        """Flatten any supported input type to a single :class:`trimesh.Trimesh`.

        Accepts a file path (``str`` / ``Path``), a :class:`trimesh.Trimesh`,
        or a :class:`trimesh.Scene`.  For scenes and scene files, all Trimesh
        geometries are concatenated with their scene-graph transforms baked in.
        """
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        elif isinstance(mesh, (Path, str)):
            loaded = trimesh.load(mesh, force='mesh', process=False)
            if isinstance(loaded, trimesh.Trimesh):
                return loaded
            elif isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) == 0:
                    raise ValueError("No geometry found in the scene.")
                # Use to_geometry() to correctly bake in transforms from the scene graph
                return loaded.to_geometry() if hasattr(loaded, 'to_geometry') else loaded.dump(concatenate=True)
            else:
                raise TypeError(f"Loaded object is neither a Trimesh nor a Scene: {type(loaded)}")
        elif isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("No geometry found in the scene.")
            # Use to_geometry() to correctly bake in transforms from the scene graph
            return mesh.to_geometry() if hasattr(mesh, 'to_geometry') else mesh.dump(concatenate=True)
        elif mesh is None:
            raise ValueError("Input cannot be None")
        else:
            raise TypeError(f"Input must be a Trimesh, Scene, or file path; got {type(mesh)}")

    # ------------------------------------------------------------------
    # Private — IO helpers
    # ------------------------------------------------------------------
    def _import_obj(
        self,
        obj_path: str,
        output_format: Literal["mesh", "arrays"] = "arrays",
    ) -> Union["trimesh.Trimesh", tuple[np.ndarray, np.ndarray]]:
        """Parse an OBJ file and return either raw arrays or a :class:`trimesh.Trimesh`.

        Faces are returned as-is (quads remain as 4-element rows when
        *output_format* is ``"arrays"``).  Pass ``output_format="mesh"`` to get a
        :class:`trimesh.Trimesh` instead (trimesh will triangulate quads).
        """
        if not Path(obj_path).is_file():
            raise QuadWildError(f"Expected output file not found: {obj_path}")
        with open(obj_path) as fh:
            tokens_by_line = [t for line in fh if (t := line.split())]
        verts = [[float(x) for x in t[1:4]] for t in tokens_by_line if t[0] == "v"]
        faces = [[int(tok.split("/")[0]) - 1 for tok in t[1:]] for t in tokens_by_line if t[0] == "f"]
        v = np.array(verts, dtype=np.float64)

        if output_format == "mesh":
            # Triangulate quads (and n-gons) so trimesh gets a uniform Nx3 array.
            tris = [
                [face[0], face[i], face[i + 1]]
                for face in faces
                for i in range(1, len(face) - 1)
            ]
            return trimesh.Trimesh(
                vertices=v,
                faces=np.array(tris, dtype=np.int64),
                process=False,
            )

        # "arrays" — preserve quads; pad shorter faces so the array is rectangular.
        face_lengths = {len(face) for face in faces}
        if len(face_lengths) <= 1:
            f = np.array(faces, dtype=np.int64)
        else:
            max_len = max(face_lengths)
            f = np.array(
                [face + [face[-1]] * (max_len - len(face)) for face in faces],
                dtype=np.int64,
            )
        return v, f

    def _export_obj(
        self,
        mesh: Union["trimesh.Trimesh", tuple[np.ndarray, np.ndarray]],
        obj_path: str,
    ) -> None:
        """Write a minimal OBJ file with per-face normals.

        *mesh* may be a :class:`trimesh.Trimesh` **or** a
        ``(vertices, faces)`` tuple of numpy arrays.  When raw arrays are
        supplied, per-face normals are computed from the first three vertices
        of each polygon.

        Format::

            v  x y z          (one vertex per line)
            vn nx ny nz       (one face-normal per line, 1-indexed from faces)
            f  vi//ni vi//ni vi//ni   (vertex index // normal index, 1-based)

        This is the exact format consumed by ``remeshAndField2``.
        """
        if isinstance(mesh, trimesh.Trimesh):
            verts, faces, normals = mesh.vertices, mesh.faces, mesh.face_normals
        else:
            verts, faces = mesh
            v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(lens, 1e-10)
        with open(obj_path, "w") as f:
            f.write("# OBJ file\n")
            f.writelines(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
            f.writelines(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n" for n in normals)
            f.writelines(
                f"f {' '.join(f'{int(vi) + 1}//{fi + 1}' for vi in face)}\n"
                for fi, face in enumerate(faces)
            )

    def _export_sharp(
        self,
        mesh: "trimesh.Trimesh",
        sharp_path: str,
        sharp_angle: float,
        *,
        pre_merge_boundary_positions: Optional[set] = None,
    ) -> int:
        """Write the ``.sharp`` feature file consumed by ``remeshAndField2``.

        File format::

            <count>
            <convexity>,<face_index>,<edge_index_in_face>
            ...

        *edge_index_in_face* is the 0-based position of the edge in the face's
        local edge list, following trimesh's adjacency edge ordering:

        * 0 → vertices 0–1 of the face
        * 1 → vertices 1–2 of the face
        * 2 → vertices 0–2 of the face

        Returns the number of sharp edges written.
        """
        if mesh.faces.shape[1] != 3:
            raise QuadWildError("_export_sharp requires a triangulated mesh")

        def edge_in_face(face, ev):
            ev_set = frozenset(ev)
            for i, (a, b) in enumerate(
                ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])),
            ):
                if frozenset((a, b)) == ev_set:
                    return i
            return -1

        def edge_convexity(fi0, fi1):
            n0, c0, c1 = mesh.face_normals[fi0], mesh.triangles_center[fi0], mesh.triangles_center[fi1]
            return 1 if float(np.dot(n0, c1 - c0)) < 0.0 else 0

        threshold_rad = math.radians(sharp_angle)
        entries = []
        added   = set()

        adj        = mesh.face_adjacency
        adj_edges  = mesh.face_adjacency_edges
        adj_angles = mesh.face_adjacency_angles

        for k in np.where(adj_angles > threshold_rad)[0]:
            fi0      = int(adj[k, 0])
            fi1      = int(adj[k, 1])
            ev       = (int(adj_edges[k, 0]), int(adj_edges[k, 1]))
            ei_local = edge_in_face(mesh.faces[fi0], ev)
            if ei_local < 0:
                continue
            entries.append((edge_convexity(fi0, fi1), fi0, ei_local))
            added.add((fi0, ei_local))

        if pre_merge_boundary_positions:
            verts_rounded = np.round(mesh.vertices, decimals=6)
            n_uv = 0
            for k in range(len(adj)):
                pos_pair = frozenset((
                    tuple(verts_rounded[int(adj_edges[k, 0])]),
                    tuple(verts_rounded[int(adj_edges[k, 1])]),
                ))
                if pos_pair not in pre_merge_boundary_positions:
                    continue
                fi0      = int(adj[k, 0])
                fi1      = int(adj[k, 1])
                ev       = (int(adj_edges[k, 0]), int(adj_edges[k, 1]))
                ei_local = edge_in_face(mesh.faces[fi0], ev)
                if ei_local < 0 or (fi0, ei_local) in added:
                    continue
                entries.append((edge_convexity(fi0, fi1), fi0, ei_local))
                added.add((fi0, ei_local))
                n_uv += 1
            if n_uv:
                log.info("[sharp]  UV seam edges added: %d", n_uv)

        n_faces     = len(mesh.faces)
        unique_inv  = mesh.edges_unique_inverse
        edge_counts = np.bincount(unique_inv, minlength=len(mesh.edges_unique))
        is_boundary = edge_counts[unique_inv] == 1

        face_idxs  = np.repeat(np.arange(n_faces), 3)
        local_idxs = np.tile(np.arange(3), n_faces)
        entries.extend(
            (1, fi, ei)
            for fi, ei in zip(face_idxs[is_boundary].tolist(), local_idxs[is_boundary].tolist())
            if (fi, ei) not in added
        )

        with open(sharp_path, "w") as f:
            f.write(f"{len(entries)}\n")
            f.writelines(f"{c},{fi},{ei}\n" for c, fi, ei in entries)
        return len(entries)
