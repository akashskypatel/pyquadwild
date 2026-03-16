"""quadwild.py — Standalone Python wrapper for QuadWild + Bi-MDF quad-remesher.

Exposes a single class :class:`QuadWild` with two public methods:

* ``__init__`` — configure paths to shared libraries and config files.
* ``process``  — run the full quad-remeshing pipeline on any trimesh-compatible
  mesh or scene, returning a :class:`trimesh.Scene`.

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
import logging
import math
import os
import platform
import shutil
import tempfile
from ctypes import (
    POINTER, Structure, byref, c_bool, c_char_p, c_double, c_float, c_int, cdll,
)
from os import path
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import trimesh

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_ILP_METHODS: dict[str, int] = {
    "LEASTSQUARES": 1,
    "ABS": 2,
}

# Paths are relative to ``config_dir`` supplied at construction time.
_FLOW_CONFIGS: dict[str, str] = {
    "SIMPLE": "main_config/flow_virtual_simple.json",
    "HALF":   "main_config/flow_virtual_half.json",
}

_SATSUMA_CONFIGS: dict[str, str] = {
    "DEFAULT":    "satsuma/default.json",
    "MST":        "satsuma/approx-mst.json",
    "ROUND2EVEN": "satsuma/approx-round2even.json",
    "SYMMDC":     "satsuma/approx-symmdc.json",
    "EDGETHRU":   "satsuma/edgethru.json",
    "LEMON":      "satsuma/lemon.json",
    "NODETHRU":   "satsuma/nodethru.json",
}


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

        self._qw, self._qp = self._load_libs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"],
        *,
        # ── Stage-1 (remeshAndField) ────────────────────────────────
        enable_preprocess: bool = True,
        enable_sharp: bool = True,
        sharp_angle: float = 45.0,
        # ── Stage-3 (quadPatches) ───────────────────────────────────
        enable_smoothing: bool = True,
        scale_factor: float = 1.0,
        target_quad_count: Optional[int] = None,
        fixed_chart_clusters: int = 0,
        # ILP objective
        alpha: float = 0.005,
        ilp_method: str = "LEASTSQUARES",
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
        flow_config: str = "SIMPLE",
        satsuma_config: str = "LEMON",
        # Callback schedule (8-element lists; defaults are used when None)
        callback_time_limit: Optional[List[float]] = None,
        callback_gap_limit: Optional[List[float]] = None,
        # ── Processing strategy ────────────────────────────────────
        force_merge: bool = False,
        # Diagnostics
        debug_dir: Optional[Union[str, Path]] = None,
    ) -> "trimesh.Scene":
        """Run the full QuadWild + Bi-MDF quad-remeshing pipeline.

        The processing strategy is chosen automatically based on the input:

        * A :class:`trimesh.Trimesh` or a single-geometry scene is processed
          by :meth:`_process_mesh` (the entire mesh in one shot).
        * A scene with **multiple geometries** is processed by
          :meth:`_process_scene` (each geometry independently; the original
          scene graph is reconstructed with the remeshed geometries).

        Set *force_merge* to ``True`` to override the automatic strategy and
        always merge all geometries into a single mesh before processing.

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
            ILP solver variant — ``"LEASTSQUARES"`` or ``"ABS"``.
        time_limit:
            Hard time limit in seconds for the ILP solver.
        gap_limit:
            ILP stops early when the optimality gap reaches this value.
            ``0.0`` disables early stopping.
        minimum_gap:
            The ILP must achieve at least this optimality gap.
        flow_config:
            Flow-solver preset — ``"SIMPLE"`` or ``"HALF"``.
        satsuma_config:
            Satsuma matching preset — one of ``"DEFAULT"``, ``"MST"``,
            ``"ROUND2EVEN"``, ``"SYMMDC"``, ``"EDGETHRU"``, ``"LEMON"``,
            ``"NODETHRU"``.
        callback_time_limit:
            8-element list of callback time checkpoints (seconds).  Uses a
            sensible default when *None*.
        callback_gap_limit:
            8-element list of gap thresholds corresponding to each callback
            checkpoint.  Uses a sensible default when *None*.
        debug_dir:
            Optional path to a directory where **all intermediate pipeline
            files** are copied after each stage, keeping their original names.
            The directory is created if it does not exist.  Enabling this also
            activates verbose ``logging.DEBUG`` output on the ``quadwild``
            logger (name = ``__name__``).

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
              normal fixes are currently commented out in ``_preprocess_mesh``.
              Blender's bmesh always produces a clean, consistently-wound mesh.
            * **Coordinate space**: QRemeshify strips the object's translation
              (applies only rotation + scale) so the mesh is centred near the
              origin.  This wrapper processes the mesh in whatever space it
              was loaded in; large translations can cause floating-point
              precision issues inside the C++ solver.

        Returns
        -------
        trimesh.Scene
            Quad-remeshed mesh wrapped in a :class:`trimesh.Scene` under the
            key ``"mesh"``.

        Raises
        ------
        QuadWildError
            If input validation fails or any pipeline stage fails.
        ValueError
            If an unsupported enum-style string is passed.
        """
        if ilp_method not in _ILP_METHODS:
            raise ValueError(f"ilp_method must be one of {list(_ILP_METHODS)}, got {ilp_method!r}")
        if flow_config not in _FLOW_CONFIGS:
            raise ValueError(f"flow_config must be one of {list(_FLOW_CONFIGS)}, got {flow_config!r}")
        if satsuma_config not in _SATSUMA_CONFIGS:
            raise ValueError(f"satsuma_config must be one of {list(_SATSUMA_CONFIGS)}, got {satsuma_config!r}")

        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        cb_time = callback_time_limit or [3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0]
        cb_gap  = callback_gap_limit  or [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]

        scene       = self._to_scene(mesh)
        geom_meshes = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.Trimesh)}

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

        if not force_merge and len(geom_meshes) > 1:
            return self._process_scene(
                scene,
                target_quad_count=target_quad_count,
                debug_dir=debug_dir,
                **pipeline_kwargs,
            )

        tri_mesh = self._to_mesh(scene)
        result = self._process_mesh(
            tri_mesh,
            target_quad_count=target_quad_count,
            debug_dir=debug_dir,
            **pipeline_kwargs,
        )
        if debug_dir is not None:
            log.info("[debug]  intermediate files saved to %s", debug_dir)
        return trimesh.Scene({"mesh": result})

    # ------------------------------------------------------------------
    # Private — pipeline helpers
    # ------------------------------------------------------------------

    def _process_scene(
        self,
        scene: "trimesh.Scene",
        *,
        target_quad_count: Optional[int],
        debug_dir: Optional[Path],
        **pipeline_kwargs,
    ) -> "trimesh.Scene":
        """Process each :class:`trimesh.Trimesh` geometry in *scene* in parallel.

        Each geometry is handed off to :meth:`_process_mesh` via a
        :class:`~concurrent.futures.ThreadPoolExecutor`.  Because
        :meth:`_process_mesh` blocks on a C++ subprocess (``proc.join``),
        the GIL is released during the wait and all subprocesses run
        concurrently without pickling or ctypes complications.

        *target_quad_count* is distributed proportionally across geometries by
        face count.  All other solver parameters are forwarded via
        *pipeline_kwargs* to :meth:`_process_mesh`.
        """
        geom_meshes = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.Trimesh)}
        log.info("[_process_scene]  %d geometries to process in parallel", len(geom_meshes))

        _MIN_QUADS_PER_PART = 50

        if target_quad_count is not None and target_quad_count > 0:
            total_faces = sum(len(g.faces) for g in geom_meshes.values())
            if total_faces > 0:
                per_geom_target: dict = {
                    name: max(_MIN_QUADS_PER_PART, round(target_quad_count * len(geom.faces) / total_faces))
                    for name, geom in geom_meshes.items()
                }
            else:
                equal_share = max(_MIN_QUADS_PER_PART, target_quad_count // len(geom_meshes))
                per_geom_target = {name: equal_share for name in geom_meshes}
        else:
            per_geom_target = {name: None for name in geom_meshes}

        def _remesh_process(name: str, geom: "trimesh.Trimesh") -> tuple:
            log.info("[_process_scene]  processing geometry %r …", name)
            geom_debug = (debug_dir / name) if debug_dir else None
            if geom_debug:
                geom_debug.mkdir(parents=True, exist_ok=True)
            result = self._process_mesh(
                geom.copy(),
                target_quad_count=per_geom_target[name],
                debug_dir=geom_debug,
                **pipeline_kwargs,
            )
            log.info("[_process_scene]  done  geometry %r", name)
            if geom_debug:
                log.info("[_process_scene]  geometry %r files saved to %s", name, geom_debug)
            return name, result

        remeshed = dict([_remesh_process(name, geom) for name, geom in geom_meshes.items()])

        new_scene = scene.copy()
        new_scene.geometry.update(remeshed)
        return new_scene

    def _process_mesh(
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
        ilp_method: str,
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
        flow_config: str,
        satsuma_config: str,
        callback_time_limit: List[float],
        callback_gap_limit: List[float],
    ) -> "trimesh.Trimesh":
        """Run the full three-stage C++ pipeline on a single mesh.

        Returns the quad-remeshed :class:`trimesh.Trimesh`.
        """
        # Detect UV seam edges *before* vertex merge collapses them.
        pre_merge_boundary = self._get_boundary(mesh) if enable_sharp else set()

        mesh = self._preprocess_mesh(mesh)

        log.info(
            "[stage 0 / input]  vertices=%d  faces=%d  watertight=%s",
            len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
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
            current_scale_factor = self._estimate_scale_factor(mesh, target_quad_count)
            log.info(
                "[stage 0 / input]  target_quad_count=%d → scale_factor=%.4f",
                target_quad_count, current_scale_factor,
            )

        log.info(
            "[stage 0 / params]  enable_preprocess=%s  enable_sharp=%s  "
            "sharp_angle=%.1f  enable_smoothing=%s  scale_factor=%.4f  "
            "ilp_method=%s  alpha=%.4f  time_limit=%d  flow_config=%s  satsuma_config=%s",
            enable_preprocess, enable_sharp, sharp_angle, enable_smoothing,
            current_scale_factor, ilp_method, alpha, time_limit, flow_config, satsuma_config,
        )

        tmpdir = tempfile.mkdtemp(prefix="quadwild_")
        try:
            base       = path.join(tmpdir, "mesh")
            obj_path   = base + ".obj"
            sharp_path = base + "_rem.sharp"
            field_path = base + "_rem.rosy"
            # Derived by convention — the C++ libs write these themselves
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
            self._call_remesh_and_field(
                obj_path=obj_path,
                sharp_path=sharp_path,
                field_path=field_path,
                enable_preprocess=enable_preprocess,
                enable_sharp=enable_sharp and n_features > 0,
                sharp_angle=sharp_angle,
            )
            self._call_trace(remeshed_base)
            self._call_quadrangulate(
                traced_path=traced_path,
                enable_smoothing=enable_smoothing,
                scale_factor=current_scale_factor,
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
                callback_time_limit=callback_time_limit,
                callback_gap_limit=callback_gap_limit,
            )
            log.info("[stages 1–3 / done]")

            remeshed_path = remeshed_base + ".obj"
            if path.isfile(remeshed_path):
                rem = self._import_obj(remeshed_path)
                log.info(
                    "[stage 1]  remeshed mesh: vertices=%d  faces=%d",
                    len(rem.vertices), len(rem.faces),
                )
                if debug_dir:
                    shutil.copy2(remeshed_path, debug_dir / "02_remeshed.obj")
            if path.isfile(traced_path):
                tr = self._import_obj(traced_path)
                log.info(
                    "[stage 2]  traced mesh:   vertices=%d  faces=%d",
                    len(tr.vertices), len(tr.faces),
                )
                if debug_dir:
                    shutil.copy2(traced_path, debug_dir / "03_traced.obj")
            if path.isfile(out_path) and debug_dir:
                shutil.copy2(out_path, debug_dir / "04_quadrangulation.obj")
            if path.isfile(out_smooth) and debug_dir:
                shutil.copy2(out_smooth, debug_dir / "05_quadrangulation_smooth.obj")

            result_path = out_smooth if (enable_smoothing and path.isfile(out_smooth)) else out_path
            result_mesh = self._import_obj(result_path)
            log.info(
                "[stage 3 / done]   result mesh:   vertices=%d  faces=%d  smoothed=%s",
                len(result_mesh.vertices), len(result_mesh.faces),
                enable_smoothing and path.isfile(out_smooth),
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # self._recompute_smooth_normals(result_mesh)
        return result_mesh
    
    # ------------------------------------------------------------------
    # Private — library loading
    # ------------------------------------------------------------------

    def _load_libs(self):
        _LIB_NAMES = {
            "Windows": ("lib_quadwild.dll", "lib_quadpatches.dll"),
            "Darwin":  ("liblib_quadwild.dylib", "liblib_quadpatches.dylib"),
            "Linux":   ("liblib_quadwild.so", "liblib_quadpatches.so"),
        }
        qw_name, qp_name = _LIB_NAMES.get(
            platform.system(), ("liblib_quadwild.so", "liblib_quadpatches.so"),
        )
        qw_path, qp_path = (str(self._libs_dir / n) for n in (qw_name, qp_name))

        for lib_path in (qw_path, qp_path):
            if not path.isfile(lib_path):
                raise QuadWildError(f"Library not found: {lib_path}")

        qw = cdll.LoadLibrary(qw_path)
        qp = cdll.LoadLibrary(qp_path)

        qw.remeshAndField2.argtypes = [POINTER(_Parameters), c_char_p, c_char_p, c_char_p]
        qw.remeshAndField2.restype  = None
        qw.trace2.argtypes          = [c_char_p]
        qw.trace2.restype           = c_bool
        qp.quadPatches.argtypes     = [c_char_p, POINTER(_QRParameters), c_float, c_int, c_bool]
        qp.quadPatches.restype      = c_int

        return qw, qp

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _to_scene(
        self, 
        mesh: Union[str, Path, "trimesh.Trimesh", "trimesh.Scene"]
    ) -> "trimesh.Scene":
        """Normalise any supported input type to a trimesh.Scene."""
        if isinstance(mesh, Path) or isinstance(mesh, str):
            return trimesh.load(mesh, force='scene', process=False, merge_primitives=False, skip_materials=False, maintain_order=True)
        elif isinstance(mesh, trimesh.Scene):
            return mesh
        elif isinstance(mesh, trimesh.Trimesh):
            return trimesh.Scene(mesh)
        else:
            raise TypeError("Input must be a Trimesh or Scene")

    def _to_mesh(
        self, 
        scene: "trimesh.Scene"
    ) -> "trimesh.Trimesh":
        """Concatenate every Trimesh geometry in the scene into one mesh."""
        if isinstance(scene, trimesh.Trimesh):
            return scene
        elif isinstance(scene, (Path, str)):
            loaded = trimesh.load(scene, force='mesh', process=False)
            if isinstance(loaded, trimesh.Trimesh):
                return loaded
            elif isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) == 0:
                    raise ValueError("No geometry found in the scene.")            
                # Use to_geometry() to correctly bake in transforms from the scene graph
                return loaded.to_geometry() if hasattr(loaded, 'to_geometry') else loaded.dump(concatenate=True)
            else:
                raise TypeError(f"Loaded object is neither a Trimesh nor a Scene: {type(loaded)}")
        elif isinstance(scene, trimesh.Scene):
            if len(scene.geometry) == 0:
                raise ValueError("No geometry found in the scene.")        
            # Use to_geometry() to correctly bake in transforms from the scene graph
            return scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
        elif scene is None:
            raise ValueError("Input cannot be None")
        else:
            raise TypeError(f"Input must be a Trimesh, Scene, or file path; got {type(scene)}")

    def _estimate_scale_factor(self, mesh: "trimesh.Trimesh", target_quad_count: int) -> float:
        """Estimate a ``scale_factor`` that targets *target_quad_count* output quads.

        Derivation: with ``scale_factor=1`` the solver produces roughly
        ``F_in / 2`` quads (each input triangle maps to half a quad).  Quad
        count scales as ``1 / scale_factor²``, so:

            target = (F_in / 2) / scale_factor²
            → scale_factor = sqrt(F_in / (2 × target))

        The result is clamped to ``[0.01, 10.0]``.
        """
        f_in = len(mesh.faces)
        if f_in == 0 or target_quad_count <= 0:
            return 1.0
        sf = math.sqrt(f_in / (2.0 * target_quad_count))
        return max(0.01, min(10.0, sf))

    def _get_boundary(self, mesh: "trimesh.Trimesh") -> set:
        """Return geometric positions of all boundary edges as ``frozenset`` pairs.

        On a mesh loaded from a UV-mapped format (OBJ, glTF, …), trimesh
        splits vertices at UV seams.  Those seam edges appear as boundary
        edges here.  After a subsequent position-based ``merge_vertices``,
        UV seam edges become interior edges while real geometric boundaries
        remain.  Comparing the result of this method (called *before* merge)
        with the post-merge ``face_adjacency`` allows us to identify UV seams.

        Each element is ``frozenset((pos0_tuple, pos1_tuple))`` with positions
        rounded to 6 decimal places.
        """
        if len(mesh.faces) == 0:
            return set()
        inv    = mesh.edges_unique_inverse
        counts = np.bincount(inv, minlength=len(mesh.edges_unique))
        b_mask = counts == 1
        b_edges = mesh.edges_unique[b_mask]
        verts  = np.round(mesh.vertices, decimals=6)
        return {
            frozenset((tuple(verts[e[0]]), tuple(verts[e[1]])))
            for e in b_edges
        }

    def _smooth_normals(self, mesh: "trimesh.Trimesh") -> None:
        """Compute smooth vertex normals in-place using trimesh's angle-weighted average.

        Matches the ``_smooth_normals_trimesh`` approach from mesh_repair:
        uses ``trimesh.geometry.weighted_vertex_normals`` with per-face corner
        angles as weights, which gives smooth shading without any custom
        accumulation logic.
        """
        mesh.merge_vertices(merge_tex=False, merge_norm=False)
        mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

    def _preprocess_mesh(self, mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
        """Minimal mesh cleanup before handing off to the C++ solver.

        Mirrors QRemeshify's approach: no hole-filling.  The C++ stage
        ``remeshAndField2`` (with ``remesh=True``) is responsible for
        handling any remaining topology issues.

        Steps always applied:

        1. Merge duplicate / near-duplicate vertices (removes seam splits
           introduced by exporters; needed for correct sharp-edge detection).
        2. Remove degenerate faces.
        3. Fix inverted normals and face winding.
        """
        # O. Init variables
        threshold = 1e-6
        digits_vertex = max(1, int(-math.log10(threshold)))

        # 1. Merge duplicate / near-duplicate vertices
        mesh = trimesh.util.concatenate([mesh])  # returns a copy
        mesh.merge_vertices(
            merge_tex=False, 
            merge_norm=False,
            digits_vertex=digits_vertex,
            digits_norm=max(1, digits_vertex - 1),
            digits_uv=max(1, digits_vertex - 1),
        )

        # 2. Remove degenerate faces
        # mesh.update_faces(mesh.nondegenerate_faces())
        # mesh.update_faces(mesh.unique_faces())
        # mesh.remove_unreferenced_vertices()

        # Refresh caches so normals/UVs stay consistent
        # mesh._cache.clear()

        # 3. Fix normals / winding
        # self._smooth_normals(mesh)
        # trimesh.repair.fix_normals(mesh)
        # trimesh.repair.fix_winding(mesh)

        if not mesh.is_watertight:
            logging.warning(
                "[QuadWild] Warning: mesh is not watertight — "
                "remeshAndField will attempt to repair it."
            )

        return mesh

    # ------------------------------------------------------------------
    # Private — IO helpers
    # ------------------------------------------------------------------
    def _import_obj(self, obj_path: str) -> "trimesh.Trimesh":
        """Parse the minimal OBJ produced by the C++ libs (``v`` + ``f`` only).

        Face tokens may be in any of the OBJ forms:
        ``vi``, ``vi/vt``, ``vi//vn``, ``vi/vt/vn``.
        """
        if not path.isfile(obj_path):
            raise QuadWildError(f"Expected output file not found: {obj_path}")

        verts: list = []
        faces: list = []

        with open(obj_path) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == "v":
                    verts.append([float(x) for x in tokens[1:4]])
                elif tokens[0] == "f":
                    faces.append([int(t.split("/")[0]) - 1 for t in tokens[1:]])

        return trimesh.Trimesh(
            vertices=np.array(verts, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            process=False,
        )

    def _export_obj(self, mesh: "trimesh.Trimesh", obj_path: str) -> None:
        """Write a minimal OBJ file with per-face normals.

        Format::

            v  x y z          (one vertex per line)
            vn nx ny nz       (one face-normal per line, 1-indexed from faces)
            f  vi//ni vi//ni vi//ni   (vertex index // normal index, 1-based)

        This is the exact format consumed by ``remeshAndField2``.
        """
        verts, faces, normals = mesh.vertices, mesh.faces, mesh.face_normals
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
        entries = self._detect_sharp_edges(
            mesh, sharp_angle,
            pre_merge_boundary_positions=pre_merge_boundary_positions,
        )
        with open(sharp_path, "w") as f:
            f.write(f"{len(entries)}\n")
            f.writelines(f"{c},{fi},{ei}\n" for c, fi, ei in entries)
        return len(entries)

    # ------------------------------------------------------------------
    # Private — sharp-feature export
    # ------------------------------------------------------------------    
    def _detect_sharp_edges(
        self,
        mesh: "trimesh.Trimesh",
        sharp_angle: float,
        *,
        pre_merge_boundary_positions: Optional[set] = None,
    ) -> list:
        """Return ``(convexity, face_index, edge_index_in_face)`` for every sharp edge.

        An edge is sharp when:

        * Its dihedral angle between adjacent faces exceeds *sharp_angle*, **or**
        * It is a boundary edge (belongs to only one face), **or**
        * It is a UV seam edge (identified via *pre_merge_boundary_positions*).

        UV seam detection
        ~~~~~~~~~~~~~~~~~
        When a mesh is loaded from a format that stores per-face-vertex UVs
        (OBJ, glTF, …), trimesh duplicates vertices at UV seams.  Before the
        position-based vertex merge in ``_preprocess_mesh``, those seam edges
        appear as boundary edges.  After merge they become regular interior
        edges.  ``pre_merge_boundary_positions`` (computed by
        ``_get_boundary_edge_positions`` on the raw mesh) lets us identify
        them: any ``face_adjacency`` edge whose vertex positions were in that
        pre-merge boundary set is a UV seam.
        """
        threshold_rad = math.radians(sharp_angle)
        entries: list = []
        added: set = set()   # (face_index, edge_local) already emitted

        # ── Interior edges above the angle threshold ─────────────────────────
        adj        = mesh.face_adjacency           # (N, 2)
        adj_edges  = mesh.face_adjacency_edges     # (N, 2)
        adj_angles = mesh.face_adjacency_angles    # (N,)

        sharp_mask = adj_angles > threshold_rad
        sharp_idxs = np.where(sharp_mask)[0]

        for k in sharp_idxs:
            fi0       = int(adj[k, 0])
            fi1       = int(adj[k, 1])
            ev        = (int(adj_edges[k, 0]), int(adj_edges[k, 1]))
            ei_local  = self._edge_index_in_face(mesh.faces[fi0], ev)
            if ei_local < 0:
                continue
            convexity = self._edge_convexity(mesh, fi0, fi1)
            entries.append((convexity, fi0, ei_local))
            added.add((fi0, ei_local))

        # ── UV seam edges ─────────────────────────────────────────────────────
        # Interior edges whose geometric positions were boundary edges on the
        # raw (pre-merge) mesh — i.e. they were split at UV discontinuities.
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
                ei_local = self._edge_index_in_face(mesh.faces[fi0], ev)
                if ei_local < 0 or (fi0, ei_local) in added:
                    continue
                convexity = self._edge_convexity(mesh, fi0, fi1)
                entries.append((convexity, fi0, ei_local))
                added.add((fi0, ei_local))
                n_uv += 1
            if n_uv:
                log.info("[sharp]  UV seam edges added: %d", n_uv)

        # ── Boundary edges ────────────────────────────────────────────────────
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

        return entries

    def _edge_index_in_face(self, face: np.ndarray, edge_verts: tuple) -> int:
        """Return the local 0-based index (0–2) of *edge_verts* within *face*.

        Trimesh ordering: 0 → (v0, v1), 1 → (v1, v2), 2 → (v0, v2).
        Returns -1 if not found.
        """
        ev = frozenset(edge_verts)
        return next(
            (i for i, (a, b) in enumerate(
                ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])),
            ) if frozenset((a, b)) == ev),
            -1,
        )

    def _edge_convexity(
        self,
        mesh: "trimesh.Trimesh",
        fi0: int,
        fi1: int,
    ) -> int:
        """Return 1 (convex / ridge) or 0 (concave / valley).

        Convex means the centroid of *fi1* lies on the negative side of *fi0*'s
        oriented plane — i.e. the surface folds outward (ridge).
        """
        n0  = mesh.face_normals[fi0]
        c0  = mesh.triangles_center[fi0]
        c1  = mesh.triangles_center[fi1]
        return 1 if float(np.dot(n0, c1 - c0)) < 0.0 else 0
    
    # ------------------------------------------------------------------
    # Private — C++ library calls
    # ------------------------------------------------------------------
    # Private — C++ output suppression
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _suppress_c_output(self):
        """Redirect C-level stdout and stderr to /dev/null for the duration.

        Python's sys.stdout redirect is insufficient because the C++ shared
        libraries write directly to file descriptors 1 and 2.  We dup the
        originals, point the fds at /dev/null, then restore on exit.
        """
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

    # ------------------------------------------------------------------
    def _call_remesh_and_field(
        self,
        obj_path: str,
        sharp_path: str,
        field_path: str,
        *,
        enable_preprocess: bool,
        enable_sharp: bool,
        sharp_angle: float,
    ) -> None:
        """Stage 1 — decimation / repair + cross-field computation."""
        params = _Parameters(
            remesh=enable_preprocess,
            sharpAngle=sharp_angle if enable_sharp else -1.0,
            alpha=0.01,    # unused
            scaleFact=1.0, # unused
            hasFeature=enable_sharp,
            hasField=False,
        )
        try:
            with (contextlib.nullcontext() if log.isEnabledFor(logging.DEBUG) else self._suppress_c_output()):
                self._qw.remeshAndField2(
                    byref(params),
                    obj_path.encode(),
                    sharp_path.encode(),
                    field_path.encode(),
                )
        except Exception as exc:
            raise QuadWildError("remeshAndField2 failed") from exc

    def _call_trace(self, remeshed_base: str) -> None:
        """Stage 2 — field tracing and patch decomposition.

        *remeshed_base* is the path **without** extension; the C++ lib appends
        ``.obj`` and writes the traced patch file as ``<base>_p0.obj``.
        """
        try:
            with (contextlib.nullcontext() if log.isEnabledFor(logging.DEBUG) else self._suppress_c_output()):
                ok = self._qw.trace2(remeshed_base.encode())
        except Exception as exc:
            raise QuadWildError("trace2 failed") from exc
        if not ok:
            raise QuadWildError(
                "trace2 returned False — field tracing failed. "
                "This often means the mesh has degenerate geometry or too few faces."
            )

    def _call_quadrangulate(
        self,
        traced_path: str,
        *,
        enable_smoothing: bool,
        scale_factor: float,
        fixed_chart_clusters: int,
        alpha: float,
        ilp_method: str,
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
        flow_config: str,
        satsuma_config: str,
        callback_time_limit: List[float],
        callback_gap_limit: List[float],
    ) -> int:
        """Stage 3 — ILP quadrangulation + optional smoothing.

        The callback arrays must stay alive for the duration of the call; they
        are held as local variables (not mere temporaries) to prevent the GC
        from collecting them before the C++ lib returns.
        """
        cb_time_arr = (c_float * len(callback_time_limit))(*callback_time_limit)
        cb_gap_arr  = (c_float * len(callback_gap_limit))(*callback_gap_limit)

        p = _QRParameters()

        # ── Fixed internal settings ───────────────────────────────────────────
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

        # ── Config-file paths (must be absolute byte-strings) ─────────────────
        p.flow_config_filename    = str(self._config_dir / _FLOW_CONFIGS[flow_config]).encode()
        p.satsuma_config_filename = str(self._config_dir / _SATSUMA_CONFIGS[satsuma_config]).encode()

        # ── User-tunable ILP parameters ───────────────────────────────────────
        p.alpha                             = alpha
        p.ilpMethod                         = _ILP_METHODS[ilp_method]
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
            with (contextlib.nullcontext() if log.isEnabledFor(logging.DEBUG) else self._suppress_c_output()):
                result = self._qp.quadPatches(
                    traced_path.encode(),
                    byref(p),
                    c_float(scale_factor),
                    c_int(fixed_chart_clusters),
                    c_bool(enable_smoothing),
                )
        except Exception as exc:
            raise QuadWildError("quadPatches failed") from exc

        return result
