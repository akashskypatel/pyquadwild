"""app.py — Flask backend for the QuadWild HTML UI.

Run with:
    python app.py
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from pathlib import Path

import base64

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

import trimesh

from pyquadwild import QuadWild, QuadWildError

app = Flask(__name__, static_folder="static")

_qw = QuadWild()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/convert_to_glb", methods=["POST"])
def convert_to_glb():
    """Accept any mesh format, load with trimesh, and return GLB bytes."""
    f = request.files.get("file")
    if f is None:
        return "No file uploaded", 400

    suffix = Path(f.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp)
        tmp_path = tmp.name

    try:
        scene = trimesh.load(tmp_path, force="scene")
        buf = io.BytesIO()
        scene.export(buf, file_type="glb")
        buf.seek(0)
        return send_file(buf, mimetype="model/gltf-binary", download_name="preview.glb")
    except Exception as exc:
        return f"Conversion error: {exc}", 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.route("/remesh_quads", methods=["POST"])
def remesh_quads():
    """Run the QuadWild pipeline and return the result as GLB."""
    f = request.files.get("file")
    if f is None:
        return "No file uploaded", 400

    raw_settings = request.form.get("settings", "{}")
    try:
        s = json.loads(raw_settings)
    except json.JSONDecodeError:
        return "Invalid settings JSON", 400

    suffix = Path(f.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp)
        tmp_path = tmp.name

    try:
        scene = trimesh.load(tmp_path, force="scene")
        tqc = int(s.get("target_quad_count", 0))

        result_scene = _qw.remesh(
            scene,
            enable_preprocess=s.get("enable_preprocess", True),
            enable_sharp=s.get("enable_sharp", True),
            sharp_angle=float(s.get("sharp_angle", 45)),
            enable_smoothing=s.get("enable_smoothing", True),
            scale_factor=float(s.get("scale_factor", 1.0)),
            target_quad_count=tqc if tqc > 0 else None,
            alpha=float(s.get("alpha", 0.005)),
            ilp_method=s.get("ilp_method", "LEASTSQUARES"),
            time_limit=int(s.get("time_limit", 200)),
            gap_limit=float(s.get("gap_limit", 0.0)),
            minimum_gap=float(s.get("minimum_gap", 0.4)),
            isometry=s.get("isometry", True),
            regularity_quads=s.get("regularity_quads", True),
            regularity_non_quads=s.get("regularity_non_quads", True),
            regularity_non_quads_weight=float(
                s.get("regularity_non_quads_weight", 0.9)
            ),
            align_singularities=s.get("align_singularities", True),
            align_singularities_weight=float(
                s.get("align_singularities_weight", 0.1)
            ),
            repeat_losing_iterations=s.get("repeat_losing_iterations", True),
            repeat_losing_quads=s.get("repeat_losing_quads", False),
            repeat_losing_non_quads=s.get("repeat_losing_non_quads", False),
            repeat_losing_align=s.get("repeat_losing_align", True),
            hard_parity_constraint=s.get("hard_parity_constraint", True),
            fixed_chart_clusters=int(s.get("fixed_chart_clusters", 0)),
            merge_geometries=s.get("merge_geometries", False),
            flow_config=s.get("flow_config", "SIMPLE"),
            satsuma_config=s.get("satsuma_config", "LEMON"),
            output_format="scene",
        )

        # --- Collect all quad meshes from the returned scene (preserving names) ---
        result_geoms = {
            k: v for k, v in result_scene.geometry.items()
            if isinstance(v, trimesh.Trimesh)
        }

        # --- Build quad OBJ (preserves quad topology, all geometries) ---
        obj_lines = ["# QuadWild quad-remesh result"]
        vert_offset = 0
        for geom in result_geoms.values():
            for v in geom.vertices:
                obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            for face in geom.faces:
                face_list = [int(idx) + 1 + vert_offset for idx in face]
                obj_lines.append("f " + " ".join(str(i) for i in face_list))
            vert_offset += len(geom.vertices)
        obj_text = "\n".join(obj_lines) + "\n"

        # --- Triangulate each geometry separately; export as multi-part GLB ---
        glb_scene = trimesh.Scene()
        for name, geom in result_geoms.items():
            tri_faces = []
            for face in geom.faces:
                face_list = list(face)
                for i in range(1, len(face_list) - 1):
                    tri_faces.append([face_list[0], face_list[i], face_list[i + 1]])
            tri_geom = trimesh.Trimesh(
                vertices=geom.vertices,
                faces=np.array(tri_faces, dtype=np.int64) if tri_faces else np.empty((0, 3), dtype=np.int64),
                process=False,
            )
            glb_scene.add_geometry(tri_geom, geom_name=name)
        buf_glb = io.BytesIO()
        glb_scene.export(buf_glb, file_type="glb")

        glb_b64 = base64.b64encode(buf_glb.getvalue()).decode("ascii")
        obj_b64 = base64.b64encode(obj_text.encode("utf-8")).decode("ascii")
        return jsonify({"glb": glb_b64, "obj": obj_b64})

    except QuadWildError as exc:
        return f"QuadWild error: {exc}", 500
    except Exception as exc:
        return f"Unexpected error: {exc}", 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.route("/remesh_triangles", methods=["POST"])
def remesh_triangles():
    """Apply high-quality triangle remeshing using PyMeshLab and return GLB + OBJ."""
    try:
        import pymeshlab
    except ImportError:
        return "pymeshlab is not installed", 503

    f = request.files.get("file")
    if f is None:
        return "No file uploaded", 400

    raw_settings = request.form.get("settings", "{}")
    try:
        s = json.loads(raw_settings)
    except json.JSONDecodeError:
        s = {}

    target_len = float(s.get("tri_target_len", 1.0))
    iterations = int(s.get("tri_iterations", 3))
    featuredeg = float(s.get("tri_feature_angle", 30.0))
    adaptive = bool(s.get("tri_adaptive", False))
    checksurfdist = bool(s.get("tri_check_surf_dist", True))
    maxsurfdist_pct = float(s.get("tri_max_surf_dist", 1.0))

    suffix = Path(f.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp)
        tmp_path = tmp.name

    try:
        # Load mesh as a Scene
        loaded = trimesh.load(tmp_path, force="scene", process=False)
        remeshed_geometries = {}
        
        for geom_name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                remeshed_geometries[geom_name] = geom
                continue

            # Export current geometry to a temporary OBJ for PyMeshLab
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp_geom_in:
                geom_in_path = tmp_geom_in.name
            geom.export(geom_in_path)

            try:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(geom_in_path)

                # Pre-cleanup: remove degenerate geometry
                ms.apply_filter("meshing_remove_duplicate_vertices")
                ms.apply_filter("meshing_remove_duplicate_faces")
                ms.apply_filter("meshing_repair_non_manifold_edges")
                ms.apply_filter("meshing_repair_non_manifold_vertices")

                # Apply isotropic explicit remeshing
                ms.apply_filter("meshing_isotropic_explicit_remeshing",
                    targetlen=pymeshlab.PercentageValue(target_len),
                    iterations=iterations,
                    adaptive=adaptive,
                    featuredeg=featuredeg,
                    checksurfdist=checksurfdist,
                    maxsurfdist=pymeshlab.PercentageValue(maxsurfdist_pct),
                )
                
                with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp_geom_out:
                    geom_out_path = tmp_geom_out.name
                
                ms.save_current_mesh(geom_out_path)
                
                # Load the remeshed geometry back using trimesh
                remeshed_geom = trimesh.load(geom_out_path, process=False)
                if isinstance(remeshed_geom, trimesh.Scene):
                    remeshed_geom = remeshed_geom.dump(concatenate=True)
                
                remeshed_geometries[geom_name] = remeshed_geom
            finally:
                Path(geom_in_path).unlink(missing_ok=True)
                if 'geom_out_path' in locals():
                    Path(geom_out_path).unlink(missing_ok=True)
        
        # Reconstruct the scene preserving the original graph (transformations)
        out_scene = trimesh.Scene()
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            if geom_name in remeshed_geometries:
                out_scene.add_geometry(
                    remeshed_geometries[geom_name], 
                    geom_name=geom_name, 
                    node_name=node_name, 
                    transform=transform
                )
            
        # Build GLB preview
        buf_glb = io.BytesIO()
        out_scene.export(buf_glb, file_type="glb")

        # Build OBJ text natively from the scene
        obj_export = out_scene.export(file_type="obj")
        if isinstance(obj_export, bytes):
            obj_text = obj_export.decode("utf-8")
        else:
            obj_text = obj_export

        glb_b64 = base64.b64encode(buf_glb.getvalue()).decode("ascii")
        obj_b64 = base64.b64encode(obj_text.encode("utf-8")).decode("ascii")
        
        return jsonify({"glb": glb_b64, "obj": obj_b64})

    except Exception as exc:
        logging.exception("PyMeshLab remesh error")
        return f"PyMeshLab error: {exc}", 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
