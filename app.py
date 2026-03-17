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

from flask import Flask, request, send_file, send_from_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

import trimesh

from src.quadwild import QuadWild, QuadWildError

app = Flask(__name__, static_folder="static", static_url_path="")

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


@app.route("/remesh", methods=["POST"])
def remesh():
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
        geom_count = sum(
            1 for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)
        )

        tqc = int(s.get("target_quad_count", 0))

        result = _qw.remesh(
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
        )

        buf = io.BytesIO()
        result.export(buf, file_type="glb")
        buf.seek(0)
        return send_file(buf, mimetype="model/gltf-binary", download_name="result.glb")

    except QuadWildError as exc:
        return f"QuadWild error: {exc}", 500
    except Exception as exc:
        return f"Unexpected error: {exc}", 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
