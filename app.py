"""app.py — Gradio demo for the QuadWild quad-remesher.

Run with:
    python app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import trimesh

from src.quadwild import QuadWild, QuadWildError

# Instantiated once — libraries are loaded at startup, not per-click.
_qw = QuadWild()


def run(
    input_path: str | None,
    # ── Basic settings ─────────────────────────────────────
    enable_preprocess: bool,
    enable_sharp: bool,
    sharp_angle: float,
    enable_smoothing: bool,
    scale_factor: float,
    target_quad_count: int,
    # ── Advanced — ILP objective ────────────────────────────
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
    fixed_chart_clusters: int,
    # ── Advanced — solver presets ───────────────────────────
    flow_config: str,
    satsuma_config: str,
) -> str | None:
    if input_path is None:
        return None
    try:
        scene = trimesh.load(input_path, force="scene")
        result = _qw.process(
            scene,
            enable_preprocess=enable_preprocess,
            enable_sharp=enable_sharp,
            sharp_angle=sharp_angle,
            enable_smoothing=enable_smoothing,
            scale_factor=scale_factor,
            target_quad_count=int(target_quad_count) if target_quad_count and target_quad_count > 0 else None,
            alpha=alpha,
            ilp_method=ilp_method,
            time_limit=int(time_limit),
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
            fixed_chart_clusters=int(fixed_chart_clusters),
            flow_config=flow_config,
            satsuma_config=satsuma_config,
        )
        stem = Path(input_path).stem
        tmp_dir = tempfile.mkdtemp()
        out_path = str(Path(tmp_dir) / f"{stem}_remeshed.glb")
        result.export(out_path)
        return out_path
    except QuadWildError as exc:
        print(f"[QuadWild] Pipeline error: {exc}")
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[QuadWild] Unexpected error: {exc}")
        raise gr.Error(f"Unexpected error: {exc}") from exc


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="QuadWild — Quad Remesher") as demo:
        gr.Markdown(
            "# QuadWild — Quad Remesher\n"
            "Upload a 3-D mesh, tune the parameters, and press **Remesh** to "
            "generate a clean quad topology."
        )

        # ── Viewers ──────────────────────────────────────────────────────────
        with gr.Row():
            input_model = gr.Model3D(
                label="Input",
                display_mode="solid",
                height=420,
                scale=1,
            )
            output_model = gr.Model3D(
                label="Output (quad mesh)",
                display_mode="solid",
                height=420,
                scale=1,
            )

        remesh_btn = gr.Button("Remesh", variant="primary", size="lg")

        # ── Basic settings ────────────────────────────────────────────────────
        with gr.Accordion("Basic Settings", open=True):
            with gr.Row():
                enable_preprocess = gr.Checkbox(
                    value=True,
                    label="Preprocess",
                    info="Decimate, triangulate and repair the mesh before field computation.",
                )
                enable_smoothing = gr.Checkbox(
                    value=True,
                    label="Smoothing",
                    info="Apply Laplacian smoothing after quadrangulation.",
                )

            with gr.Row():
                enable_sharp = gr.Checkbox(
                    value=True,
                    label="Detect Sharp Features",
                    info="Mark edges above the angle threshold as feature lines.",
                    scale=1,
                )
                sharp_angle = gr.Slider(
                    minimum=1.0, maximum=179.0, step=0.5, value=45.0,
                    label="Sharp Angle (°)",
                    info="Dihedral angle threshold for sharp-edge detection.",
                    scale=3,
                )

            with gr.Row():
                target_quad_count = gr.Number(
                    value=0, minimum=0, precision=0,
                    label="Target Quad Count",
                    info="Approximate number of output quads. Overrides Scale Factor when > 0.",
                    scale=1,
                )
                scale_factor = gr.Slider(
                    minimum=0.1, maximum=5.0, step=0.05, value=1.0,
                    label="Scale Factor",
                    info="> 1 → larger quads (coarser);  < 1 → smaller quads (more detail). Ignored when Target Quad Count > 0.",
                    scale=2,
                )

        # ── Advanced settings ─────────────────────────────────────────────────
        with gr.Accordion("Advanced Settings", open=False):

            gr.Markdown("### ILP Objective")
            with gr.Row():
                alpha = gr.Slider(
                    minimum=0.0, maximum=0.999, step=0.001, value=0.005,
                    label="Alpha",
                    info="Blend: α → isometry,  1–α → regularity.",
                )
                ilp_method = gr.Dropdown(
                    choices=["LEASTSQUARES", "ABS"],
                    value="LEASTSQUARES",
                    label="ILP Method",
                )

            with gr.Row():
                time_limit = gr.Number(
                    value=200, minimum=1, precision=0,
                    label="Time Limit (s)",
                    info="Hard wall-clock limit for the ILP solver.",
                )
                gap_limit = gr.Number(
                    value=0.0, minimum=0.0, precision=4,
                    label="Gap Limit",
                    info="Stop early when optimality gap drops to this value (0 = disabled).",
                )
                minimum_gap = gr.Number(
                    value=0.4, minimum=0.0, precision=4,
                    label="Minimum Gap",
                    info="The solver must achieve at least this gap.",
                )

            gr.Markdown("### Regularity & Alignment")
            with gr.Row():
                isometry = gr.Checkbox(value=True, label="Isometry")
                regularity_quads = gr.Checkbox(value=True, label="Regularity Quads")
                regularity_non_quads = gr.Checkbox(value=True, label="Regularity Non-Quads")
                hard_parity_constraint = gr.Checkbox(value=True, label="Hard Parity Constraint")

            with gr.Row():
                regularity_non_quads_weight = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.9,
                    label="Regularity Non-Quads Weight",
                )
                align_singularities = gr.Checkbox(value=True, label="Align Singularities")
                align_singularities_weight = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.1,
                    label="Singularity Alignment Weight",
                )

            gr.Markdown("### Constraint Recovery")
            with gr.Row():
                repeat_losing_iterations = gr.Checkbox(value=True,  label="Repeat Iterations")
                repeat_losing_quads      = gr.Checkbox(value=False, label="Repeat Quads")
                repeat_losing_non_quads  = gr.Checkbox(value=False, label="Repeat Non-Quads")
                repeat_losing_align      = gr.Checkbox(value=True,  label="Repeat Align")

            gr.Markdown("### Solver Presets")
            with gr.Row():
                flow_config = gr.Dropdown(
                    choices=["SIMPLE", "HALF"],
                    value="SIMPLE",
                    label="Flow Config",
                    info="Flow-solver configuration preset.",
                )
                satsuma_config = gr.Dropdown(
                    choices=["LEMON", "DEFAULT", "MST", "ROUND2EVEN", "SYMMDC", "EDGETHRU", "NODETHRU"],
                    value="LEMON",
                    label="Satsuma Config",
                    info="Satsuma matching preset.",
                )

            fixed_chart_clusters = gr.Number(
                value=0, minimum=0, precision=0,
                label="Fixed Chart Clusters",
                info="Force a fixed number of chart clusters (0 = auto).",
            )

        # ── Wiring ────────────────────────────────────────────────────────────
        all_inputs = [
            input_model,
            enable_preprocess, enable_sharp, sharp_angle, enable_smoothing, scale_factor, target_quad_count,
            alpha, ilp_method, time_limit, gap_limit, minimum_gap,
            isometry, regularity_quads, regularity_non_quads, regularity_non_quads_weight,
            align_singularities, align_singularities_weight,
            repeat_losing_iterations, repeat_losing_quads, repeat_losing_non_quads, repeat_losing_align,
            hard_parity_constraint, fixed_chart_clusters,
            flow_config, satsuma_config,
        ]

        remesh_btn.click(fn=run, inputs=all_inputs, outputs=output_model)

    return demo


if __name__ == "__main__":
    build_demo().launch()
