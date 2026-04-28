import argparse
from pathlib import Path

import matplotlib

# Headless backend for cluster/non-GUI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _collect_grasp_data(pred_grasps_cam, scores):
    centers = []
    confs = []
    dirs = []
    for key in pred_grasps_cam:
        grasps = pred_grasps_cam[key]
        if len(grasps) == 0:
            continue
        centers.append(grasps[:, :3, 3])
        confs.append(scores[key])
        # Use local +X axis as a grasp direction cue in camera frame.
        dirs.append(grasps[:, :3, 0])

    if not centers:
        return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 3))
    return (
        np.concatenate(centers, axis=0),
        np.concatenate(confs, axis=0),
        np.concatenate(dirs, axis=0),
    )


def _select_topk(centers, confs, dirs, top_k):
    if top_k <= 0 or len(confs) <= top_k:
        return centers, confs, dirs
    idx = np.argsort(confs)[-top_k:]
    idx = idx[np.argsort(confs[idx])[::-1]]
    return centers[idx], confs[idx], dirs[idx]


def _save_view(
    pc,
    centers,
    confs,
    dirs,
    x_idx,
    y_idx,
    u_idx,
    v_idx,
    title,
    out_path,
    point_size=0.2,
    grasp_size=10.0,
    arrow_scale=0.04,
    dpi=300,
):
    plt.figure(figsize=(8, 7))
    plt.scatter(pc[:, x_idx], pc[:, y_idx], s=point_size, c="lightgray", alpha=0.25)

    if len(centers):
        sc = plt.scatter(
            centers[:, x_idx],
            centers[:, y_idx],
            s=grasp_size,
            c=confs,
            cmap="viridis",
            edgecolors="none",
        )
        plt.colorbar(sc, label="grasp score")
        # Direction arrows make grasp endpoint/orientation easier to read.
        plt.quiver(
            centers[:, x_idx],
            centers[:, y_idx],
            dirs[:, u_idx] * arrow_scale,
            dirs[:, v_idx] * arrow_scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0018,
            alpha=0.65,
            color="tab:red",
        )
        best_idx = int(np.argmax(confs))
        plt.scatter(
            centers[best_idx, x_idx],
            centers[best_idx, y_idx],
            s=64,
            marker="*",
            c="red",
            label="best score",
        )
        plt.legend(loc="upper right")

    plt.axis("equal")
    plt.title(title)
    axis_names = ["X", "Y", "Z"]
    plt.xlabel(f"{axis_names[x_idx]} (m)")
    plt.ylabel(f"{axis_names[y_idx]} (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _save_interactive_3d_html(pc, centers, confs, out_html, max_pc_points=50000):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    if len(pc) > max_pc_points:
        idx = np.random.choice(len(pc), max_pc_points, replace=False)
        pc_plot = pc[idx]
    else:
        pc_plot = pc

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pc_plot[:, 0],
            y=pc_plot[:, 1],
            z=pc_plot[:, 2],
            mode="markers",
            marker=dict(size=1, color="lightgray", opacity=0.25),
            name="point cloud",
        )
    )

    if len(centers):
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker=dict(size=3, color=confs, colorscale="Viridis", opacity=0.9),
                name="grasp centers",
            )
        )
        best_idx = int(np.argmax(confs))
        fig.add_trace(
            go.Scatter3d(
                x=[centers[best_idx, 0]],
                y=[centers[best_idx, 1]],
                z=[centers[best_idx, 2]],
                mode="markers",
                marker=dict(size=7, color="red", symbol="diamond"),
                name="best grasp",
            )
        )

    fig.update_layout(
        title="3D grasp overview (offline interactive)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def export_projection_images(npz_path, output_dir, top_k=300, save_html_3d=True):
    data = np.load(npz_path, allow_pickle=True)
    pc = data["pc_full"]
    pred_grasps_cam = data["pred_grasps_cam"].item()
    scores = data["scores"].item()

    centers, confs, dirs = _collect_grasp_data(pred_grasps_cam, scores)
    centers, confs, dirs = _select_topk(centers, confs, dirs, top_k=top_k)

    stem = Path(npz_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_view(
        pc,
        centers,
        confs,
        dirs,
        x_idx=0,
        y_idx=1,
        u_idx=0,
        v_idx=1,
        title="Top view (XY)",
        out_path=output_dir / f"{stem}_top_xy.png",
    )
    _save_view(
        pc,
        centers,
        confs,
        dirs,
        x_idx=0,
        y_idx=2,
        u_idx=0,
        v_idx=2,
        title="Side view (XZ)",
        out_path=output_dir / f"{stem}_side_xz.png",
    )
    _save_view(
        pc,
        centers,
        confs,
        dirs,
        x_idx=1,
        y_idx=2,
        u_idx=1,
        v_idx=2,
        title="Front view (YZ)",
        out_path=output_dir / f"{stem}_front_yz.png",
    )

    html_saved = False
    if save_html_3d:
        html_saved = _save_interactive_3d_html(
            pc,
            centers,
            confs,
            out_html=output_dir / f"{stem}_3d.html",
        )
    return html_saved


def main():
    parser = argparse.ArgumentParser(
        description="Export offline grasp projection figures from prediction npz files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Input glob or file path, e.g. "results/predictions_*.npz"',
    )
    parser.add_argument(
        "--output_dir",
        default="results/figures",
        help="Directory to save rendered png images.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=300,
        help="Only visualize top-K grasps by score. <=0 means all grasps.",
    )
    parser.add_argument(
        "--no_html_3d",
        action="store_true",
        help="Disable saving offline interactive 3D html.",
    )
    args = parser.parse_args()

    input_paths = sorted(Path(".").glob(args.input))
    if not input_paths:
        # If user passes an absolute file path.
        p = Path(args.input)
        if p.exists():
            input_paths = [p]

    if not input_paths:
        raise FileNotFoundError(f"No input files matched: {args.input}")

    output_dir = Path(args.output_dir)
    any_html_saved = False
    for npz_path in input_paths:
        html_saved = export_projection_images(
            npz_path,
            output_dir,
            top_k=args.top_k,
            save_html_3d=not args.no_html_3d,
        )
        any_html_saved = any_html_saved or html_saved
        print(f"Saved projection figures for: {npz_path}")

    print(f"All figures saved to: {output_dir.resolve()}")
    if not args.no_html_3d and not any_html_saved:
        print("Plotly not found, skipped 3D html export.")


if __name__ == "__main__":
    main()
