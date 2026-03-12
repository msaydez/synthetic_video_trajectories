import numpy as np
import matplotlib.pyplot as plt

from helper import (
    load_gt_tracks_all_splits_by_viewpoints,
    load_gen_tracks_all_splits_select_viewpoints,
    normalize_segments_running_mean_direction,
    wasserstein_per_component, debug_extreme_tracks,
    filter_segments_by_motion, normalize_segments_by_trajectory_direction
)


# ============================================================
# Flatten normalized components
# ============================================================

def flatten_component(segments_norm, col_idx):
    """
    Flatten one normalized component across:
    segments_norm[split][viewpoint][sequence_name][track_id] -> ndarray(N,6)

    col_idx:
      2 -> cx_tilde
      3 -> cy_tilde
      4 -> w_tilde
      5 -> h_tilde
    """
    vals = []

    for split_data in segments_norm.values():
        for view_data in split_data.values():
            for seq_data in view_data.values():
                for _, arr in seq_data.items():
                    if arr is None or len(arr) == 0:
                        continue
                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= col_idx + 1:
                        vals.append(arr[:, col_idx])

    if len(vals) == 0:
        return np.array([], dtype=float)

    return np.concatenate(vals)


def match_sample_size(a, b, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))

    if len(a) > n:
        a = a[rng.choice(len(a), size=n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), size=n, replace=False)]

    return a, b


# ============================================================
# Plot helpers for distributions
# ============================================================

def plot_hist_overlay(a, b, name, bins=100, label_a="GT", label_b="GEN"):
    plt.figure(figsize=(7, 4))
    plt.hist(a, bins=bins, density=True, alpha=0.5, label=label_a)
    plt.hist(b, bins=bins, density=True, alpha=0.5, label=label_b)
    plt.xlabel(name)
    plt.ylabel("density")
    plt.title(f"Histogram: {name}")
    plt.legend()
    plt.tight_layout()


def plot_kde_overlay(a, b, name, label_a="GT", label_b="GEN", num=400):
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print(f"[WARN] scipy not available, skipping KDE for {name}")
        return

    if len(a) < 2 or len(b) < 2:
        print(f"[WARN] not enough samples for KDE: {name}")
        return

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if np.isclose(lo, hi):
        lo -= 1e-3
        hi += 1e-3

    xs = np.linspace(lo, hi, num)
    kde_a = gaussian_kde(a)
    kde_b = gaussian_kde(b)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, kde_a(xs), label=label_a)
    plt.plot(xs, kde_b(xs), label=label_b)
    plt.xlabel(name)
    plt.ylabel("density")
    plt.title(f"KDE: {name}")
    plt.legend()
    plt.tight_layout()


def plot_2d_density(x, y, title, bins=120):
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins, density=True, cmap='cividis')
    plt.colorbar(label="density")
    plt.xlabel("cx_tilde")
    plt.ylabel("cy_tilde")
    plt.title(title)
    plt.tight_layout()


def plot_2d_density_comparison(ax, x, y, title, xlim=None, ylim=None, bins=120):
    h = ax.hist2d(
        x, y,
        bins=bins,
        density=True,
        range=None if (xlim is None or ylim is None) else [xlim, ylim],
        cmap='cividis'
    )
    ax.set_aspect(0.5)
    ax.set_xlabel("cx_tilde")
    ax.set_ylabel("cy_tilde")
    ax.set_title(title)

    return h


# ============================================================
# Helpers to get one example sequence / track
# ============================================================

def sort_track(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > 1:
        arr = arr[np.argsort(arr[:, 0])]
    return arr


def get_centers(arr):
    x = arr[:, 2]
    y = arr[:, 3]
    w = arr[:, 4]
    h = arr[:, 5]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    return cx, cy, w, h


def find_example_sequence(segments):
    """
    Returns first non-empty (split, viewpoint, sequence_name, seq_data).
    """
    for split, split_data in segments.items():
        for viewpoint, view_data in split_data.items():
            for sequence_name, seq_data in view_data.items():
                if seq_data and len(seq_data) > 0:
                    return split, viewpoint, sequence_name, seq_data
    raise ValueError("No non-empty sequence found.")


def find_example_track(seq_data, min_len=5):
    """
    Returns (track_key, arr) for the longest track or first one above min_len.
    """
    best_key = None
    best_arr = None
    best_len = -1

    for track_key, arr in seq_data.items():
        if arr is None or len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 6:
            continue
        if len(arr) > best_len:
            best_len = len(arr)
            best_key = track_key
            best_arr = arr

    if best_arr is None or best_len < min_len:
        raise ValueError("No suitable example track found.")
    return best_key, sort_track(best_arr)


import numpy as np


def compute_sequence_mean_direction(seq_data, eps=1e-9, min_disp=0.5):
    """
    Backward-compatible wrapper.

    If seq_data is:
      - a dict of tracks: returns the average sequence direction
      - a single track array: returns that track's own direction

    This preserves compatibility with older code while allowing
    per-track normalization.
    """

    # Case 1: single track array passed by mistake / for compatibility
    if not hasattr(seq_data, "items"):
        return compute_track_direction(seq_data, eps=eps, min_disp=min_disp)

    motions = []

    for _, arr in seq_data.items():
        if arr is None or len(arr) == 0:
            continue

        arr = sort_track(arr)
        cx, cy, _, _ = get_centers(arr)

        if len(cx) < 2:
            continue

        dx = cx[-1] - cx[0]
        dy = cy[-1] - cy[0]
        disp = np.hypot(dx, dy)

        if disp < max(eps, min_disp):
            continue

        motions.append([dx / disp, dy / disp])

    if len(motions) == 0:
        return np.array([1.0, 0.0], dtype=float)

    motions = np.asarray(motions, dtype=float)
    mean_dir = motions.mean(axis=0)
    norm = np.linalg.norm(mean_dir)

    if norm < eps:
        return np.array([1.0, 0.0], dtype=float)

    return mean_dir / norm


def compute_track_direction(arr, eps=1e-9, min_disp=0.5):
    """
    Compute the normalized start-to-end direction of a single track.

    Returns a 2D unit vector [dx, dy] / ||d||.
    Falls back to [1, 0] if the track is too short or displacement is too small.
    """

    if arr is None or len(arr) == 0:
        return np.array([1.0, 0.0], dtype=float)

    arr = sort_track(arr)
    cx, cy, _, _ = get_centers(arr)

    if len(cx) < 2:
        return np.array([1.0, 0.0], dtype=float)

    dx = cx[-1] - cx[0]
    dy = cy[-1] - cy[0]
    disp = np.hypot(dx, dy)

    if disp < max(eps, min_disp):
        return np.array([1.0, 0.0], dtype=float)

    return np.array([dx / disp, dy / disp], dtype=float)


def rotation_from_mean_direction(mean_dir):
    theta = np.arctan2(mean_dir[1], mean_dir[0])
    c = np.cos(-theta)
    s = np.sin(-theta)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)


def process_example_track(arr, seq_data=None):
    """
    Produces raw, relative, rotated, processed trajectory for one track.

    Updated behavior:
    - normalization uses the track's own direction
    - seq_data is kept as an optional argument for compatibility
    """

    arr = sort_track(arr)
    cx, cy, w, h = get_centers(arr)

    cx1, cy1 = cx[0], cy[0]

    rel_x = cx - cx1
    rel_y = cy - cy1

    # Per-track direction
    track_dir = compute_track_direction(arr)
    R = rotation_from_mean_direction(track_dir)

    dxy = np.column_stack([rel_x, rel_y])
    dxy_rot = dxy @ R.T

    rot_x = dxy_rot[:, 0]
    rot_y = dxy_rot[:, 1]

    den_x = max(abs(rot_x[-1]), 1e-9)
    den_y = max(np.max(np.abs(rot_y)), 1e-9)

    cx_tilde = rot_x / den_x
    cy_tilde = rot_y / den_y

    return {
        "raw_x": cx,
        "raw_y": cy,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "rot_x": rot_x,
        "rot_y": rot_y,
        "proc_x": cx_tilde,
        "proc_y": cy_tilde,
        "mean_dir": track_dir,   # kept key name for compatibility
    }


# ============================================================
# Example trajectory plots
# ============================================================

def plot_example_pipeline(example_dict, title_prefix="", invert_y=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # raw
    axes[0].plot(example_dict["raw_x"], example_dict["raw_y"], marker="o")
    axes[0].scatter(example_dict["raw_x"][0], example_dict["raw_y"][0], s=50, label="start")
    axes[0].set_title(f"{title_prefix} Raw trajectory")
    axes[0].set_xlabel("cx")
    axes[0].set_ylabel("cy")
    axes[0].axis("equal")
    if invert_y:
        axes[0].invert_yaxis()

    # rotated
    axes[1].plot(example_dict["rot_x"], example_dict["rot_y"], marker="o")
    axes[1].scatter(example_dict["rot_x"][0], example_dict["rot_y"][0], s=50, label="start")
    axes[1].axhline(0, linewidth=1)
    axes[1].axvline(0, linewidth=1)
    axes[1].set_title(f"{title_prefix} Rotated trajectory")
    axes[1].set_xlabel("along-flow")
    axes[1].set_ylabel("cross-flow")
    axes[1].axis("equal")

    # processed
    axes[2].plot(example_dict["proc_x"], example_dict["proc_y"], marker="o")
    axes[2].scatter(example_dict["proc_x"][0], example_dict["proc_y"][0], s=50, label="start")
    axes[2].axhline(0, linewidth=1)
    axes[2].axvline(0, linewidth=1)
    axes[2].set_title(f"{title_prefix} Processed trajectory")
    axes[2].set_xlabel("cx_tilde")
    axes[2].set_ylabel("cy_tilde")

    for ax in axes:
        ax.legend()

    plt.tight_layout()


def extract_example_track_data(segments, min_len=5):
    """
    Returns metadata + processed example for one dataset.
    """
    split, viewpoint, sequence_name, seq_data = find_example_sequence(segments)
    track_key, arr = find_example_track(seq_data, min_len=min_len)
    ex = process_example_track(arr, seq_data)

    return {
        "split": split,
        "viewpoint": viewpoint,
        "sequence_name": sequence_name,
        "track_key": track_key,
        "example": ex,
    }


import numpy as np


def find_example_tracks(seq_data, n_tracks=3, min_len=5):
    """
    Returns up to n_tracks example tracks from a sequence.

    Strategy:
      - keep only valid tracks
      - sort by track length descending
      - return the top n_tracks

    Output:
      list of (track_key, sorted_arr)
    """
    candidates = []

    for track_key, arr in seq_data.items():
        if arr is None or len(arr) == 0:
            continue

        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 6:
            continue

        if len(arr) < min_len:
            continue

        candidates.append((track_key, sort_track(arr), len(arr)))

    if len(candidates) == 0:
        raise ValueError("No suitable example tracks found.")

    # longest first
    candidates.sort(key=lambda x: x[2], reverse=True)

    return [(track_key, arr) for track_key, arr, _ in candidates[:n_tracks]]


def extract_example_tracks_data(segments, min_len=5, n_tracks=3):
    """
    Returns metadata + processed examples for one dataset.

    Output:
      {
        "split": ...,
        "viewpoint": ...,
        "sequence_name": ...,
        "tracks": [
            {
                "track_key": ...,
                "example": ...
            },
            ...
        ]
      }
    """
    split, viewpoint, sequence_name, seq_data = find_example_sequence(segments)
    selected_tracks = find_example_tracks(seq_data, n_tracks=n_tracks, min_len=min_len)

    examples = []
    for track_key, arr in selected_tracks:
        ex = process_example_track(arr, seq_data)
        examples.append({
            "track_key": track_key,
            "example": ex,
        })

    return {
        "split": split,
        "viewpoint": viewpoint,
        "sequence_name": sequence_name,
        "tracks": examples,
    }

def plot_sequence_examples(segments, label="GT"):
    """
    Finds one example sequence and one example track,
    then plots raw / rotated / processed trajectory.
    """
    info = extract_example_track_data(segments)
    ex = info["example"]
    prefix = f"{label} | seq={info['sequence_name']} | track={info['track_key']}"
    plot_example_pipeline(ex, title_prefix=prefix)


def plot_example_overlay(gt_segments, gen_segments, invert_y=True, min_len=5):
    """
    Plot one GT example and one GEN example on the same figure.
    They do not need to come from the same sequence/track id.
    """
    gt_info = extract_example_track_data(gt_segments, min_len=min_len)
    gen_info = extract_example_track_data(gen_segments, min_len=min_len)

    gt = gt_info["example"]
    gen = gen_info["example"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # raw
    axes[0].plot(gt["raw_x"], gt["raw_y"], marker="o", label="GT")
    axes[0].plot(gen["raw_x"], gen["raw_y"], marker="o", label="GEN")
    axes[0].scatter(gt["raw_x"][0], gt["raw_y"][0], s=50)
    axes[0].scatter(gen["raw_x"][0], gen["raw_y"][0], s=50)
    axes[0].set_title("Raw trajectory: GT vs GEN")
    axes[0].set_xlabel("cx")
    axes[0].set_ylabel("cy")
    axes[0].axis("equal")
    if invert_y:
        axes[0].invert_yaxis()

    # rotated
    axes[1].plot(gt["rot_x"], gt["rot_y"], marker="o", label="GT")
    axes[1].plot(gen["rot_x"], gen["rot_y"], marker="o", label="GEN")
    axes[1].scatter(gt["rot_x"][0], gt["rot_y"][0], s=50)
    axes[1].scatter(gen["rot_x"][0], gen["rot_y"][0], s=50)
    axes[1].axhline(0, linewidth=1)
    axes[1].axvline(0, linewidth=1)
    axes[1].set_title("Rotated trajectory: GT vs GEN")
    axes[1].set_xlabel("along-flow")
    axes[1].set_ylabel("cross-flow")
    axes[1].axis("equal")

    # processed
    axes[2].plot(gt["proc_x"], gt["proc_y"], marker="o", label="GT")
    axes[2].plot(gen["proc_x"], gen["proc_y"], marker="o", label="GEN")
    axes[2].scatter(gt["proc_x"][0], gt["proc_y"][0], s=50)
    axes[2].scatter(gen["proc_x"][0], gen["proc_y"][0], s=50)
    axes[2].axhline(0, linewidth=1)
    axes[2].axvline(0, linewidth=1)
    axes[2].set_title("Processed trajectory: GT vs GEN")
    axes[2].set_xlabel("cx_tilde")
    axes[2].set_ylabel("cy_tilde")

    subtitle = (
        f"GT: seq={gt_info['sequence_name']}, track={gt_info['track_key']} | "
        f"GEN: seq={gen_info['sequence_name']}, track={gen_info['track_key']}"
    )
    fig.suptitle(subtitle, fontsize=10)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


def compute_binned_stats(tracks, x_key="proc_x", y_key="proc_y", qlo=25, qhi=75, n_bins=20, use_median=True):
    import numpy as np

    xs_all = []
    ys_all = []

    for tr in tracks:
        ex = tr["example"]
        xs = np.asarray(ex[x_key], dtype=float)
        ys = np.asarray(ex[y_key], dtype=float)

        m = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[m]
        ys = ys[m]

        if len(xs) > 0:
            xs_all.append(xs)
            ys_all.append(ys)

    if not xs_all:
        return None

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)

    eps = 1e-8

    # ---- endpoint groups ----
    mask_0 = np.isclose(xs_all, 0.0, atol=eps)
    mask_1 = np.isclose(xs_all, 1.0, atol=eps)

    # ---- interior groups ----
    mask_mid = (~mask_0) & (~mask_1) & (xs_all > 0.0) & (xs_all < 1.0)
    xs_mid = xs_all[mask_mid]
    ys_mid = ys_all[mask_mid]

    # interior bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x_vals = []
    center_vals = []
    lo_vals = []
    hi_vals = []

    def summarize(x_loc, y_vals):
        if len(y_vals) == 0:
            return
        center = np.median(y_vals) if use_median else np.mean(y_vals)
        lo = np.percentile(y_vals, qlo)
        hi = np.percentile(y_vals, qhi)

        x_vals.append(x_loc)
        center_vals.append(center)
        lo_vals.append(lo)
        hi_vals.append(hi)

    # x = 0 exactly
    summarize(0.0, ys_all[mask_0])

    # interior bins
    if len(xs_mid) > 0:
        bin_idx = np.digitize(xs_mid, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            yb = ys_mid[bin_idx == b]
            if len(yb) == 0:
                continue
            summarize(bin_centers[b], yb)

    # x = 1 exactly
    summarize(1.0, ys_all[mask_1])

    x_vals = np.asarray(x_vals)
    center_vals = np.asarray(center_vals)
    lo_vals = np.asarray(lo_vals)
    hi_vals = np.asarray(hi_vals)

    order = np.argsort(x_vals)
    return x_vals[order], center_vals[order], lo_vals[order], hi_vals[order]

def plot_processed_trajectory_envelope(gt_data, gen_data, spread=(25, 75), n_bins=20):

    fig, ax = plt.subplots(figsize=(6, 6))

    datasets = [
        ("GT", gt_data, "tab:blue"),
        ("GEN", gen_data, "tab:orange"),
    ]


    qlo, qhi = spread

    for label, data, color in datasets:

        stats = compute_binned_stats(
            data["tracks"],
            x_key="proc_x",
            y_key="proc_y",
            qlo=qlo,
            qhi=qhi,
            n_bins=n_bins,
        )

        if stats is None:
            continue

        x, y, lo, hi = stats

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)

        ax.plot(
            x[valid],
            y[valid],
            linewidth=2.5,
            color=color,
            label=f"{label} median",
        )

        ax.fill_between(
            x[valid],
            lo[valid],
            hi[valid],
            color=color,
            alpha=0.20,
            label=f"{label} {qlo}-{qhi}%",
        )

    #ax.set_title(r"Normalized trajectory shape")
    ax.set_xlabel(r"$\tilde{c}_x$")
    ax.set_ylabel(r"$\tilde{c}_y$")
    #ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect(0.5)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def compute_binned_stats_mean(
    tracks,
    x_key="proc_x",
    y_key="proc_y",
    n_bins=20,
    use_mean=True,
    ddof=0,
):
    import numpy as np

    xs_all = []
    ys_all = []

    for tr in tracks:
        ex = tr["example"]
        xs = np.asarray(ex[x_key], dtype=float)
        ys = np.asarray(ex[y_key], dtype=float)

        m = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[m]
        ys = ys[m]

        if len(xs) > 0:
            xs_all.append(xs)
            ys_all.append(ys)

    if not xs_all:
        return None

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)

    eps = 1e-8

    # endpoint groups
    mask_0 = np.isclose(xs_all, 0.0, atol=eps)
    mask_1 = np.isclose(xs_all, 1.0, atol=eps)

    # interior groups
    mask_mid = (~mask_0) & (~mask_1) & (xs_all > 0.0) & (xs_all < 1.0)
    xs_mid = xs_all[mask_mid]
    ys_mid = ys_all[mask_mid]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x_vals = []
    center_vals = []
    lo_vals = []
    hi_vals = []

    def summarize(x_loc, y_vals):
        if len(y_vals) == 0:
            return

        center = np.mean(y_vals) if use_mean else np.median(y_vals)
        std = np.std(y_vals, ddof=ddof)
        print(std)

        lo = center - std
        hi = center + std

        x_vals.append(x_loc)
        center_vals.append(center)
        lo_vals.append(lo)
        hi_vals.append(hi)

    # x = 0 exactly
    summarize(0.0, ys_all[mask_0])

    # interior bins
    if len(xs_mid) > 0:
        bin_idx = np.digitize(xs_mid, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            yb = ys_mid[bin_idx == b]
            if len(yb) == 0:
                continue
            summarize(bin_centers[b], yb)

    # x = 1 exactly
    summarize(1.0, ys_all[mask_1])

    x_vals = np.asarray(x_vals)
    center_vals = np.asarray(center_vals)
    lo_vals = np.asarray(lo_vals)
    hi_vals = np.asarray(hi_vals)

    order = np.argsort(x_vals)
    return x_vals[order], center_vals[order], lo_vals[order], hi_vals[order]

def plot_processed_trajectory_envelope_mean(gt_data, gen_data, n_bins=20):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    datasets = [
        ("GT", gt_data, "tab:blue"),
        ("GEN", gen_data, "tab:orange"),
    ]

    for label, data, color in datasets:
        stats = compute_binned_stats_mean(
            data["tracks"],
            x_key="proc_x",
            y_key="proc_y",
            n_bins=n_bins,
            use_mean=True,
            ddof=0,
        )

        if stats is None:
            continue

        x, y, lo, hi = stats
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)

        ax.plot(
            x[valid],
            y[valid],
            linewidth=2.5,
            color=color,
            label=f"{label} mean",
        )

        ax.fill_between(
            x[valid],
            lo[valid],
            hi[valid],
            color=color,
            alpha=0.20,
            label=f"{label} mean ± std",
        )

    ax.set_xlabel(r"$\tilde{c}_x$")
    ax.set_ylabel(r"$\tilde{c}_y$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect(0.5)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    dataset_root = "C:/Users/Saydez/OneDrive/Documents/Phd/Codes/tools/slope_track"
    dataset_root_gen = "gen"

    # ----------------------------
    # Load GT
    # ----------------------------
    segments = load_gt_tracks_all_splits_by_viewpoints(
        dataset_root=dataset_root,
        viewpoints=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    # ----------------------------
    # Load GEN
    # ----------------------------
    segments_gen_B = load_gen_tracks_all_splits_select_viewpoints(
        dataset_root=dataset_root_gen,
        splits=("train", "val", "test"),
        selected_viewpoints=[1,2,3,4,5,6,7,8],#,2,3,4,5,6,
        base_seq_id_for_view1=33,
        num_viewpoints=8,
    )

    # ----------------------------
    # Normalize
    # ----------------------------

    segments_f = segments #filter_segments_by_motion(segments, min_displacement=40.0)
    segments_g = segments_gen_B #filter_segments_by_motion(segments_gen_B, min_displacement=30.0)
    A_norm = normalize_segments_by_trajectory_direction(segments_f)
    B_norm = normalize_segments_by_trajectory_direction(segments_g)

    # ----------------------------
    # Wasserstein summary
    # ----------------------------
    w = wasserstein_per_component(A_norm, B_norm)
    print("Wasserstein per component:")
    print(w)

    # ----------------------------
    # Flatten normalized components
    # ----------------------------
    A_cx = flatten_component(A_norm, 2)
    A_cy = flatten_component(A_norm, 3)
    A_w  = flatten_component(A_norm, 4)
    A_h  = flatten_component(A_norm, 5)

    B_cx = flatten_component(B_norm, 2)
    B_cy = flatten_component(B_norm, 3)
    B_w  = flatten_component(B_norm, 4)
    B_h  = flatten_component(B_norm, 5)

    print("GT sample sizes:", len(A_cx), len(A_cy), len(A_w), len(A_h))
    print("GEN sample sizes:", len(B_cx), len(B_cy), len(B_w), len(B_h))
    print(f"GT / GEN size ratio: {len(A_cx) / max(len(B_cx), 1):.2f}x")

    # matched-size versions for plotting
    A_cx_plot, B_cx_plot = match_sample_size(A_cx, B_cx, seed=0)
    A_cy_plot, B_cy_plot = match_sample_size(A_cy, B_cy, seed=0)
    A_w_plot,  B_w_plot  = match_sample_size(A_w,  B_w,  seed=0)
    A_h_plot,  B_h_plot  = match_sample_size(A_h,  B_h,  seed=0)

    print("Using matched sample size for plots:", len(A_cx_plot))

    #debug_extreme_tracks(B_norm, col_idx=2, threshold=10.0, name="cx_tilde")

    """# ----------------------------
    # Plot separate examples
    # ----------------------------
    plot_sequence_examples(segments, label="GT")
    plot_sequence_examples(segments_gen_B, label="GEN")

    # ----------------------------
    # Plot GT and GEN example on same figure
    # ----------------------------
    plot_example_overlay(segments, segments_gen_B)"""

    gt_examples = extract_example_tracks_data(segments, min_len=5, n_tracks=555528)
    gen_examples = extract_example_tracks_data(segments_gen_B, min_len=5, n_tracks=555528)
    plot_processed_trajectory_envelope(gt_examples, gen_examples)
    plot_processed_trajectory_envelope_mean(gt_examples, gen_examples)

    A_mask = np.abs(A_cx_plot) < 2
    B_mask = np.abs(B_cx_plot) < 2

    A_cx_plot = A_cx_plot[A_mask]
    A_cy_plot = A_cy_plot[A_mask]

    B_cx_plot = B_cx_plot[B_mask]
    B_cy_plot = B_cy_plot[B_mask]

    # ----------------------------
    # 1D histograms
    # ----------------------------
    plot_hist_overlay(A_cx_plot, B_cx_plot, "cx_tilde", label_a="GT", label_b="GEN")
    plot_hist_overlay(A_cy_plot, B_cy_plot, "cy_tilde", label_a="GT", label_b="GEN")
    plot_hist_overlay(A_w_plot,  B_w_plot,  "w_tilde",  label_a="GT", label_b="GEN")
    plot_hist_overlay(A_h_plot,  B_h_plot,  "h_tilde",  label_a="GT", label_b="GEN")

    # ----------------------------
    # 1D KDE curves
    # ----------------------------
    plot_kde_overlay(A_cx_plot, B_cx_plot, "cx_tilde", label_a="GT", label_b="GEN")
    plot_kde_overlay(A_cy_plot, B_cy_plot, "cy_tilde", label_a="GT", label_b="GEN")
    plot_kde_overlay(A_w_plot,  B_w_plot,  "w_tilde",  label_a="GT", label_b="GEN")
    plot_kde_overlay(A_h_plot,  B_h_plot,  "h_tilde",  label_a="GT", label_b="GEN")

    # ----------------------------
    # Separate 2D densities
    # ----------------------------
    plot_2d_density(A_cx_plot, A_cy_plot, "GT: 2D density of processed trajectories")
    plot_2d_density(B_cx_plot, B_cy_plot, "GEN: 2D density of processed trajectories")

    # ----------------------------
    # Side-by-side 2D density comparison
    # ----------------------------
    xlo = min(A_cx_plot.min(), B_cx_plot.min())
    xhi = max(A_cx_plot.max(), B_cx_plot.max())
    ylo = min(A_cy_plot.min(), B_cy_plot.min())
    yhi = max(A_cy_plot.max(), B_cy_plot.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h1 = plot_2d_density_comparison(
        axes[0], A_cx_plot, A_cy_plot,
        title="GT: cx_tilde vs cy_tilde",
        xlim=(0, 1), #(xlo, xhi)
        ylim=(ylo, yhi),
        bins=120
    )
    fig.colorbar(h1[3], ax=axes[0], label="density")

    h2 = plot_2d_density_comparison(
        axes[1], B_cx_plot, B_cy_plot,
        title="GEN: cx_tilde vs cy_tilde",
        xlim=(0, 1),
        ylim=(ylo, yhi),
        bins=120
    )
    fig.colorbar(h2[3], ax=axes[1], label="density")

    plt.tight_layout()
    plt.show()