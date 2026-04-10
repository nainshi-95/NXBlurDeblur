import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) basic utils
# =========================================================
def bilinear_sample(img, ys, xs):
    """
    img: [H, W]
    ys, xs: float coordinate arrays with same shape
    border replication
    """
    H, W = img.shape
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = ys - y0
    wx = xs - x0

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return (
        Ia * (1 - wy) * (1 - wx)
        + Ib * (1 - wy) * wx
        + Ic * wy * (1 - wx)
        + Id * wy * wx
    )


def shift_patch(img, dx=0.0, dy=0.0):
    """
    shift 2D image by (dx, dy) using bilinear interpolation
    output(y, x) = input(y - dy, x - dx)
    """
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return bilinear_sample(img, yy - dy, xx - dx)


def shift_1d(arr, dy):
    """
    arr: [k]
    shift along vertical direction with linear interpolation
    out[y] = arr[y - dy]
    """
    k = len(arr)
    y = np.arange(k, dtype=np.float64)
    src = np.clip(y - dy, 0, k - 1)

    y0 = np.floor(src).astype(int)
    y1 = np.clip(y0 + 1, 0, k - 1)
    w = src - y0

    return arr[y0] * (1 - w) + arr[y1] * w


def zncc(a, b, eps=1e-6):
    """
    zero-mean normalized cross-correlation
    higher is better
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    a = a - a.mean()
    b = b - b.mean()

    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + eps
    return (a * b).sum() / denom


# =========================================================
# 2) synthetic data generation
# =========================================================
def make_edge_image(H=64, W=16, theta_deg=18, edge_x0=2.6, softness=0.8):
    """
    Make a smooth oblique edge image.
    theta_deg = angle of edge normal
    """
    theta = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    dist = (xx - edge_x0) * np.cos(theta) + (yy - H / 2) * np.sin(theta)
    img = 0.5 * (1.0 + np.tanh(dist / softness))
    return img


def make_kx6_left_boundary_case(k=48, true_dy=0.5, angle_deg=18):
    """
    left 3 cols  = reference
    right 3 cols = current predictor
    predictor side has vertical misalignment only
    """
    base = make_edge_image(H=64, W=16, theta_deg=angle_deg, edge_x0=2.6, softness=0.8)

    y0 = (64 - k) // 2
    y1 = y0 + k

    ref_full = shift_patch(base, dx=0.0, dy=0.0)
    pred_full = shift_patch(base, dx=0.0, dy=true_dy)

    left = ref_full[y0:y1, 0:3]     # [k, 3]
    right = pred_full[y0:y1, 3:6]   # [k, 3]

    patch = np.concatenate([left, right], axis=1)  # [k, 6]
    return left, right, patch


# =========================================================
# 3) 3-way score
# =========================================================
CANDIDATES = {
    "up":   -0.5,
    "stop":  0.0,
    "down":  0.5,
}


def score_shift_zncc(left, right, dy):
    """
    left, right: [k, 3]
    Compare left[:, x] and shifted right[:, x] for each x in {0,1,2}
    using ZNCC. Lower score is better (negative correlation).
    """
    k = left.shape[0]
    total_corr = 0.0

    for x in range(3):
        r_shift = shift_1d(right[:, x], dy)
        corr = zncc(left[:, x], r_shift)
        total_corr += corr

    avg_corr = total_corr / 3.0
    score = -avg_corr  # lower is better
    return score


def score_3way(left, right):
    scores = {}
    for name, dy in CANDIDATES.items():
        scores[name] = score_shift_zncc(left, right, dy)
    return scores


def decide_3way(left, right, threshold=0.01):
    """
    gain-based decision:
      gain_up   = S(stop) - S(up)
      gain_down = S(stop) - S(down)

    lower score is better.
    """
    s = score_3way(left, right)

    gain_up = s["stop"] - s["up"]
    gain_down = s["stop"] - s["down"]

    if max(gain_up, gain_down) < threshold:
        pred = "stop"
    elif gain_up > gain_down:
        pred = "up"
    else:
        pred = "down"

    return pred, s, gain_up, gain_down


# =========================================================
# 4) helper for visualization
# =========================================================
def make_corrected_patch(left, right, pred_label):
    dy = CANDIDATES[pred_label]
    right_corr = np.stack([shift_1d(right[:, x], dy) for x in range(3)], axis=1)
    patch_corr = np.concatenate([left, right_corr], axis=1)
    return right_corr, patch_corr


def true_label_from_true_dy(true_dy, tol=0.25):
    if true_dy < -tol:
        return "up"
    elif true_dy > tol:
        return "down"
    else:
        return "stop"


# =========================================================
# 5) main visualization routine
# =========================================================
def visualize_cases(cases, threshold=0.01, save_path=None):
    """
    cases: list of tuples
      (title, true_dy, angle_deg)
    """
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3.5 * len(cases)))
    if len(cases) == 1:
        axes = axes[None, :]

    summary = []

    for row, (title, true_dy, angle_deg) in enumerate(cases):
        left, right, patch = make_kx6_left_boundary_case(
            k=48,
            true_dy=true_dy,
            angle_deg=angle_deg
        )

        pred, scores, gain_up, gain_down = decide_3way(left, right, threshold=threshold)
        right_corr, patch_corr = make_corrected_patch(left, right, pred)

        gt_label = true_label_from_true_dy(true_dy)

        # -------------------------
        # panel 1: input patch
        # -------------------------
        ax = axes[row, 0]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axvline(2.5, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{title}\nGT={gt_label}, Pred={pred}")
        ax.set_xticks(range(6))
        ax.set_yticks([])

        # -------------------------
        # panel 2: corrected patch
        # -------------------------
        ax = axes[row, 1]
        ax.imshow(patch_corr, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axvline(2.5, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"corrected patch ({pred})")
        ax.set_xticks(range(6))
        ax.set_yticks([])

        # -------------------------
        # panel 3: 3-way scores
        # -------------------------
        ax = axes[row, 2]
        names = ["up", "stop", "down"]
        vals = [scores[n] for n in names]
        bars = ax.bar(names, vals)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)
        ax.set_title("3-way score (lower is better)")
        ax.set_ylabel("score")

        # -------------------------
        # panel 4: row signals
        # -------------------------
        ax = axes[row, 3]
        y = np.arange(left.shape[0])

        ax.plot(left[:, 2], y, label="left[:,2] (ref seam)")
        ax.plot(right[:, 0], y, label="right[:,0] raw")
        ax.plot(right_corr[:, 0], y, label=f"right[:,0] corrected ({pred})")
        ax.invert_yaxis()
        ax.set_title(
            f"gain_up={gain_up:.4f}, gain_down={gain_down:.4f}"
        )
        ax.set_xlabel("intensity")
        ax.legend(fontsize=8)

        summary.append({
            "title": title,
            "true_dy": true_dy,
            "gt_label": gt_label,
            "pred": pred,
            "scores": scores,
            "gain_up": gain_up,
            "gain_down": gain_down,
        })

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

    print("\nSummary")
    correct = 0
    for item in summary:
        is_correct = (item["gt_label"] == item["pred"])
        correct += int(is_correct)

        print(
            f"\n{item['title']} | true_dy={item['true_dy']:.2f} "
            f"| GT={item['gt_label']} | Pred={item['pred']} "
            f"| correct={is_correct}"
        )
        print(
            f"  scores: up={item['scores']['up']:.4f}, "
            f"stop={item['scores']['stop']:.4f}, "
            f"down={item['scores']['down']:.4f}"
        )
        print(
            f"  gains: gain_up={item['gain_up']:.4f}, "
            f"gain_down={item['gain_down']:.4f}"
        )

    acc = correct / len(summary)
    print(f"\nAccuracy: {correct}/{len(summary)} = {acc:.4f}")


# =========================================================
# 6) run test
# =========================================================
if __name__ == "__main__":
    cases = [
        ("true up 0.5px",   -0.5, 18),
        ("true stop 0.0px",  0.0, 18),
        ("true down 0.5px",  0.5, 18),
        ("true up 1.0px",   -1.0, 18),
        ("true down 1.0px",  1.0, 18),
        ("true down 0.5px, steeper edge", 0.5, 32),
        ("true up 0.5px, steeper edge",  -0.5, 32),
    ]

    visualize_cases(
        cases,
        threshold=0.01,
        save_path="left_boundary_zncc_3way_test.png"
    )
