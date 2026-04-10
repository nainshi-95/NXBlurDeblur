import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) basic utils
# =========================================================
def conv2_same(img, kernel):
    H, W = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(pad[i:i+kh, j:j+kw] * kernel)
    return out


def sobel_x(img):
    # horizontal gradient -> vertical edge detection
    k = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]], dtype=np.float64) / 4.0
    return conv2_same(img, k)


def bilinear_sample(img, ys, xs):
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
    output(y,x) = input(y-dy, x-dx)
    """
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return bilinear_sample(img, yy - dy, xx - dx)


def shift_1d(arr, dy):
    """
    arr: [k]
    out[y] = arr[y-dy]
    """
    k = len(arr)
    y = np.arange(k, dtype=np.float64)
    src = np.clip(y - dy, 0, k - 1)

    y0 = np.floor(src).astype(int)
    y1 = np.clip(y0 + 1, 0, k - 1)
    w = src - y0

    return arr[y0] * (1 - w) + arr[y1] * w


# =========================================================
# 2) synthetic data
# =========================================================
def make_edge_image(H=64, W=16, theta_deg=18, edge_x0=2.6, softness=0.8):
    """
    theta_deg: angle of edge normal
    """
    theta = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = (xx - edge_x0) * np.cos(theta) + (yy - H / 2) * np.sin(theta)
    img = 0.5 * (1.0 + np.tanh(dist / softness))
    return img


def make_kx6_left_boundary_case(k=48, true_dy=0.5, angle_deg=18):
    """
    left 3 cols: reference
    right 3 cols: predictor
    predictor side has vertical misalignment true_dy
    """
    base = make_edge_image(H=64, W=16, theta_deg=angle_deg, edge_x0=2.6, softness=0.8)

    y0 = (64 - k) // 2
    y1 = y0 + k

    ref_full = shift_patch(base, dx=0.0, dy=0.0)
    pred_full = shift_patch(base, dx=0.0, dy=true_dy)

    left = ref_full[y0:y1, 0:3]
    right = pred_full[y0:y1, 3:6]
    patch = np.concatenate([left, right], axis=1)

    return left, right, patch


# =========================================================
# 3) soft edge trajectory extraction
# =========================================================
def extract_edge_curve(strip3, eps=1e-6):
    """
    strip3: [k,3]

    returns:
      pos      : [k] soft edge position inside 3-col strip
      strength : [k] row-wise edge strength
      gx_abs   : [k,3]
    """
    gx = sobel_x(strip3)
    gx_abs = np.abs(gx) + eps

    xs = np.arange(3, dtype=np.float64)[None, :]
    pos = (gx_abs * xs).sum(axis=1) / gx_abs.sum(axis=1)
    strength = gx_abs.max(axis=1)

    return pos, strength, gx_abs


# =========================================================
# 4) 3-way score using position + slope continuity
# =========================================================
CANDIDATES = {
    "up":   -0.5,
    "stop":  0.0,
    "down":  0.5,
}


def trajectory_score(left, right, cand_dy,
                     strength_tau=0.02,
                     alpha=2.0,   # position mismatch weight
                     beta=1.2,    # slope mismatch weight
                     gamma=0.5):  # seam intensity weight
    """
    left, right: [k,3]
    cand_dy: correction applied to right side

    lower is better
    """
    pL, sL, _ = extract_edge_curve(left)
    pR, sR, _ = extract_edge_curve(right)

    # shift right trajectory by candidate dy
    pR_shift = shift_1d(pR, cand_dy)
    sR_shift = shift_1d(sR, cand_dy)

    # row weights: only trust rows where both sides have edge
    w = np.minimum(sL, sR_shift)
    mask = (w > strength_tau).astype(np.float64)
    wm = w * mask
    wm_sum = max(wm.sum(), 1e-6)

    # 1) position continuity
    pos_term = (np.abs(pL - pR_shift) * wm).sum() / wm_sum

    # 2) slope continuity: trend of edge movement across rows
    dpL = np.diff(pL)
    dpR = np.diff(pR_shift)

    w2 = np.minimum(w[:-1], w[1:])
    mask2 = (w2 > strength_tau).astype(np.float64)
    wm2 = w2 * mask2
    wm2_sum = max(wm2.sum(), 1e-6)

    slope_term = (np.abs(dpL - dpR) * wm2).sum() / wm2_sum

    # 3) seam intensity continuity
    # compare seam-adjacent columns after shifting right[:,0]
    r0_shift = shift_1d(right[:, 0], cand_dy)
    seam_term = np.mean(np.abs(left[:, 2] - r0_shift))

    total = alpha * pos_term + beta * slope_term + gamma * seam_term

    return {
        "total": total,
        "pos_term": pos_term,
        "slope_term": slope_term,
        "seam_term": seam_term,
        "pL": pL,
        "pR": pR,
        "pR_shift": pR_shift,
        "sL": sL,
        "sR": sR,
        "w": w,
    }


def score_3way(left, right):
    out = {}
    for name, dy in CANDIDATES.items():
        out[name] = trajectory_score(left, right, dy)
    return out


def decide_3way(left, right, threshold=0.01):
    """
    gain-based decision relative to stop
    """
    scored = score_3way(left, right)

    s_up = scored["up"]["total"]
    s_stop = scored["stop"]["total"]
    s_down = scored["down"]["total"]

    gain_up = s_stop - s_up
    gain_down = s_stop - s_down

    if max(gain_up, gain_down) < threshold:
        pred = "stop"
    elif gain_up > gain_down:
        pred = "up"
    else:
        pred = "down"

    return pred, scored, gain_up, gain_down


# =========================================================
# 5) visualization helpers
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


def visualize_cases(cases, threshold=0.01, save_path=None):
    fig, axes = plt.subplots(len(cases), 5, figsize=(20, 3.6 * len(cases)))
    if len(cases) == 1:
        axes = axes[None, :]

    summary = []

    for row, (title, true_dy, angle_deg) in enumerate(cases):
        left, right, patch = make_kx6_left_boundary_case(
            k=48,
            true_dy=true_dy,
            angle_deg=angle_deg
        )

        pred, scored, gain_up, gain_down = decide_3way(left, right, threshold=threshold)
        gt_label = true_label_from_true_dy(true_dy)

        right_corr, patch_corr = make_corrected_patch(left, right, pred)

        # panel 1: input patch
        ax = axes[row, 0]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axvline(2.5, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{title}\nGT={gt_label}, Pred={pred}")
        ax.set_xticks(range(6))
        ax.set_yticks([])

        # panel 2: corrected patch
        ax = axes[row, 1]
        ax.imshow(patch_corr, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axvline(2.5, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"corrected ({pred})")
        ax.set_xticks(range(6))
        ax.set_yticks([])

        # panel 3: candidate scores
        ax = axes[row, 2]
        names = ["up", "stop", "down"]
        vals = [scored[n]["total"] for n in names]
        bars = ax.bar(names, vals)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)
        ax.set_title("candidate score")
        ax.set_ylabel("lower is better")

        # panel 4: raw trajectories
        ax = axes[row, 3]
        y = np.arange(len(scored[pred]["pL"]))
        ax.plot(scored[pred]["pL"], y, label="left curve")
        ax.plot(scored[pred]["pR"], y, label="right curve raw")
        ax.invert_yaxis()
        ax.set_xlabel("edge pos in 3-col strip")
        ax.set_title("raw edge trajectory")
        ax.legend(fontsize=8)

        # panel 5: aligned trajectories
        ax = axes[row, 4]
        ax.plot(scored[pred]["pL"], y, label="left curve")
        ax.plot(scored[pred]["pR_shift"], y, label=f"right shifted ({pred})")
        ax.invert_yaxis()
        ax.set_xlabel("edge pos in 3-col strip")
        ax.set_title(
            f"aligned trajectory\n"
            f"gain_up={gain_up:.4f}, gain_down={gain_down:.4f}"
        )
        ax.legend(fontsize=8)

        summary.append({
            "title": title,
            "true_dy": true_dy,
            "gt_label": gt_label,
            "pred": pred,
            "scores": {n: scored[n]["total"] for n in names},
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
            f"| GT={item['gt_label']} | Pred={item['pred']} | correct={is_correct}"
        )
        print(
            f"  scores: "
            f"up={item['scores']['up']:.4f}, "
            f"stop={item['scores']['stop']:.4f}, "
            f"down={item['scores']['down']:.4f}"
        )
        print(
            f"  gains: "
            f"gain_up={item['gain_up']:.4f}, "
            f"gain_down={item['gain_down']:.4f}"
        )

    acc = correct / len(summary)
    print(f"\nAccuracy: {correct}/{len(summary)} = {acc:.4f}")


# =========================================================
# 6) run
# =========================================================
if __name__ == "__main__":
    cases = [
        ("true up 0.5px",   -0.5, 18),
        ("true stop 0.0px",  0.0, 18),
        ("true down 0.5px",  0.5, 18),
        ("true up 1.0px",   -1.0, 18),
        ("true down 1.0px",  1.0, 18),
        ("true down 0.5px, steeper edge",  0.5, 32),
        ("true up 0.5px, steeper edge",   -0.5, 32),
        ("true stop, steeper edge",        0.0, 32),
    ]

    visualize_cases(
        cases,
        threshold=0.01,
        save_path="left_boundary_trajectory_score_test.png"
    )
