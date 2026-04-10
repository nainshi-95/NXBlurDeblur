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
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return bilinear_sample(img, yy - dy, xx - dx)


def shift_1d_along_y(arr, dy):
    """
    arr: [k]
    shift in y-direction with linear interpolation
    arr_shifted[y] = arr[y - dy]
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
    oblique edge
    theta_deg is angle of edge normal
    """
    theta = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = (xx - edge_x0) * np.cos(theta) + (yy - H/2) * np.sin(theta)
    img = 0.5 * (1.0 + np.tanh(dist / softness))
    return img


def make_kx6_left_boundary_case(k=48, true_dy=0.5, angle_deg=18):
    """
    left 3 cols = reference
    right 3 cols = predictor
    predictor side has vertical misalignment only
    """
    base = make_edge_image(H=64, W=16, theta_deg=angle_deg, edge_x0=2.6, softness=0.8)
    y0 = (64 - k) // 2
    y1 = y0 + k

    ref_full = shift_patch(base, dx=0.0, dy=0.0)
    pred_full = shift_patch(base, dx=0.0, dy=true_dy)

    left = ref_full[y0:y1, 0:3]     # [k,3]
    right = pred_full[y0:y1, 3:6]   # [k,3]
    patch = np.concatenate([left, right], axis=1)
    return left, right, patch


# =========================================================
# 3) edge trajectory extraction
# =========================================================
def extract_edge_curve(strip3, eps=1e-6):
    """
    strip3: [k,3]

    returns
      pos      : [k]  row-wise edge position in strip
      strength : [k]  row-wise edge strength
      gx_abs   : [k,3]
    """
    gx = sobel_x(strip3)
    gx_abs = np.abs(gx) + eps

    xs = np.arange(3, dtype=np.float64)[None, :]
    pos = (gx_abs * xs).sum(axis=1) / gx_abs.sum(axis=1)
    strength = gx_abs.max(axis=1)

    return pos, strength, gx_abs


# =========================================================
# 4) trajectory-based 3-way score
# =========================================================
CANDIDATES = {
    "up":   -0.5,
    "stop":  0.0,
    "down":  0.5,
}


def trajectory_score(left, right, cand_dy, strength_tau=0.03,
                     alpha=2.0, beta=1.0, gamma=0.5):
    """
    left, right: [k,3]
    cand_dy: candidate correction applied to right in y-direction

    lower is better
    """
    # extract row-wise edge curves
    pL, sL, _ = extract_edge_curve(left)
    pR, sR, _ = extract_edge_curve(right)

    # shift right trajectory in y
    pR_shift = shift_1d_along_y(pR, cand_dy)
    sR_shift = shift_1d_along_y(sR, cand_dy)

    # edge-valid rows
    w = np.minimum(sL, sR_shift)
    mask = (w > strength_tau).astype(np.float64)
    mask_sum = max(mask.sum(), 1.0)

    # 1) position mismatch
    pos_term = (np.abs(pL - pR_shift) * w * mask).sum() / (w * mask).sum().clip(min=1e-6)

    # 2) slope mismatch
    dpL = np.diff(pL)
    dpR = np.diff(pR_shift)
    w2 = np.minimum(w[:-1], w[1:])
    mask2 = (w2 > strength_tau).astype(np.float64)
    slope_term = (np.abs(dpL - dpR) * w2 * mask2).sum() / (w2 * mask2).sum().clip(min=1e-6)

    # 3) seam intensity consistency
    # compare seam-adjacent columns after shifting right[:,0]
    r0 = shift_1d_along_y(right[:, 0], cand_dy)
    seam_term = np.mean(np.abs(left[:, 2] - r0))

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


# =========================================================
# 5) visualization
# =========================================================
def visualize_cases(cases):
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3.5 * len(cases)))
    if len(cases) == 1:
        axes = axes[None, :]

    summary = []

    for row, (title, true_dy, angle_deg) in enumerate(cases):
        left, right, patch = make_kx6_left_boundary_case(k=48, true_dy=true_dy, angle_deg=angle_deg)
        scored = score_3way(left, right)

        names = list(CANDIDATES.keys())
        vals = [scored[n]["total"] for n in names]
        pred = names[int(np.argmin(vals))]

        # panel 1: input patch
        ax = axes[row, 0]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.axvline(2.5, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{title}\ninput kx6 patch")
        ax.set_xticks(range(6))
        ax.set_yticks([])

        # panel 2: score bars
        ax = axes[row, 1]
        bars = ax.bar(names, vals)
        bars[int(np.argmin(vals))].set_edgecolor("black")
        bars[int(np.argmin(vals))].set_linewidth(2)
        ax.set_title("trajectory score")
        ax.set_ylabel("lower is better")

        # panel 3: raw trajectories
        ax = axes[row, 2]
        pL = scored[pred]["pL"]
        pR = scored[pred]["pR"]
        y = np.arange(len(pL))
        ax.plot(pL, y, label="left curve")
        ax.plot(pR, y, label="right curve (raw)")
        ax.invert_yaxis()
        ax.set_xlabel("edge position in 3-col strip")
        ax.set_title("raw edge trajectory")
        ax.legend(fontsize=8)

        # panel 4: aligned trajectories
        ax = axes[row, 3]
        pR_shift = scored[pred]["pR_shift"]
        ax.plot(pL, y, label="left curve")
        ax.plot(pR_shift, y, label=f"right shifted ({pred})")
        ax.invert_yaxis()
        ax.set_xlabel("edge position in 3-col strip")
        ax.set_title(f"aligned trajectory, best={pred}")
        ax.legend(fontsize=8)

        summary.append((
            title,
            true_dy,
            pred,
            {n: round(scored[n]["total"], 4) for n in names}
        ))

    plt.tight_layout()
    plt.show()

    print("\nSummary")
    for title, true_dy, pred, scores in summary:
        print(f"\n{title} | true_dy={true_dy} | predicted best={pred}")
        print(scores)


# =========================================================
# 6) run test
# =========================================================
cases = [
    ("true up 0.5px",   -0.5, 18),
    ("true stop 0.0px",  0.0, 18),
    ("true down 0.5px",  0.5, 18),
    ("true up 1.0px",   -1.0, 18),
    ("true down 1.0px",  1.0, 18),
    ("true down 0.5px, steeper edge", 0.5, 32),
]

visualize_cases(cases)
