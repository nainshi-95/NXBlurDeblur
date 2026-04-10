import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) basic ops
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


def sobel_y(img):
    k = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]], dtype=np.float64) / 4.0
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


# =========================================================
# 2) synthetic data
# =========================================================
def make_edge_image(H=64, W=16, theta_deg=20, edge_x0=2.7, softness=0.8):
    """
    theta_deg: angle of edge normal
    """
    theta = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = (xx - edge_x0) * np.cos(theta) + (yy - H / 2) * np.sin(theta)
    img = 0.5 * (1.0 + np.tanh(dist / softness))
    return img


def make_kx6_left_boundary_case(k=48, true_dy=0.5, angle_deg=20):
    """
    left 3 cols : reference
    right 3 cols: predictor
    predictor side has vertical misalignment true_dy
    """
    base = make_edge_image(H=64, W=16, theta_deg=angle_deg, edge_x0=2.7, softness=0.8)

    y0 = (64 - k) // 2
    y1 = y0 + k

    ref_full = shift_patch(base, dx=0.0, dy=0.0)
    pred_full = shift_patch(base, dx=0.0, dy=true_dy)

    left = ref_full[y0:y1, 0:3]
    right = pred_full[y0:y1, 3:6]
    patch = np.concatenate([left, right], axis=1)
    return left, right, patch


# =========================================================
# 3) orientation-guided interpolation score
# =========================================================
CANDIDATES = {
    "up":   -0.5,
    "stop":  0.0,
    "down":  0.5,
}


def estimate_row_orientation(patch6, eps=1e-6):
    """
    patch6: [k,6]
    For each row, estimate edge normal from seam-near region.
    returns:
      nx, ny, mag: [k]
    """
    gx = sobel_x(patch6)
    gy = sobel_y(patch6)
    mag = np.sqrt(gx**2 + gy**2) + eps

    # seam-near columns: 2 and 3
    cols = [2, 3]
    w = mag[:, cols]
    gxw = gx[:, cols]
    gyw = gy[:, cols]

    nx = (gxw * w).sum(axis=1) / (w.sum(axis=1) + eps)
    ny = (gyw * w).sum(axis=1) / (w.sum(axis=1) + eps)

    norm = np.sqrt(nx**2 + ny**2) + eps
    nx = nx / norm
    ny = ny / norm
    mag_row = w.mean(axis=1)

    return nx, ny, mag_row


def sample_profiles_for_row(left, right_shifted, y, nx, ny,
                            normal_offsets=(-1.5, -0.5, 0.5, 1.5),
                            tangent_offsets=(-1.0, 0.0, 1.0)):
    """
    left, right_shifted: [k,3]
    y: scalar row center
    nx, ny: normal direction for this row

    seam-adjacent anchor points:
      left  anchor at x=2
      right anchor at x=0

    normal profile: sample across edge normal
    tangent profile: sample along edge tangent
    """
    # tangent direction = rotate normal by +90 deg
    tx, ty = -ny, nx

    # anchor points in local strip coordinates
    lx0, rx0 = 2.0, 0.0
    ly0, ry0 = float(y), float(y)

    left_normal = []
    right_normal = []

    for s in normal_offsets:
        left_normal.append(
            bilinear_sample(left,
                            np.array([[ly0 + s * ny]]),
                            np.array([[lx0 + s * nx]])).item()
        )
        right_normal.append(
            bilinear_sample(right_shifted,
                            np.array([[ry0 + s * ny]]),
                            np.array([[rx0 + s * nx]])).item()
        )

    left_tangent = []
    right_tangent = []

    for s in tangent_offsets:
        left_tangent.append(
            bilinear_sample(left,
                            np.array([[ly0 + s * ty]]),
                            np.array([[lx0 + s * tx]])).item()
        )
        right_tangent.append(
            bilinear_sample(right_shifted,
                            np.array([[ry0 + s * ty]]),
                            np.array([[rx0 + s * tx]])).item()
        )

    return (
        np.array(left_normal, dtype=np.float64),
        np.array(right_normal, dtype=np.float64),
        np.array(left_tangent, dtype=np.float64),
        np.array(right_tangent, dtype=np.float64),
    )


def candidate_score(left, right, cand_dy,
                    mag_tau=0.03,
                    normal_offsets=(-1.5, -0.5, 0.5, 1.5),
                    tangent_offsets=(-1.0, 0.0, 1.0),
                    alpha=1.6,   # normal profile consistency
                    beta=1.0,    # tangent profile consistency
                    gamma=0.4):  # seam raw intensity jump
    """
    lower is better
    """
    right_shifted = shift_patch(right, dx=0.0, dy=cand_dy)
    patch6 = np.concatenate([left, right_shifted], axis=1)

    nx, ny, mag_row = estimate_row_orientation(patch6)
    k = left.shape[0]

    total_normal = 0.0
    total_tangent = 0.0
    total_seam = 0.0
    total_w = 0.0

    used_rows = []
    debug_profiles = None

    for y in range(1, k - 1):
        w = mag_row[y]
        if w < mag_tau:
            continue

        ln, rn, lt, rt = sample_profiles_for_row(
            left, right_shifted, y, nx[y], ny[y],
            normal_offsets=normal_offsets,
            tangent_offsets=tangent_offsets,
        )

        normal_err = np.mean(np.abs(ln - rn))
        tangent_err = np.mean(np.abs(lt - rt))
        seam_err = abs(left[y, 2] - right_shifted[y, 0])

        total_normal += w * normal_err
        total_tangent += w * tangent_err
        total_seam += w * seam_err
        total_w += w

        used_rows.append(y)

        # keep one middle-row profile for visualization
        if debug_profiles is None and abs(y - k // 2) <= 2:
            debug_profiles = {
                "y": y,
                "ln": ln,
                "rn": rn,
                "lt": lt,
                "rt": rt,
                "nx": nx[y],
                "ny": ny[y],
            }

    if total_w < 1e-8:
        # no reliable edge rows
        return {
            "total": 1e9,
            "normal_term": 1e9,
            "tangent_term": 1e9,
            "seam_term": 1e9,
            "right_shifted": right_shifted,
            "patch6": patch6,
            "used_rows": used_rows,
            "debug_profiles": debug_profiles,
            "nx": nx,
            "ny": ny,
            "mag_row": mag_row,
        }

    normal_term = total_normal / total_w
    tangent_term = total_tangent / total_w
    seam_term = total_seam / total_w

    total = alpha * normal_term + beta * tangent_term + gamma * seam_term

    return {
        "total": total,
        "normal_term": normal_term,
        "tangent_term": tangent_term,
        "seam_term": seam_term,
        "right_shifted": right_shifted,
        "patch6": patch6,
        "used_rows": used_rows,
        "debug_profiles": debug_profiles,
        "nx": nx,
        "ny": ny,
        "mag_row": mag_row,
    }


def score_3way(left, right):
    out = {}
    for name, dy in CANDIDATES.items():
        out[name] = candidate_score(left, right, dy)
    return out


def decide_3way(left, right, threshold=0.005):
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


def true_label_from_true_dy(true_dy, tol=0.25):
    if true_dy < -tol:
        return "up"
    elif true_dy > tol:
        return "down"
    else:
        return "stop"


# =========================================================
# 4) visualization
# =========================================================
def visualize_cases(cases, threshold=0.005, save_path=None):
    fig, axes = plt.subplots(len(cases), 5, figsize=(20, 3.8 * len(cases)))
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

        best = scored[pred]
        patch_corr = best["patch6"]

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

        # panel 4: row-wise edge strength
        ax = axes[row, 3]
        ax.plot(best["mag_row"], label="edge strength")
        for yy in best["used_rows"]:
            ax.axvline(yy, color="gray", alpha=0.05)
        ax.set_title(
            f"gain_up={gain_up:.4f}, gain_down={gain_down:.4f}"
        )
        ax.set_xlabel("row")
        ax.legend(fontsize=8)

        # panel 5: normal/tangent profiles at one debug row
        ax = axes[row, 4]
        dbg = best["debug_profiles"]
        if dbg is not None:
            x1 = np.arange(len(dbg["ln"]))
            x2 = np.arange(len(dbg["lt"]))
            ax.plot(x1, dbg["ln"], marker="o", label="left normal")
            ax.plot(x1, dbg["rn"], marker="o", label="right normal")
            ax.plot(x2, dbg["lt"], marker="x", label="left tangent")
            ax.plot(x2, dbg["rt"], marker="x", label="right tangent")
            ax.set_title(
                f"profiles @ row {dbg['y']}\n"
                f"n=({dbg['nx']:.2f},{dbg['ny']:.2f})"
            )
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no strong edge row", ha="center", va="center")
            ax.set_title("profiles")
        ax.set_xlabel("sample index")

        summary.append({
            "title": title,
            "true_dy": true_dy,
            "gt_label": gt_label,
            "pred": pred,
            "scores": {n: scored[n]["total"] for n in ["up", "stop", "down"]},
            "gain_up": gain_up,
            "gain_down": gain_down,
            "normal_terms": {n: scored[n]["normal_term"] for n in ["up", "stop", "down"]},
            "tangent_terms": {n: scored[n]["tangent_term"] for n in ["up", "stop", "down"]},
            "seam_terms": {n: scored[n]["seam_term"] for n in ["up", "stop", "down"]},
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
            f"  total: "
            f"up={item['scores']['up']:.4f}, "
            f"stop={item['scores']['stop']:.4f}, "
            f"down={item['scores']['down']:.4f}"
        )
        print(
            f"  normal: "
            f"up={item['normal_terms']['up']:.4f}, "
            f"stop={item['normal_terms']['stop']:.4f}, "
            f"down={item['normal_terms']['down']:.4f}"
        )
        print(
            f"  tangent: "
            f"up={item['tangent_terms']['up']:.4f}, "
            f"stop={item['tangent_terms']['stop']:.4f}, "
            f"down={item['tangent_terms']['down']:.4f}"
        )
        print(
            f"  seam: "
            f"up={item['seam_terms']['up']:.4f}, "
            f"stop={item['seam_terms']['stop']:.4f}, "
            f"down={item['seam_terms']['down']:.4f}"
        )
        print(
            f"  gains: gain_up={item['gain_up']:.4f}, "
            f"gain_down={item['gain_down']:.4f}"
        )

    acc = correct / len(summary)
    print(f"\nAccuracy: {correct}/{len(summary)} = {acc:.4f}")


# =========================================================
# 5) run
# =========================================================
if __name__ == "__main__":
    cases = [
        ("true up 0.5px",   -0.5, 18),
        ("true stop 0.0px",  0.0, 18),
        ("true down 0.5px",  0.5, 18),
        ("true up 1.0px",   -1.0, 18),
        ("true down 1.0px",  1.0, 18),
        ("true up 0.5px, steeper",   -0.5, 32),
        ("true stop, steeper",        0.0, 32),
        ("true down 0.5px, steeper",  0.5, 32),
    ]

    visualize_cases(
        cases,
        threshold=0.005,
        save_path="left_boundary_orientation_interp_test.png"
    )
