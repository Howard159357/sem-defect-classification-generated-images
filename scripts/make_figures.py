"""Figures of the report, drawn from the one-shot test results (public AnomalyGen v1.0.0, commit fcad44cd).

  python scripts/make_figures.py                      the charts and diagrams (fig00, 01, 02, 06-12) from results/;
                                                      everything they need is in this repo
  python scripts/make_figures.py --private-root DIR --generated-dir SUB --test-dir SUB
                                                      also: refresh results/per_chain_recall.json from the per-image
                                                      predictions and draw the image panels fig03, fig04, fig05. DIR
                                                      is the private working tree with the dataset; the two SUB
                                                      folders (relative to DIR) hold the contact sheets of the
                                                      generated images and the prediction files of the test run.
                                                      None of these are redistributed here.

Inputs (results/):
  v2_stats.json          statistics of the single test run
  design.json            design facts and design-stage measurements for figures 2, 5 and 12 (not test results)
  per_chain_recall.json  recalls per chain, derived from the per-image predictions with --private-root and checked
                         against v2_stats.json on every run
Outputs (figures/): fig00 (headline) and fig01..fig12 in the order they appear in README.md and DETAILS.md, each as *_light.png and
*_dark.png (the pages pick one by colour scheme). All figures are 10 in wide at 180 dpi (2x the README column) and saved as
palette PNGs.

Rules followed in every chart: one y axis; colour follows the kind of training data everywhere (grey = real images
only, blue = pasted, green = rotated paste, orange = generated), with marker shape as a second cue; direct labels or
a legend; no status colours. The image panels (fig03, fig04, fig05) only ever read TRAINING images (class-3 chains
s0-s2, first 4 images; the first defect-free training image) and the synthetic images made from training images;
validation and test images are never opened by this script.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
DATASET = "Carinthia-S by C. Kofler and V. Hasić (CC BY 4.0), https://zenodo.org/records/16895427"
W, DPI = 10.0, 180
SRC = {}     # "generated", "test": private folders, set from the command line

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", fg2="#59636e", grid="#d8dee4", paste="#2a78d6", gen="#eb6834",
                  rot="#1baf7a", neutral="#6b737b", band=0.18, seq=("#f1f5fb", "#1c5cab")),
    "dark": dict(bg="#0d1117", fg="#f0f6fc", fg2="#9aa4af", grid="#30363d", paste="#3987e5", gen="#d95926",
                 rot="#199e70", neutral="#9aa2aa", band=0.28, seq=("#151b23", "#6da7ec")),
}
ALGS = [("dinov2", "DINOv2-L + LR (frozen)", "1 fit per chain"),
        ("resnet18", "ResNet-18 (fine-tuned)", "6 seeds per chain"),
        ("convnext_tiny", "ConvNeXt-T (fine-tuned)", "6 seeds per chain")]
ARMS = ["R", "P", "P_box", "G"]
ARM_TEXT = {"R": "R: 4 real images only", "P": "P: + pasted copies", "P_box": "P_box: + pasted with 24 px surround",
            "P_rot": "P_rot: + rotated paste", "G": "G: + generated images"}
MARK = {"R": "o", "P": "v", "P_box": "s", "P_rot": "^", "G": "D"}


def arm_col(t, arm):
    return {"R": t["neutral"], "P": t["paste"], "P_box": t["paste"], "P_rot": t["rot"], "G": t["gen"]}[arm]


def sgn(x):
    """+0.039 with a typographic minus."""
    return f"{x:+.3f}".replace("-", "−")


def stats():
    return json.loads((RES / "v2_stats.json").read_text(encoding="utf-8"))


def style(t):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "text.color": t["fg"], "axes.labelcolor": t["fg2"], "axes.edgecolor": t["grid"],
        "xtick.color": t["fg2"], "ytick.color": t["fg2"], "xtick.labelcolor": t["fg2"], "ytick.labelcolor": t["fg2"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    })


def head(fig, t, title, sub):
    """Title and subtitle, wrapped to the figure width; spacing in inches so it does not grow with the height."""
    h = fig.get_figheight()
    title = textwrap.fill(title, 74)
    sub = textwrap.fill(sub, 116)
    y = 1 - 0.20 / h
    fig.text(0.02, y, title, fontsize=13, fontweight="bold", color=t["fg"], va="top", linespacing=1.25)
    y -= (0.275 * (title.count("\n") + 1) + 0.08) / h
    fig.text(0.02, y, sub, fontsize=9.8, color=t["fg2"], va="top", linespacing=1.35)


def foot(fig, t, text):
    text = "\n".join(textwrap.fill(line, 124) for line in text.split("\n"))
    fig.text(0.02, 0.014, text, fontsize=9, color=t["fg2"], va="bottom", linespacing=1.35)


def save(fig, name, mode):
    from PIL import Image
    FIG.mkdir(exist_ok=True)
    p = FIG / f"{name}_{mode}.png"
    fig.savefig(p, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    with Image.open(p) as im:
        small = im.convert("RGB").quantize(256, dither=Image.Dither.NONE)
    small.save(p, optimize=True)


def spread(vals, gap):
    """Label positions next to vals, pushed apart to at least gap; the order of the values is kept."""
    out = np.array(vals, dtype=float)
    order = np.argsort(out, kind="stable")
    for a, b in zip(order[:-1], order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out - (out.mean() - np.mean(vals))


def frame(ax, t):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(t["grid"])


def design():
    """Design facts and design-stage measurements (results/design.json); not results of the test run."""
    return json.loads((RES / "design.json").read_text(encoding="utf-8"))


def per_chain_doc():
    return json.loads((RES / "per_chain_recall.json").read_text(encoding="utf-8"))


def check_per_chain(S):
    """The per-chain file must reproduce the published means."""
    doc = per_chain_doc()
    for k, v in doc["class3_recall"].items():
        assert abs(np.mean(v) - S["absolute"][k]["3"]) < 1e-9, k
        assert abs(np.mean(doc["class4_recall"][k]) - S["absolute"][k]["4"]) < 1e-9, k
    for n in ("n4", "n10"):
        q = doc["primary"][n]
        assert abs(np.mean(q["G*"]) - S["primary"][n]["recA"]) < 1e-9, n
        assert abs(np.mean(q["R"]) - S["primary"][n]["recB"]) < 1e-9, n
        assert abs(np.mean(q["P*"]) - S["H-v2b"][n]["recB"]) < 1e-9, n
    return doc


# ------------------------------------------------------------------------------------------------ fig00

def fig00(S, t, mode):
    """Headline figure for the top of the README: class-3 errors out of all test predictions at n = 4, per training
    set at the setting chosen on validation data. Errors are counted from the confusion matrices where they exist
    and from the per-chain recalls otherwise; both must agree."""
    doc = check_per_chain(S)
    q = doc["primary"]["n4"]
    cm = {a: np.array(S["confusion"][f"dinov2|{a}|n4|primary"]) for a in ("R", "G*", "P*")}
    per_chain = cm["R"][0].sum() // 3                                  # class-3 test images per chain
    err = {a: int(round(sum((1 - v) * per_chain for v in q[a]))) for a in ("R", "P*", "P_rot*", "G*")}
    for a, m in cm.items():
        assert err[a] == m[0, 1] + m[0, 2], a
    total = 3 * per_chain
    rows = [("R", "Real images only", t["neutral"], False),
            ("P*", "+ pasted copies of the same defects", t["paste"], False),
            ("P_rot*", "+ rotated pasted copies", t["rot"], True),
            ("G*", "+ generated images (AnomalyGen)", t["gen"], False)]
    fig = plt.figure(figsize=(W, 4.6))
    ax = fig.add_axes([0.345, 0.17, 0.43, 0.56])
    for i, (a, name, col, explo) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, err[a], height=0.62, color=t["bg"] if explo else col, edgecolor=col, linewidth=1.6,
                hatch="////" if explo else None, zorder=3)
        ax.text(err[a] + 2.5, y, f"{err[a]}", va="center", fontsize=14, fontweight="bold", color=t["fg"])
        ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11.5,
                color=t["fg"], fontweight="bold" if a in ("R", "G*") else "normal")
        if explo:
            ax.text(-0.02, y - 0.33, "exploratory comparison", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=9.5, color=t["fg2"], style="italic")
    p = S["primary"]["n4"]
    gain = err["R"] - err["G*"]
    ax.annotate("", xy=(err["G*"], -0.45), xytext=(err["R"], -0.45), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=t["gen"], lw=1.8, shrinkA=0, shrinkB=0))
    fig.text(0.79, 0.43, f"{gain} fewer errors\n({gain / err['R']:.0%} fewer)", fontsize=13, fontweight="bold",
             color=t["gen"], va="center")
    fig.text(0.79, 0.30, f"class-3 recall\n{p['recB']:.1%} → {p['recA']:.1%}", fontsize=11, color=t["fg"],
             va="center", linespacing=1.35)
    ax.set_xlim(0, max(err.values()) * 1.12)
    ax.set_ylim(-0.75, len(rows) - 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(f"Class-3 test images classified wrongly, out of {total:,} (fewer is better)", fontsize=10.5)
    head(fig, t, f"With 4 real images of a defect class, generated images cut class-3 errors from {err['R']} to "
                 f"{err['G*']}",
         f"DINOv2 + logistic regression. {per_chain} held-out test images × 3 sets of 4 real images. Pre-registered "
         "comparison against real images only and against plain paste; rotated paste is exploratory.")
    save(fig, "fig00_hero", mode)


# ------------------------------------------------------------------------------------------------ fig01

def fig01(S, t, mode):
    """The pre-registered result in absolute numbers: class-3 recall per chain, real images only -> + generated."""
    doc = check_per_chain(S)
    fig = plt.figure(figsize=(W, 5.6))
    panels = [("n4", "n = 4 real images: pre-registered hypothesis test"),
              ("n10", "n = 10 real images: described only")]
    for i, (n, name) in enumerate(panels):
        ax = fig.add_axes([0.085 + i * 0.49, 0.215, 0.385, 0.42])
        r, g = doc["primary"][n]["R"], doc["primary"][n]["G*"]
        x = S["primary"][n]
        for k in range(3):
            ax.plot([0, 1], [r[k], g[k]], color=t["fg2"], lw=1.0, alpha=0.7, zorder=2)
        ax.plot([-0.13, 0.13], [np.mean(r)] * 2, color=t["neutral"], lw=3.4, solid_capstyle="round", zorder=3)
        ax.plot([0.87, 1.13], [np.mean(g)] * 2, color=t["gen"], lw=3.4, solid_capstyle="round", zorder=3)
        ax.scatter([0] * 3, r, s=52, marker=MARK["R"], facecolor=t["bg"], edgecolor=t["neutral"], linewidth=1.7, zorder=4)
        ax.scatter([1] * 3, g, s=52, marker=MARK["G"], facecolor=t["bg"], edgecolor=t["gen"], linewidth=1.7, zorder=4)
        # chain labels in muted ink, the mean in bold beside its bar; all four are spread so none is misread
        for vals, lx, ha, side in ((r, -0.17, "right", "R"), (g, 1.17, "left", "G")):
            pos = spread(list(vals) + [np.mean(vals)], 0.0118)
            for k in range(3):
                txt = f"s{k}  {vals[k]:.3f}" if side == "R" else f"{vals[k]:.3f}  s{k}"
                ax.text(lx, pos[k], txt, ha=ha, va="center", fontsize=9.3, color=t["fg2"])
            ax.text(lx, pos[3], f"mean {np.mean(vals):.3f}" if side == "R" else f"{np.mean(vals):.3f} mean",
                    ha=ha, va="center", fontsize=10.5, fontweight="bold",
                    color=t["fg"])
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(0.84, 1.005)
        ax.set_yticks([0.85, 0.90, 0.95, 1.00])
        ax.set_xticks([0, 1], ["R\nreal images only", "G*\n+ generated images"])
        ax.tick_params(axis="x", length=0, pad=6, labelcolor=t["fg"])
        ax.tick_params(axis="y", length=0)
        ax.yaxis.grid(True, color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_ylabel("Class-3 recall (300 held-out test images)")
        verdict = ("supported (single hypothesis test; guardrails passed)" if x["final"] == "supported"
                   else f"{x['final']} (set in advance)")
        ax.text(0, 1.215, name, transform=ax.transAxes, fontsize=11, fontweight="bold", color=t["fg"], va="bottom")
        ax.text(0, 1.045, f"mean {x['recB']:.3f} → {x['recA']:.3f}:  {sgn(x['delta'])}  "
                f"[{sgn(x['lo'])}, {sgn(x['hi'])}]\n{verdict}", transform=ax.transAxes, fontsize=9.8, color=t["fg"],
                va="bottom", linespacing=1.4)
    p = S["primary"]["n4"]
    head(fig, t, f"With 4 real images of the scarce class, class-3 recall rose from {p['recB']:.3f} to "
                 f"{p['recA']:.3f} after generated images were added",
         "DINOv2-L + logistic regression. One line per chain (s0, s1, s2); thick bars: mean. Square brackets: 95% "
         "interval of the difference. G*: number and weight of generated images chosen on validation data.")
    foot(fig, t, "Supported = mean difference ≥ +0.02, the hypothesis test against 0 passes, and class-4 and good recall do not "
                 "drop by more than 0.02 (rules set in advance). Source: results/per_chain_recall.json, "
                 "results/v2_stats.json.")
    save(fig, "fig01_primary", mode)


# ------------------------------------------------------------------------------------------------ fig02

def fig02(t, mode):
    """Design overview: what is fixed first, what each arm trains on, what is measured."""
    fig = plt.figure(figsize=(W, 5.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=t["bg"], edgecolor=col, linewidth=1.6, zorder=2))
        ax.add_patch(plt.Rectangle((x, y + h - 0.08), w, 0.08, facecolor=col, edgecolor="none", zorder=3))
        ax.text(x + 0.14, y + h - 0.24, title, fontsize=11, fontweight="bold", color=t["fg"], va="top", zorder=4)
        ax.text(x + 0.14, y + h - 0.66, body, fontsize=10, color=t["fg"], va="top", linespacing=1.42, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1.4, shrinkA=0, shrinkB=0))

    g = t["fg2"]
    d = design()
    di, sp, ge = d["dataset_images"], d["split"], d["generation"]
    trio = lambda x: " / ".join(f"{x[c]:,}" for c in ("class 3", "class 4", "good"))  # noqa: E731
    box(0.05, 2.95, 2.75, 2.2, "1  Dataset", f"Carinthia-S wafer SEM\nclass 3: {di['class 3']:,} images\n"
        f"class 4: {di['class 4']:,}, good: {di['good']:,}\n128 px crops on the defect", g)
    box(0.05, 0.15, 2.75, 2.4, "2  Split, fixed first", f"class 3: {sp['chains']} disjoint chains\n"
        f"of n = {sp['chain_sizes'][0]} … {sp['chain_sizes'][1]} train images\n"
        f"class 4, good: {sp['train']['class 4']} train each\nvalidation {trio(sp['validation'])}\n"
        f"test {trio(sp['test'])} (locked)", g)
    box(3.15, 2.95, 2.75, 2.2, "3  Generator (cloud)", f"one per (n, chain): {ge['generators']}\n"
        f"sees n real class-3 images\n+ {ge['backgrounds']} good backgrounds\n"
        f"{ge['steps']:,} steps → {ge['images_per_generator']} images", t["gen"])
    box(3.15, 0.15, 2.75, 2.4, "4  Paste controls", f"same n defects, cut out,\npasted on the same {ge['backgrounds']}\n"
        "backgrounds: P, P_box\n(+ real surround), P_rot\n(pixels rotated)", t["paste"])
    box(6.25, 1.35, 2.6, 2.6, "5  Class-3 training set", "R   real images only\nG   R + generated\nP   R + pasted\n"
        "(P_box, P_rot likewise)\nclass 4, good: unchanged", g)
    box(9.2, 2.95, 2.75, 2.2, "6  Classifiers (laptop)", "DINOv2-L, frozen, +\nlogistic regression\n"
        "ResNet-18, ConvNeXt-T,\nfine-tuned", g)
    box(9.2, 0.15, 2.75, 2.4, "7  Measured", "validation: pick number\nand weight of synthetic\nimages; final test, once:\n"
        "class-3 recall; class 4 and\ngood must not drop > 0.02", g)
    arrow(1.425, 2.95, 1.425, 2.55)
    arrow(2.8, 1.35, 3.15, 1.35)
    arrow(2.8, 2.2, 3.15, 3.3)
    arrow(5.9, 3.7, 6.25, 3.2)
    arrow(5.9, 1.5, 6.25, 2.1)
    arrow(8.85, 3.2, 9.2, 3.7)
    arrow(10.575, 2.95, 10.575, 2.55)
    head(fig, t, "Design: one scarce class, one generator per chain, paste controls",
         "Colour as in every chart: orange = generated, blue = pasted. Validation and test images never enter "
         "training or generation, and test images never enter the choice of settings; a script checks this.")
    save(fig, "fig02_design", mode)


# ------------------------------------------------------------------------------------------------ image helpers

def crop(img, cy, cx, size=128):
    h, w = img.shape
    y0, x0 = min(max(cy - size // 2, 0), h - size), min(max(cx - size // 2, 0), w - size)
    return img[y0:y0 + size, x0:x0 + size]


def train_rows(private, role, n):
    if not (role.startswith("c3_chain") or role == "good_train"):
        raise ValueError(f"{role}: the image panels may show training images only")
    with open(private / "downstream" / "v2" / "splits_v2.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == role]
    return sorted(rows, key=lambda r: int(r["order"]))[:n]


def defect_centre(private, row):
    from PIL import Image
    p = private / "carinthia_s_upload" / row["path"]
    mp = private / "carinthia_s_upload" / row["path"].replace("/anomaly_image/", "/mask/")
    ys, xs = np.nonzero(np.asarray(Image.open(mp.with_name(mp.stem + "_mask.png")).convert("L")) >= 128)
    return np.asarray(Image.open(p).convert("L")), int(round(ys.mean())), int(round(xs.mean()))


def real_crops(private, chain, n=4):
    """128 px crops of the first n real class-3 TRAINING images of a chain."""
    return [crop(*defect_centre(private, r)) for r in train_rows(private, f"c3_chain{chain}", n)]


def sheet(path, n):
    """Contact sheet: 192 px tiles, 8 per row, in generation order; the centre 128 px is the 128 px crop."""
    from PIL import Image
    if n > 8:
        raise ValueError("only the first row of a contact sheet (8 tiles) is read")
    s = np.asarray(Image.open(path).convert("L"))
    return [s[32:160, k * 192 + 32:k * 192 + 160] for k in range(n)]


# ------------------------------------------------------------------------------------------------ fig03

def fig03(t, mode, private):
    """4 x 4 crops: the 4 real training defects of chain s0 and the first 4 synthetic images of each kind
    (synthetic image t uses real defect t mod 4, so every column is one defect)."""
    v2 = private / "downstream" / "v2"
    rows = [("Real\n(the 4 training images)", real_crops(private, 0)),
            ("P: cut-and-paste\n(pixels not rotated)", sheet(v2 / "paste" / "sheet_n4.jpg", 4)),
            ("P_rot: cut-and-paste\n(rotated)", sheet(v2 / "paste_rot" / "sheet_n4.jpg", 4)),
            ("G: generated\n(new pixels; only the\nmask shape is reused)",
             sheet(SRC["generated"] / "sheet_n4_s0.jpg", 4))]
    hgt = 9.6
    fig = plt.figure(figsize=(W, hgt))
    w = 0.178
    h = w * W / hgt
    x0, y0 = 0.25, 0.085
    for r, (name, imgs) in enumerate(rows):
        yy = y0 + (3 - r) * (h + 0.011)
        fig.text(x0 - 0.015, yy + h / 2, name, ha="right", va="center", fontsize=10.5, color=t["fg"], linespacing=1.4)
        for c, im in enumerate(imgs):
            ax = fig.add_axes([x0 + c * (w + 0.008), yy, w, h])
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
            frame(ax, t)
            if r == 0:
                ax.set_title(f"Defect {c + 1}", fontsize=10, color=t["fg2"], pad=5)
    head(fig, t, "What the classifier is trained on: real, pasted and generated crops",
         "Chain s0, n = 4: the 4 real SEM images of the scarce class and the first 4 synthetic images of each kind; "
         "128 px crops. Each column starts from the same real defect.")
    foot(fig, t, f"Images: {DATASET}; cropped. Rows 2 to 4 are derived images: the defect is pasted or generated "
                 "onto defect-free images of the same dataset. Training images only; no validation or test image is "
                 "shown.")
    save(fig, "fig03_arms", mode)


# ------------------------------------------------------------------------------------------------ fig04

def fig04(t, mode, private):
    """Each generator next to the real defects it was fine-tuned on: first 4 real training crops of the chain and
    the generator's first 6 images (128 px crops, generation order)."""
    d = SRC["generated"]
    hgt = 7.3
    fig = plt.figure(figsize=(W, hgt))
    tile, gap = 0.78, 0.045                 # inches
    xr, xg = 1.2, 1.2 + 4 * (tile + gap) + 0.2
    ytop = hgt - 1.42
    for r, (n, k) in enumerate((n, k) for n in (4, 10) for k in (0, 1, 2)):
        y = ytop - (r + 1) * (tile + gap) - (0.2 if n == 10 else 0)
        fig.text((xr - 0.12) / W, (y + tile / 2) / hgt, f"n = {n}, s{k}", ha="right", va="center", fontsize=10.5,
                 color=t["fg"])
        imgs = [(xr + c * (tile + gap), im) for c, im in enumerate(real_crops(private, k))] + \
               [(xg + c * (tile + gap), im) for c, im in enumerate(sheet(d / f"sheet_n{n}_s{k}.jpg", 6))]
        for x, im in imgs:
            ax = fig.add_axes([x / W, y / hgt, tile / W, tile / hgt])
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
            frame(ax, t)
    ylab = (ytop - gap + 0.06) / hgt
    fig.text(xr / W, ylab, "Real training images (first 4)", fontsize=10, color=t["fg"],
             fontweight="bold", va="bottom")
    fig.text(xg / W, ylab, "Generated images (first 6 of 400)", fontsize=10, color=t["fg"], fontweight="bold",
             va="bottom")
    head(fig, t, "Generated class-3 images next to the real defects each generator saw",
         "AnomalyGen v1.0.0, step 12,000. Each row is one generator, fine-tuned on n real class-3 images of one "
         "chain; 128 px crops. Mask shapes come from that chain's real defects, scaled and rotated.")
    foot(fig, t, f"Images: {DATASET}; cropped. The generated images are derived images: defects generated onto "
                 "defect-free training images of the same dataset. Training images only; no validation or test "
                 "image is shown.")
    save(fig, "fig04_generated", mode)


# ------------------------------------------------------------------------------------------------ fig05

def fig05(t, mode, private):
    """Why the task uses defect-centred crops: one class-3 and one good TRAINING image with the mean grey level of
    each row (background columns only), and the design-stage diagnosis (real images only; not results of this
    study)."""
    from PIL import Image
    dg = design()["design_stage_diagnosis"]
    diagnosis, top_share = dg["class3_recall"], dg["share_brighter_at_top"]
    c3 = train_rows(private, "c3_chain0", 1)[0]
    good = train_rows(private, "good_train", 1)[0]
    a, cy, cx = defect_centre(private, c3)
    b = np.asarray(Image.open(private / "carinthia_s_upload" / good["path"]).convert("L"))
    y0, x0 = min(max(cy - 64, 0), a.shape[0] - 128), min(max(cx - 64, 0), a.shape[1] - 128)
    hgt = 5.4
    fig = plt.figure(figsize=(W, hgt))
    w = 0.205
    h = w * W / hgt * a.shape[0] / a.shape[1]
    yb = 0.30
    stroke = [pe.withStroke(linewidth=3, foreground="#0d1117")]
    for i, (im, name) in enumerate(((a, "Class 3 (training image)"), (b, "Good (training image)"))):
        xl = 0.02 + i * 0.315
        ax = fig.add_axes([xl, yb, w, h])
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        frame(ax, t)
        ax.set_title(name, fontsize=10.5, color=t["fg"], pad=6, loc="left")
        if i == 0:
            ax.add_patch(plt.Rectangle((x0, y0), 128, 128, fill=False, edgecolor="#ffffff", linewidth=1.8,
                                       path_effects=stroke))
            ax.text(x0 + 64, y0 - 9, "128 px crop", ha="center", va="bottom", fontsize=9.8, color="#ffffff",
                    fontweight="bold", path_effects=stroke)
        bg = np.concatenate([im[:, :100], im[:, -100:]], axis=1).mean(axis=1)       # background columns only
        bg = np.convolve(np.pad(bg[10:-10], 12, mode="edge"), np.ones(25) / 25, mode="valid")
        px = fig.add_axes([xl + w + 0.008, yb, 0.062, h])
        px.plot(bg, np.arange(len(bg)) + 10, color=t["fg"], lw=1.6)
        px.set_ylim(im.shape[0] - 1, 0)
        px.set_xlim(84, 112)
        px.set_xticks([90, 105])
        px.set_yticks([])
        px.tick_params(axis="x", length=0, labelsize=9, pad=2)
        px.xaxis.grid(True, color=t["grid"], lw=0.8)
        px.spines["left"].set_visible(False)
        px.set_title("row mean", fontsize=9.5, color=t["fg2"], pad=6)
        top, bot = im[:40].mean(), im[-40:].mean()
        fig.text(xl, yb - 0.07, f"brighter at the {'top' if top > bot else 'bottom'}: grey level\n"
                 f"{top:.0f} at the top edge, {bot:.0f} at the bottom", fontsize=9.8, color=t["fg"], va="top",
                 linespacing=1.35)
    ax = fig.add_axes([0.79, yb, 0.185, h])
    ys, labels = [], []
    for gi, (grp, (vis, hid)) in enumerate(diagnosis.items()):
        base = 3.0 - gi * 2.6
        ax.barh([base, base - 0.8], [vis, hid], height=0.62, color=t["neutral"], zorder=3)
        ax.patches[-1].set_alpha(0.45)
        for y, v in ((base, vis), (base - 0.8, hid)):
            ax.text(v + 0.03, y, f"{v:.3f}", va="center", fontsize=10.5, color=t["fg"])
        ax.text(-0.02, base + 0.62, grp, fontsize=10.5, fontweight="bold", color=t["fg"], va="center", ha="right",
                transform=ax.get_yaxis_transform())
        ys += [base, base - 0.8]
        labels += ["defect visible", "defect hidden"]
    ax.set_yticks(ys, labels, fontsize=10)
    ax.tick_params(axis="y", length=0, labelcolor=t["fg"])
    ax.set_ylim(-1.0, 4.1)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.5, 1.0])
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Class-3 recall\n(4 real images)")
    head(fig, t, "Why defect-centred crops: on whole images the background alone gives the class away",
         f"Class-3 images are mostly brighter at the top edge ({top_share['class 3']:.0%}), good images rarely "
         f"({top_share['good']:.0%}). A classifier that sees only the background of a whole image still reaches "
         f"{diagnosis['Whole image'][1]:.3f}; on a crop without the defect it drops to {diagnosis['128 px crop'][1]:.3f}.")
    foot(fig, t, f"Images: {DATASET}; training images only. Row mean: mean grey level of each image row over the "
                 "left and right 100 columns (10 edge rows left out); the edge values under the images are means of the top or "
                 "bottom 40 rows. Bars: design-stage diagnosis with DINOv2-L + LR on the first-round "
                 "split; not a result of this study.")
    save(fig, "fig05_why_crop", mode)


# ------------------------------------------------------------------------------------------------ fig06

def fig06(S, t, mode):
    E = S["explore"]
    groups = [
        ("Primary (registered; tested at n = 4)",
         [("G* − R,  n = 4", S["primary"]["n4"], True), ("G* − R,  n = 10", S["primary"]["n10"], False)]),
        ("vs. paste, not rotated (registered)",
         [("G* − P*,  n = 4", S["H-v2b"]["n4"], True), ("G* − P*,  n = 10", S["H-v2b"]["n10"], False)]),
        ("vs. rotated paste (exploratory)",
         [("G* − P_rot*,  n = 4", E["dinov2|G* - P_rot*|n4"], False),
          ("G* − P_rot*,  n = 10", E["dinov2|G* - P_rot*|n10"], False)]),
    ]
    fig = plt.figure(figsize=(W, 5.7))
    ax = fig.add_axes([0.375, 0.225, 0.36, 0.515])
    lab = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0
    for title, rows in groups:
        fig.text(0.02, y, title, transform=lab, fontsize=10, fontweight="bold", color=t["fg"], va="center")
        y += 1.0
        for name, r, tested in rows:
            ax.plot([r["lo"], r["hi"]], [y, y], color=t["gen"], lw=2.2, solid_capstyle="round", zorder=3)
            ax.scatter(r["chains"], [y + 0.32] * len(r["chains"]), s=34, marker="o", facecolor=t["bg"],
                       edgecolor=t["fg2"], linewidth=1.2, zorder=3)
            ax.scatter([r["delta"]], [y], s=74, marker="D", zorder=4, linewidth=1.8, edgecolor=t["gen"],
                       facecolor=t["gen"] if tested else t["bg"])
            fig.text(0.36, y, name, transform=lab, ha="right", va="center", fontsize=10.5, color=t["fg"])
            fig.text(0.765, y - 0.06, f"{sgn(r['delta'])}  [{sgn(r['lo'])}, {sgn(r['hi'])}]", transform=lab,
                     va="center", fontsize=10.5, color=t["fg"])
            if tested:
                v = r["judgement"]
                why = "; ".join(r.get("why", []))
                v = f"{v}: {why}" if why else f"{v} (hypothesis test)"
            elif "judgement" in r:
                v = "described only (set in advance)"
            else:
                v = "exploratory, not tested"
            fig.text(0.765, y + 0.38, v, transform=lab, va="center", fontsize=9.3, color=t["fg2"])
            y += 1.0
        y += 0.35
    ax.set_ylim(y - 0.55, -0.55)
    ax.set_yticks([])
    ax.axvline(0, color=t["fg2"], lw=1.1, zorder=1)
    ax.axvline(0.02, color=t["fg2"], lw=1.1, ls=(0, (5, 4)), zorder=1)
    top = ax.get_xaxis_transform()
    ax.text(0, 1.01, "no difference ", transform=top, ha="right", va="bottom", fontsize=9.3, color=t["fg2"])
    ax.text(0.02, 1.01, " +0.02 margin, set in advance", transform=top, ha="left", va="bottom", fontsize=9.3,
            color=t["fg2"])
    ax.set_xlim(-0.03, 0.075)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06], ["−0.02", "0", "+0.02", "+0.04", "+0.06"])
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Difference in class-3 recall (mean of 3 chains)")
    fig.text(0.765, 1.012, "Δ  [95% interval]", fontsize=9.3, color=t["fg2"], va="bottom",
             transform=transforms.blended_transform_factory(fig.transFigure, ax.transAxes))
    a, b = S["primary"]["n4"], S["H-v2b"]["n4"]
    assert a["final"] == "supported" and b["judgement"] == "supported"
    head(fig, t, f"Pre-registered test, 4 real images: with generated images, class-3 recall was {sgn(a['delta'])} "
                 f"above real images only and {sgn(b['delta'])} above paste",
         "DINOv2-L + LR, 300 held-out test images. G*, P*, P_rot*: number and weight of synthetic images chosen per chain "
         "on validation data only.")
    foot(fig, t, "Diamond: mean difference (filled = hypothesis test, open = not tested). Bar: 95% interval, paired "
                 "bootstrap over chains and images. Small circles: the 3 chains. Supported = mean difference ≥ +0.02 "
                 "and the hypothesis test against 0 passes; the primary row also needs the guardrails (rules set in advance). "
                 "P_rot is the paste set as first registered. "
                 "Source: results/v2_stats.json.")
    save(fig, "fig06_forest", mode)


# ------------------------------------------------------------------------------------------------ fig07

def fig07(S, t, mode):
    pc = check_per_chain(S)["class3_recall"]
    ref = S["absolute"]["dinov2|R"]["3"]
    assert S["absolute"]["resnet18|G"]["3"] < ref and S["absolute"]["convnext_tiny|G"]["3"] < ref
    fig = plt.figure(figsize=(W, 6.0))
    ax = fig.add_axes([0.075, 0.275, 0.775, 0.475])
    xt, xl = [], []
    for i, (alg, name, runs) in enumerate(ALGS):
        for j, arm in enumerate(ARMS):
            x, v, col = i * 5 + j, pc[f"{alg}|{arm}"], arm_col(t, arm)
            m = S["absolute"][f"{alg}|{arm}"]["3"]
            ax.plot([x - 0.36, x + 0.36], [m, m], color=col, lw=3.2, solid_capstyle="round", zorder=3)
            ax.scatter(x + np.array([-0.2, 0, 0.2]), v, s=48, marker=MARK[arm], facecolor=t["bg"], edgecolor=col,
                       linewidth=1.5, zorder=4)
            ax.text(x, 1.035, f"{m:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10,
                    color=t["fg"], fontweight="bold" if arm == "G" else "normal")
            xt.append(x)
            xl.append(arm)
        ax.text(i * 5 + 1.5, -0.125, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.5,
                color=t["fg"], fontweight="bold")
        ax.text(i * 5 + 1.5, -0.195, runs, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5,
                color=t["fg2"])
    ax.text(-0.012, 1.035, "mean", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=t["fg2"])
    ax.axhline(ref, color=t["fg2"], lw=1.1, ls=(0, (5, 4)), zorder=1)
    ax.text(1.012, ref, f"{ref:.3f}\nDINOv2-L + LR,\nreal images only", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9.5, color=t["fg"], linespacing=1.35)
    ax.set_xticks(xt, xl)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.8, 13.8)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.001, 0.1))
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Class-3 recall (300 held-out test images)")
    hd = [Line2D([], [], marker=MARK[a], ls="", markersize=7.5, markerfacecolor=t["bg"],
                 markeredgecolor=arm_col(t, a), markeredgewidth=1.5, label=ARM_TEXT[a]) for a in ARMS]
    hd.append(Line2D([], [], color=t["fg2"], lw=3.2, solid_capstyle="round", label="bar: mean"))
    fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.012, 0.075), ncol=5, fontsize=9.3, handletextpad=0.6,
               columnspacing=1.0, labelcolor=t["fg2"])
    head(fig, t, "4 real images: CNNs with generated images stay below DINOv2 trained on real images alone",
         "Carinthia-S, defect-centred 128 px crops. Fixed setting: 200 synthetic images, weight 1 : 1. "
         "Points: one per chain, left to right s0, s1, s2 (CNNs: mean of 6 seeds).")
    foot(fig, t, "Secondary analysis: added after the validation results were seen, settings fixed before the test "
                 "run. Source: results/v2_stats.json, results/per_chain_recall.json.")
    save(fig, "fig07_classifiers", mode)


# ------------------------------------------------------------------------------------------------ fig08

def fig08(S, t, mode):
    """Class-3 gain against class-4 change (G − R, n = 4, fixed setting), per chain: the guardrail in one picture."""
    doc = check_per_chain(S)
    c3, c4 = doc["class3_recall"], doc["class4_recall"]
    fig = plt.figure(figsize=(W, 5.5))
    failed = []
    for i, (alg, name, _) in enumerate(ALGS):
        ax = fig.add_axes([0.085 + i * 0.31, 0.215, 0.255, 0.405])
        dx = np.array(c4[f"{alg}|G"]) - np.array(c4[f"{alg}|R"])
        dy = np.array(c3[f"{alg}|G"]) - np.array(c3[f"{alg}|R"])
        key = f"{alg}|H-A1"
        assert abs(dy.mean() - S["secondary"][key]["delta"]) < 1e-9, key
        assert abs(dx.mean() - S["secondary"][key]["guard"]["4"]["diff"]) < 1e-9, key
        ok = S["secondary"][key]["guard"]["pass"]
        if not ok:
            failed.append(name.split(" (")[0])
        ax.axvspan(-0.06, -0.02, color=t["neutral"], alpha=0.09, lw=0, zorder=0)
        ax.axvline(-0.02, color=t["fg2"], lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.axvline(0, color=t["fg2"], lw=1.1, zorder=1)
        ax.axhline(0, color=t["fg2"], lw=1.1, zorder=1)
        ax.scatter(dx, dy, s=52, marker=MARK["G"], facecolor=t["bg"], edgecolor=t["gen"], linewidth=1.6, zorder=3)
        ax.scatter([dx.mean()], [dy.mean()], s=120, marker=MARK["G"], facecolor=t["gen"], edgecolor=t["bg"],
                   linewidth=1.4, zorder=4)
        order = np.argsort(dy)
        off = {int(k): 3 for k in order}
        for lo, hi in zip(order[:-1], order[1:]):
            if dy[hi] - dy[lo] < 0.03 and abs(dx[hi] - dx[lo]) < 0.012:
                off[int(lo)], off[int(hi)] = -10, 6
        for k in range(3):
            left = dx[k] < dx.mean()            # label on the side away from the mean diamond
            ax.annotate(f"s{k}", (dx[k], dy[k]), xytext=(-8 if left else 8, off[k]), textcoords="offset points",
                        fontsize=10, color=t["fg2"], ha="right" if left else "left")
        ax.set_xlim(-0.06, 0.011)
        ax.set_ylim(-0.035, 0.25)
        ax.set_xticks([-0.04, -0.02, 0], ["−0.04", "−0.02", "0"])
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25], ["0", "+0.05", "+0.10", "+0.15", "+0.20", "+0.25"])
        ax.yaxis.grid(True, color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.text(0, 1.30, name, transform=ax.transAxes, fontsize=11, fontweight="bold", color=t["fg"], va="bottom")
        ax.text(0, 1.165, f"class 3 {sgn(dy.mean())},  class 4 {sgn(dx.mean())}", transform=ax.transAxes,
                fontsize=9.8, color=t["fg"], va="bottom")
        ax.text(0, 1.04, "✓ within the guardrail" if ok else "✕ guardrail failed", transform=ax.transAxes,
                fontsize=10.5, color=t["fg"], va="bottom", fontweight="bold")
        ax.set_xlabel("Change in class-4 recall")
        ax.text(-0.058, -0.018, "beyond −0.02", fontsize=9.3, color=t["fg2"], va="center")
        if i == 0:
            ax.set_ylabel("Change in class-3 recall")
    assert failed == ["ConvNeXt-T"], failed
    head(fig, t, "ConvNeXt-T pays for its class-3 gain with class-4 recall beyond the guardrail; DINOv2 and "
                 "ResNet-18 stay within it",
         "Change when 200 generated images are added to 4 real images (G − R). Open diamonds: the 3 chains (CNNs: "
         "mean of 6 seeds). Filled diamond: mean of the chains, which is what the pre-set guardrail judges.")
    foot(fig, t, "Secondary analysis (fixed setting, weight 1 : 1). Good-image recall did not change in any run. "
                 "Source: results/v2_stats.json, results/per_chain_recall.json.")
    save(fig, "fig08_tradeoff", mode)


# ------------------------------------------------------------------------------------------------ fig09

def fig09(S, t, mode):
    rc = S["explore"]["real_curve"]
    eq = S["explore"]["equivalent_real"]
    n_all = np.array(sorted(int(k) for k in rc))
    n = n_all[n_all <= 100]
    m = np.array([rc[str(k)][0] for k in n])
    sd = np.array([rc[str(k)][1] for k in n])
    fig = plt.figure(figsize=(W, 5.7))
    ax = fig.add_axes([0.085, 0.185, 0.89, 0.575])
    ax.fill_between(n, m - sd, np.minimum(m + sd, 1.0), color=t["neutral"], alpha=t["band"], lw=0, zorder=1)
    ax.plot(n, m, color=t["neutral"], lw=2, marker=MARK["R"], markersize=6.5, markerfacecolor=t["neutral"],
            markeredgecolor=t["bg"], markeredgewidth=1.2, zorder=3)
    lo = 0.82
    for key, n0 in (("n4", 4), ("n10", 10)):
        g, e = S["primary"][key]["recA"], eq[f"G*|{key}"]
        ax.plot([n0, e], [g, g], color=t["gen"], lw=1.3, ls=(0, (2, 2)), zorder=4)
        ax.plot([e, e], [lo, g], color=t["gen"], lw=1.3, ls=(0, (2, 2)), zorder=2)
        ax.scatter([e], [g], s=46, marker="o", facecolor=t["bg"], edgecolor=t["gen"], linewidth=1.6, zorder=5)
        ax.scatter([n0], [g], s=90, marker=MARK["G"], facecolor=t["gen"], edgecolor=t["bg"], linewidth=1.4, zorder=6)
        ax.text(n0, g + 0.006, f"{g:.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=t["fg"])
        ax.text(e * 1.035, lo + 0.004, f"≈ {e:.1f} real\nimages", ha="left", va="bottom", fontsize=10, color=t["fg"],
                linespacing=1.25)
    ax.set_xscale("log")
    ax.set_xticks(n, [str(k) for k in n])
    ax.minorticks_off()
    ax.set_xlim(3.3, 125)
    ax.set_ylim(lo, 1.004)
    ax.set_yticks(np.arange(0.82, 1.001, 0.02))
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Number of real class-3 training images (log scale)")
    ax.set_ylabel("Class-3 recall (300 held-out test images)")
    hd = [Line2D([], [], color=t["neutral"], lw=2, marker=MARK["R"], markersize=6.5, markeredgecolor=t["bg"],
                 label="Real images only: mean ± SD over chains"),
          Line2D([], [], color=t["gen"], ls="", marker=MARK["G"], markersize=8, markeredgecolor=t["bg"],
                 label="4 or 10 real images + generated images (G*, mean of 3 chains)"),
          Line2D([], [], color=t["gen"], lw=1.3, ls=(0, (2, 2)), marker="o", markersize=6, markerfacecolor=t["bg"],
                 markeredgecolor=t["gen"], label="Same recall on the real-only curve (point estimate)")]
    ax.legend(handles=hd, loc="lower right", fontsize=9.5, labelcolor=t["fg2"], bbox_to_anchor=(1.0, 0.17),
              frameon=True, facecolor=t["bg"], edgecolor="none", framealpha=1.0)
    rest = ", ".join(f"{rc[str(k)][0]:.3f} at {k}" for k in n_all[n_all > 100])
    head(fig, t, f"4 real images plus generated images do about as well as {eq['G*|n4']:.1f} real images; "
                 f"10 plus generated, about {eq['G*|n10']:.1f}",
         "DINOv2-L + LR on defect-centred crops. Real-only curve: 10 chains for n = 4 to 50, 5 sets for n = 100. "
         f"Beyond the plot the curve stays flat: {rest} real images.")
    foot(fig, t, "Exploratory, point estimates without intervals; read off a monotone fit of the real-only curve, "
                 "interpolated on log n. The band is the SD across chains, not an interval for the diamonds. "
                 "Source: results/v2_stats.json.")
    save(fig, "fig09_real_curve", mode)


# ------------------------------------------------------------------------------------------------ fig10

G_VALUES = [10, 25, 50, 100, 200, 400]


def fig10(S, t, mode):
    """Effect of the number of synthetic images (DINOv2, weight 1 : 1), by kind of synthetic image."""
    gc = S["explore"]["g_curve"]
    arms = [("G", "-"), ("P_rot", "-"), ("P_box", (0, (4, 2))), ("P", "-")]
    fig = plt.figure(figsize=(W, 5.6))
    for i, n in enumerate((4, 10)):
        ax = fig.add_axes([0.085 + i * 0.465, 0.265, 0.375, 0.43])
        vals = [gc[f"{arm}|n{n}|r1"] for arm, _ in arms]
        for (arm, ls), v in zip(arms, vals):
            col = arm_col(t, arm)
            ax.plot(G_VALUES, v, color=col, lw=2, ls=ls, marker=MARK[arm], markersize=8, markerfacecolor=t["bg"],
                    markeredgecolor=col, markeredgewidth=1.6, zorder=3)
        for (arm, _), v, y in zip(arms, vals, spread([v[-1] for v in vals], 0.0055)):
            ax.text(500, y, arm, color=t["fg"], fontsize=10, va="center", ha="left")
            ax.plot([416, 485], [v[-1], y], color=arm_col(t, arm), lw=0.9, clip_on=False, zorder=2)
        ax.axhline(0, color=t["fg2"], lw=1.1, zorder=1)
        ax.axhline(0.02, color=t["fg2"], lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(G_VALUES, [str(g) for g in G_VALUES])
        ax.minorticks_off()
        ax.set_xlim(8.5, 440)
        ax.set_ylim(-0.02, 0.052)
        ax.set_yticks([-0.02, 0, 0.02, 0.04], ["−0.02", "0", "+0.02", "+0.04"] if i == 0 else [""] * 4)
        ax.yaxis.grid(True, color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Synthetic class-3 images added (log scale)")
        ax.set_title(f"{n} real class-3 images", fontsize=11, fontweight="bold", color=t["fg"], pad=10, loc="left")
        if i == 0:
            ax.text(430, 0.0212, "+0.02 margin", fontsize=9.5, color=t["fg2"], va="bottom", ha="right")
            ax.set_ylabel("Change in class-3 recall vs. R")
    names = {"G": "G: generated", "P_rot": "P_rot: rotated paste", "P_box": "P_box: paste + real surround",
             "P": "P: paste, not rotated"}
    hd = [Line2D([], [], color=arm_col(t, a), lw=2, ls=ls, marker=MARK[a], markersize=8, markerfacecolor=t["bg"],
                 markeredgecolor=arm_col(t, a), markeredgewidth=1.6, label=names[a]) for a, ls in arms]
    fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.012, 0.085), ncol=4, fontsize=9.5, handlelength=2.6,
               columnspacing=1.4, labelcolor=t["fg2"])
    g4 = gc["G|n4|r1"]
    assert min(g4[1:]) > 0.035 and max(max(gc["P|n4|r1"]), max(gc["P_box|n4|r1"])) < 0.005
    head(fig, t, "4 real images: from 25 generated images on, recall was about +0.04 above real images only; "
                 "paste without rotation stayed at or just below 0",
         "DINOv2-L + LR, weight real : synthetic = 1 : 1, 300 held-out test images, mean of 3 chains. "
         "R = real images only.")
    foot(fig, t, "Exploratory, described only: no test, no interval. The pre-registered analysis uses the setting "
                 "chosen per chain on validation data (fig. 6). Source: results/v2_stats.json.")
    save(fig, "fig10_g_curve", mode)


# ------------------------------------------------------------------------------------------------ fig11

def fig11(S, t, mode):
    cms = [("R: 4 real images only", np.array(S["confusion"]["dinov2|R|n4|primary"])),
           ("G*: 4 real images + generated images", np.array(S["confusion"]["dinov2|G*|n4|primary"]))]
    cmap = LinearSegmentedColormap.from_list("seq", list(t["seq"]))
    hi_ink, lo_ink = ("#ffffff", "#0d1117") if mode == "light" else ("#0d1117", t["fg"])
    names = ["Class 3", "Class 4", "Good"]
    fig = plt.figure(figsize=(W, 5.2))
    for i, (title, cm) in enumerate(cms):
        ax = fig.add_axes([0.14 + i * 0.44, 0.215, 0.36, 0.405])
        share = cm / cm.sum(1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for r in range(3):
            for c in range(3):
                if cm[r, c] == 0:
                    ax.text(c, r, "0", ha="center", va="center", fontsize=12, color=t["fg2"])
                    continue
                ink = hi_ink if share[r, c] >= 0.5 else lo_ink
                ax.text(c, r - 0.07, f"{cm[r, c]:,}", ha="center", va="center", fontsize=15, color=ink,
                        fontweight="bold" if (r, c) == (0, 1) else "normal")
                ax.text(c, r + 0.25, f"{share[r, c]:.1%}", ha="center", va="center", fontsize=9.5, color=ink)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), [f"{nm}\n({cm[k].sum():,})" for k, nm in enumerate(names)] if i == 0 else [])
        ax.tick_params(length=0, labelsize=10.5)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color=t["bg"], lw=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11, fontweight="bold", color=t["fg"], pad=42)
        ax.text(0.5, 1.115, "Predicted", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                color=t["fg2"])
        if i == 0:
            ax.set_ylabel("True class (3 chains summed)")
    a, b = cms[0][1], cms[1][1]
    assert a[0, 2] == 0 and b[0, 2] == 0
    fig.text(0.5, 0.135, f"Class 4 called class 3: {a[1, 0]} → {b[1, 0]}.  Defects called good: "
             f"{a[0, 2] + a[1, 2]} → {b[0, 2] + b[1, 2]}.  Good called defect: {a[2, 0] + a[2, 1]} → "
             f"{b[2, 0] + b[2, 1]}.", ha="center", fontsize=10.5, color=t["fg"])
    head(fig, t, f"Every class-3 error is a class-3 defect called class 4: {a[0, 1]} → {b[0, 1]} of {a[0].sum()} "
                 "with generated images",
         f"DINOv2-L + LR with 4 real class-3 images. Counts summed over 3 chains ({a[0].sum() // 3} class-3, "
         f"{a[1].sum() // 3} class-4, {a[2].sum() // 3} good test images each); % of the true class.")
    foot(fig, t, "Defect images carry a framing square that good images lack, so the error-free good row is not "
                 "evidence about defect detection. Source: results/v2_stats.json.")
    save(fig, "fig11_confusion", mode)


# ------------------------------------------------------------------------------------------------ fig12

def fig12(t, mode):
    TIMELINE = design()["timeline"]
    phase = {"r1": "Round 1", "r2": "Round 2: redesigned, registered in advance", "test": "Final test, run once"}
    fig = plt.figure(figsize=(W, 4.9))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.77])
    ax.set_xlim(-0.62, len(TIMELINE) - 0.38)
    ax.set_ylim(-0.86, 1.0)
    ax.axis("off")
    ax.plot([-0.5, len(TIMELINE) - 0.5], [0, 0], color=t["grid"], lw=2, zorder=1)
    for i, (ph, when, what) in enumerate(TIMELINE):
        up = i % 2 == 0
        key = ph in ("stamp", "test")
        col = t["fg"] if key else t["neutral"]
        ax.plot([i, i], [0, 0.13 if up else -0.13], color=col, lw=1.3, zorder=2)
        ax.scatter([i], [0], s=190 if key else 130, marker={"stamp": "s", "test": "D"}.get(ph, "o"), facecolor=col,
                   edgecolor=t["bg"], linewidth=2, zorder=3)
        body = textwrap.fill(what, 24)
        if up:
            ax.text(i, 0.16, when, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=t["fg"])
            ax.text(i, 0.27, body, ha="center", va="bottom", fontsize=9.8, color=t["fg"], linespacing=1.35)
        else:
            ax.text(i, -0.16, when, ha="center", va="top", fontsize=10.5, fontweight="bold", color=t["fg"])
            ax.text(i, -0.27, body, ha="center", va="top", fontsize=9.8, color=t["fg"], linespacing=1.35)
    spans = {"r1": [0, 1], "r2": [2, 6], "test": [7, 7]}
    for ph, (a, b) in spans.items():
        col = t["fg"] if ph == "test" else t["neutral"]
        ax.plot([a - 0.42, b + 0.42], [0.9, 0.9], color=col, lw=2.5, solid_capstyle="round")
        x, ha = ((a + b) / 2, "center") if ph != "test" else (b + 0.42, "right")
        ax.text(x, 0.925, phase[ph], ha=ha, va="bottom", fontsize=10.5, color=t["fg"], fontweight="bold")
    head(fig, t, "From a first design to the final pre-registered test",
         "All dates 2026, Taiwan time (UTC+8); events are evenly spaced, not to scale. Cloud GPU was used only to "
         "fine-tune the generators and generate images; every classifier ran on a laptop.")
    save(fig, "fig12_timeline", mode)


# ------------------------------------------------------------------------------------------------ private export

def export_per_chain():
    """Per-chain recalls from the per-image test predictions: class 3 and class 4 for the main table (n = 4, 200
    synthetic images, 1 : 1), and class 3 for the validation-chosen settings (R, G*, P*, P_rot*; n = 4 and 10)."""
    d = SRC["test"]
    frozen = sorted(d.glob("frozen_settings*.json"))
    assert len(frozen) == 1, "expected exactly one frozen settings file in the test folder"
    sel = json.loads(frozen[0].read_text(encoding="utf-8"))["primary_selected"]
    z = np.load(d / "dinov2" / "preds.npz")
    keys = [str(k) for k in z["keys"]]
    code = np.array([{"3": 0, "4": 1, "6": 2}[x] for x in z["labels"]])
    masks = {"3": z["roles"] == "c3_test_new", "4": z["roles"] == "c4_test"}
    cnn = {}
    for p in (d / "cnn" / "runs").glob("*.npz"):
        r = np.load(p)
        run = json.loads(str(r["run"]))
        cnn[(run["model"], run["arm"], run["n"], run["chain"], run["g"], run["seed"])] = r["pred_test"]

    def rec(p, c):
        return float((p[masks[c]] == code[masks[c]]).mean())

    out = {"3": {}, "4": {}}
    for alg, _, _ in ALGS:
        for arm in ARMS:
            g = 0 if arm == "R" else 200
            for c in out:
                v = []
                for k in range(3):
                    if alg == "dinov2":
                        preds = [z["preds"][keys.index(f"{arm}|n4|s{k}|g{g}|r1|crop128")]]
                    else:
                        preds = [cnn[(alg, arm, 4, k, g, s)] for s in range(6)]
                    v.append(float(np.mean([rec(p, c) for p in preds])))
                out[c][f"{alg}|{arm}"] = v
    prim, chosen = {}, {}
    for n in (4, 10):
        q = {"R": [rec(z["preds"][keys.index(f"R|n{n}|s{k}|g0|r1|crop128")], "3") for k in range(3)]}
        for arm in ("G", "P", "P_rot"):
            s = [sel[f"n{n}"][f"s{k}"][arm] for k in range(3)]
            q[arm + "*"] = [rec(z["preds"][keys.index(f"{arm}|n{n}|s{k}|g{s[k]['g']}|r{s[k]['ratio']}|crop128")], "3")
                            for k in range(3)]
            chosen[f"{arm}*|n{n}"] = [f"{s[k]['g']} images, {s[k]['ratio']} : 1" for k in range(3)]
        prim[f"n{n}"] = q
    doc = {"note": "recall on the test images per chain (s0, s1, s2). class3_recall / class4_recall: n = 4, 200 "
                   "synthetic images, weight 1 : 1; CNN values are the mean of 6 seeds. primary: class-3 recall of "
                   "DINOv2-L + LR at the setting chosen per chain on validation data (chosen_setting: number of "
                   "synthetic images, weight real : synthetic). Derived from the per-image predictions of the "
                   "single test run.",
           "class3_recall": out["3"], "class4_recall": out["4"], "primary": prim, "chosen_setting": chosen}
    (RES / "per_chain_recall.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--private-root", type=Path, default=None,
                    help="private working tree with the dataset; enables the image panels fig03-fig05")
    ap.add_argument("--generated-dir", type=Path, default=None,
                    help="folder with the contact sheets of the generated images, relative to --private-root")
    ap.add_argument("--test-dir", type=Path, default=None,
                    help="folder with the predictions of the test run, relative to --private-root")
    args = ap.parse_args()
    if args.private_root:
        if not (args.generated_dir and args.test_dir):
            ap.error("--private-root needs --generated-dir and --test-dir")
        SRC["generated"] = args.private_root / args.generated_dir
        SRC["test"] = args.private_root / args.test_dir
        export_per_chain()
    S = stats()
    assert S["mode"] == "test"
    for mode, t in THEMES.items():
        style(t)
        fig00(S, t, mode)
        fig01(S, t, mode)
        fig02(t, mode)
        if args.private_root:
            fig03(t, mode, args.private_root)
            fig04(t, mode, args.private_root)
            fig05(t, mode, args.private_root)
        fig06(S, t, mode)
        fig07(S, t, mode)
        fig08(S, t, mode)
        fig09(S, t, mode)
        fig10(S, t, mode)
        fig11(S, t, mode)
        fig12(t, mode)
    print("figures written to", FIG.relative_to(ROOT).as_posix() + "/",
          "" if args.private_root else "(fig03, fig04, fig05 skipped: they need --private-root)")


if __name__ == "__main__":
    main()
