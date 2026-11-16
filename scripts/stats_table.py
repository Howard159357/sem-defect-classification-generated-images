"""Write results/v2_stats.md, a readable table view of results/v2_stats.json (every number comes from the JSON).

  python scripts/stats_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALG = {"dinov2": "DINOv2-L + LR", "resnet18": "ResNet-18", "convnext_tiny": "ConvNeXt-T"}
HYP = {"H-A1": "S1 (G − R)", "H-A2": "S2 (G − P)", "H-A3": "S3 (G − P_box)"}      # JSON key -> label
G_VALUES = [10, 25, 50, 100, 200, 400]      # numbers of synthetic images in the gain table, in the order stored


def s(x):
    return f"{x:+.3f}".replace("-", "−")


def iv(lo, hi):
    return f"[{s(lo)}, {s(hi)}]"


def chains(v):
    return " / ".join(s(x) for x in v)


def pval(p):
    return "< 0.001" if p < 0.0005 else f"{p:.3f}"


def key(k):
    """A JSON key as a code span; pipes are escaped so that a Markdown table keeps it in one cell."""
    return "`" + k.replace("|", "\\|") + "`"


def tested(x):
    if x["judgement"] == "described only":
        return "not tested"
    return "passed" if x["holm_pass"] else "not passed"


def guard(g):
    a, b = g["4"], g["6"]
    ch = f"{a['chains_over']} chain{'' if a['chains_over'] == 1 else 's'}"
    return (f"class 4 {s(a['diff'])} ({ch}, {a['runs_over']} of {a['runs']} runs beyond −0.02), "
            f"good {s(b['diff'])} ({b['chains_over']}, {b['runs_over']} of {b['runs']}) → "
            f"{'pass' if g['pass'] else '**fail**'}")


def main():
    S = json.loads((RES / "v2_stats.json").read_text(encoding="utf-8"))
    E = S["explore"]
    n4 = S["primary"]["n4"]["guard"]
    L = ["# Test statistics (public AnomalyGen v1.0.0): test set, run once", "",
         "Table view of `v2_stats.json`, written by `scripts/stats_table.py`. Class-3 recall is measured on the 300 "
         f"new class-3 test images; the guardrails use {n4['4']['N']} class-4 and {n4['6']['N']} good test images. Δ "
         "is a difference in class-3 recall. Intervals are 95% paired-bootstrap intervals (2,000 resamples of chains "
         "and images). Every interval rests on 3 chains; the pre-registered plan notes for the secondary analysis "
         "that such intervals under-cover. The t interval treats the 3 chains as the sample (2 degrees of freedom).",
         "", "## Primary analysis H1 (DINOv2-L + LR, G* − R, setting chosen per chain on validation data)", "",
         "n = 4 is the single pre-registered hypothesis test (passed = p < 0.05); n = 10 is described only, as set "
         "in advance.", "",
         "| n | R | G* | Δ | 95% interval | p | hypothesis test | per-chain Δ | t interval | guardrail | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("n4", "n10"):
        x = S["primary"][n]
        L.append(f"| {n[1:]} | {x['recB']:.3f} | {x['recA']:.3f} | {s(x['delta'])} | {iv(x['lo'], x['hi'])} | "
                 f"{pval(x['p'])} | {tested(x)} | {chains(x['chains'])} | {iv(x['t_lo'], x['t_hi'])} | "
                 f"{guard(x['guard'])} | **{x['final']}** |")
    L += ["", "## H2 (DINOv2-L + LR, G* − P*; n = 4 is a single hypothesis test, n = 10 described only)", "",
          "| n | P* | G* | Δ | 95% interval | p | hypothesis test | per-chain Δ | t interval | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("n4", "n10"):
        x = S["H-v2b"][n]
        L.append(f"| {n[1:]} | {x['recB']:.3f} | {x['recA']:.3f} | {s(x['delta'])} | {iv(x['lo'], x['hi'])} | "
                 f"{pval(x['p'])} | {tested(x)} | {chains(x['chains'])} | {iv(x['t_lo'], x['t_hi'])} | "
                 f"{x['judgement']} |")
    L += ["", "## Main table: n = 4, fixed setting (200 synthetic images, weight 1 : 1; mean over chains and seeds)",
          "", "| classifier | training set | class 3 | class 4 | good | balanced accuracy | runs |",
          "|---|---|---|---|---|---|---|"]
    for k, a in S["absolute"].items():
        alg, arm = k.split("|")
        L.append(f"| {ALG[alg]} | {arm} | {a['3']:.3f} | {a['4']:.3f} | {a['6']:.3f} | {a['bacc']:.3f} | "
                 f"{a['runs']} |")
    L += ["", "## Secondary analysis (added after the validation results were seen; n = 4, fixed setting)", "",
          "Holm: correction over the 3 S1 tests, and separately over the 6 S2 and S3 tests. SD between "
          "seeds: spread of the class-3 recall of G between training seeds (CNNs only).", "",
          "| classifier | hypothesis | Δ | bootstrap interval | per-chain Δ | t interval | SD between seeds | Holm | "
          "guardrail (G vs R) | class-3 judgement | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in S["secondary"].items():
        alg, h = k.split("|")
        sd = "n/a" if x["seed_sd_A"] is None else f"{x['seed_sd_A']:.3f}"
        why = f" ({'; '.join(x['why'])})" if x.get("why") else ""
        L.append(f"| {ALG[alg]} | {HYP[h]} | {s(x['delta'])} | {iv(x['lo'], x['hi'])} | {chains(x['chains'])} | "
                 f"{iv(x['t_lo'], x['t_hi'])} | {sd} | {'passed' if x['holm_pass'] else 'not passed'} | "
                 f"{guard(x['guard'])} | {x['judgement']}{why} | **{x['final']}** |")
    L += ["", "## Confusion matrices (image counts summed over chains and seeds)", "",
          "Each cell lists how many images of the true class were called class 3, class 4 and good.", "",
          "| model | true class 3 | true class 4 | true good |", "|---|---|---|---|"]
    for k, cm in S["confusion"].items():
        L.append(f"| {key(k)} | " + " | ".join(", ".join(str(v) for v in row) for row in cm) + " |")
    L += ["", "## Exploratory comparisons (described only, no verdict)", "",
          "`*` = setting chosen on validation data; `g200`, `g50` = 200 or 50 synthetic images at weight 1 : 1; "
          "`c3_test_old` = the first-round class-3 test set (used repeatedly during design, not clean); `full224` = "
          "whole images at 224 px.", "", "| comparison | Δ | interval | per-chain Δ |", "|---|---|---|---|"]
    for k, x in E.items():
        if isinstance(x, dict) and "delta" in x:
            L.append(f"| {key(k)} | {s(x['delta'])} | {iv(x['lo'], x['hi'])} | {chains(x['chains'])} |")
    L += ["", "## Gain by number of synthetic images (DINOv2-L + LR, Δ class-3 recall vs R, mean of 3 chains)", "",
          "`r1`, `r2` = weight real : synthetic of 1 : 1 or 2 : 1. Columns: number of synthetic images.", "",
          "| training set | " + " | ".join(str(g) for g in G_VALUES) + " |", "|---|" + "---|" * len(G_VALUES)]
    for k, v in E["g_curve"].items():
        L.append(f"| {key(k)} | " + " | ".join(s(x) for x in v) + " |")
    rc = E["real_curve"]
    ns = sorted(int(k) for k in rc)
    L += ["", "## Real images only (DINOv2-L + LR, class-3 recall, mean ± SD over sets of real images)", "",
          "10 chains for n = 4 to 50; 5, 2 and 1 sets for n = 100, 200 and 500 (the ± 0.000 at 500 is a single set).",
          "", "| real class-3 images | " + " | ".join(str(k) for k in ns) + " |", "|---|" + "---|" * len(ns),
          "| recall | " + " | ".join(f"{rc[str(k)][0]:.3f} ± {rc[str(k)][1]:.3f}" for k in ns) + " |"]
    L += ["", "## Equivalent number of real images (point estimates, no interval)", "",
          "| setting | equivalent real images |", "|---|---|"]
    for k, v in E["equivalent_real"].items():
        if k != "note":
            L.append(f"| {key(k)} | {v if isinstance(v, str) else f'{v:.1f}'} |")
    (RES / "v2_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote results/v2_stats.md")


if __name__ == "__main__":
    main()
