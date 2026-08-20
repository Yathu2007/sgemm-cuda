#!/usr/bin/env python3
"""Turns results/results.csv into the README results table and the
GFLOPS-vs-N chart.

The table (markdown) is stdlib-only so the sweep always produces it. The chart
needs matplotlib; if it is missing the script says so and carries on rather
than failing the sweep.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

CUBLAS = "cublas"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["N"] = int(r["N"])
        r["gflops"] = float(r["gflops"])
        r["median_ms"] = float(r["median_ms"])
        r["max_rel_err"] = float(r["max_rel_err"])
    return rows


def summarize_clocks(path):
    """Min/median/max of the SM clock, power and temperature during the run."""
    if not os.path.exists(path):
        return None
    sm, pw, tp = [], [], []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 6 or r[0].strip() == "timestamp":
                continue
            try:
                sm.append(float(r[1].split()[0]))
                pw.append(float(r[3].split()[0]))
                tp.append(float(r[4].strip()))
            except (ValueError, IndexError):
                continue
    if not sm:
        return None

    def stat(v):
        v = sorted(v)
        return v[0], v[len(v) // 2], v[-1]

    return {
        "samples": len(sm),
        "sm": stat(sm),
        "power": stat(pw),
        "temp": stat(tp),
    }


def markdown_table(rows):
    sizes = sorted({r["N"] for r in rows})
    by_kernel = defaultdict(dict)
    for r in rows:
        by_kernel[r["kernel"]][r["N"]] = r

    baseline = by_kernel.get(CUBLAS, {})
    # cuBLAS last: it is the yardstick, not a rung of the ladder.
    order = sorted((k for k in by_kernel if k != CUBLAS)) + (
        [CUBLAS] if CUBLAS in by_kernel else []
    )
    naive = by_kernel.get("K0_naive", {})

    out = []
    out.append("### GFLOPS")
    out.append("")
    out.append("| Kernel | " + " | ".join(f"N={n}" for n in sizes) + " |")
    out.append("|---" * (len(sizes) + 1) + "|")
    for k in order:
        cells = []
        for n in sizes:
            r = by_kernel[k].get(n)
            cells.append(f"{r['gflops']:.1f}" if r else "—")
        out.append(f"| `{k}` | " + " | ".join(cells) + " |")

    if baseline:
        out += ["", "### % of cuBLAS", ""]
        out.append("| Kernel | " + " | ".join(f"N={n}" for n in sizes) + " |")
        out.append("|---" * (len(sizes) + 1) + "|")
        for k in order:
            cells = []
            for n in sizes:
                r, b = by_kernel[k].get(n), baseline.get(n)
                cells.append(
                    f"{100.0 * r['gflops'] / b['gflops']:.1f}%"
                    if r and b
                    else "—"
                )
            out.append(f"| `{k}` | " + " | ".join(cells) + " |")

    if naive:
        out += ["", "### Speedup vs K0 naive", ""]
        out.append("| Kernel | " + " | ".join(f"N={n}" for n in sizes) + " |")
        out.append("|---" * (len(sizes) + 1) + "|")
        for k in order:
            cells = []
            for n in sizes:
                r, b = by_kernel[k].get(n), naive.get(n)
                cells.append(
                    f"{r['gflops'] / b['gflops']:.2f}x" if r and b else "—"
                )
            out.append(f"| `{k}` | " + " | ".join(cells) + " |")

    out += ["", "### Median time (ms) and max relative error vs cuBLAS", ""]
    out.append("| Kernel | N | median ms | GFLOPS | % cuBLAS | max rel err |")
    out.append("|---|---|---|---|---|---|")
    for k in order:
        for n in sizes:
            r = by_kernel[k].get(n)
            if not r:
                continue
            b = baseline.get(n)
            pct = f"{100.0 * r['gflops'] / b['gflops']:.1f}%" if b else "—"
            err = "—" if r["max_rel_err"] < 0 else f"{r['max_rel_err']:.2e}"
            out.append(
                f"| `{k}` | {n} | {r['median_ms']:.4f} | {r['gflops']:.1f} | {pct} | {err} |"
            )
    return "\n".join(out)


def svg_chart(rows, path):
    """Stdlib fallback so the sweep always produces a chart, even where
    matplotlib cannot be installed (no pip / no root)."""
    W, H = 760, 480
    L, R, T, B = 70, 170, 46, 56
    pw, ph = W - L - R, H - T - B

    sizes = sorted({r["N"] for r in rows})
    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append((r["N"], r["gflops"]))
    order = sorted((k for k in by_kernel if k != CUBLAS)) + (
        [CUBLAS] if CUBLAS in by_kernel else []
    )

    gmin = min(r["gflops"] for r in rows)
    gmax = max(r["gflops"] for r in rows)
    lo, hi = math.floor(math.log10(gmin)), math.ceil(math.log10(gmax))
    xs = [math.log2(n) for n in sizes]
    x0, x1 = min(xs), max(xs)

    def px(n):
        return L + (math.log2(n) - x0) / (x1 - x0) * pw

    def py(g):
        return T + ph - (math.log10(g) - lo) / (hi - lo) * ph

    colors = ["#d1495b", "#2a9d8f", "#4c6ef5", "#e9a13b", "#8e44ad", "#16a085"]
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
    ]

    # log-decade gridlines
    for d in range(lo, hi + 1):
        for m in range(1, 10):
            g = m * 10**d
            if not (gmin * 0.9 <= g <= gmax * 1.1) or g <= 0:
                continue
            y = py(g)
            if not (T <= y <= T + ph):
                continue
            major = m == 1
            o.append(
                f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                f'stroke="#ccc" stroke-width="{1 if major else 0.4}"/>'
            )
            if major:
                o.append(
                    f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" '
                    f'text-anchor="end" fill="#333">{g:g}</text>'
                )
    for n in sizes:
        x = px(n)
        o.append(
            f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" '
            f'stroke="#ccc" stroke-width="0.6"/>'
        )
        o.append(
            f'<text x="{x:.1f}" y="{T+ph+18}" font-size="11" '
            f'text-anchor="middle" fill="#333">{n}</text>'
        )

    o.append(
        f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>'
    )

    for i, k in enumerate(order):
        pts = sorted(by_kernel[k])
        c = "#000" if k == CUBLAS else colors[i % len(colors)]
        dash = ' stroke-dasharray="6,4"' if k == CUBLAS else ""
        d = " ".join(f"{px(n):.1f},{py(g):.1f}" for n, g in pts)
        o.append(
            f'<polyline points="{d}" fill="none" stroke="{c}" '
            f'stroke-width="2"{dash}/>'
        )
        for n, g in pts:
            o.append(
                f'<circle cx="{px(n):.1f}" cy="{py(g):.1f}" r="3.5" fill="{c}"/>'
            )
        ly = T + 14 + i * 20
        o.append(
            f'<line x1="{L+pw+14}" y1="{ly}" x2="{L+pw+40}" y2="{ly}" '
            f'stroke="{c}" stroke-width="2"{dash}/>'
        )
        o.append(
            f'<text x="{L+pw+46}" y="{ly+4}" font-size="12" fill="#222">{k}</text>'
        )

    o.append(
        f'<text x="{W/2}" y="26" font-size="15" text-anchor="middle" '
        f'fill="#111">SGEMM throughput &#8212; RTX 4050 Laptop (sm_89), FP32</text>'
    )
    o.append(
        f'<text x="{L+pw/2}" y="{H-12}" font-size="12" text-anchor="middle" '
        f'fill="#333">N (square matrix dimension, log2)</text>'
    )
    o.append(
        f'<text x="16" y="{T+ph/2}" font-size="12" text-anchor="middle" '
        f'fill="#333" transform="rotate(-90 16 {T+ph/2})">GFLOPS (log10, median)</text>'
    )
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))
    print(f"   wrote {path}")
    return True


def chart(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "   matplotlib not available -- writing SVG chart instead "
            "(pip install matplotlib for the PNG)",
            file=sys.stderr,
        )
        return svg_chart(rows, os.path.splitext(path)[0] + ".svg")

    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append((r["N"], r["gflops"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    order = sorted((k for k in by_kernel if k != CUBLAS)) + (
        [CUBLAS] if CUBLAS in by_kernel else []
    )
    for k in order:
        pts = sorted(by_kernel[k])
        style = dict(marker="o", linewidth=2)
        if k == CUBLAS:
            style.update(linestyle="--", color="black", marker="s")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=k, **style)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(sorted({r["N"] for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("N (square matrix dimension)")
    ax.set_ylabel("GFLOPS (FP32, median of timed iterations)")
    ax.set_title("SGEMM throughput -- RTX 4050 Laptop (sm_89)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"   wrote {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--clocks", default="results/clocks.csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        sys.exit("no rows in " + args.results)

    table = markdown_table(rows)
    table_path = os.path.join(args.out_dir, "results_table.md")

    clk = summarize_clocks(args.clocks)
    clock_md = ""
    if clk:
        s, p, t = clk["sm"], clk["power"], clk["temp"]
        clock_md = (
            "\n### Observed GPU state during the sweep\n\n"
            f"{clk['samples']} samples at 1 Hz.\n\n"
            "| | min | median | max |\n|---|---|---|---|\n"
            f"| SM clock (MHz) | {s[0]:.0f} | {s[1]:.0f} | {s[2]:.0f} |\n"
            f"| Power (W) | {p[0]:.1f} | {p[1]:.1f} | {p[2]:.1f} |\n"
            f"| Temperature (C) | {t[0]:.0f} | {t[1]:.0f} | {t[2]:.0f} |\n"
        )

    with open(table_path, "w") as f:
        f.write(table + "\n" + clock_md)
    print(f"   wrote {table_path}")
    print(table)
    if clock_md:
        print(clock_md)

    chart(rows, os.path.join(args.out_dir, "gflops_vs_n.png"))


if __name__ == "__main__":
    main()
