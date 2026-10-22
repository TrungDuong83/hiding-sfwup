"""Figures of the paper (called by paper_assets.py): F_time, F_scal and Fig. 1 (Algorithm 1).

Each figure is written as vector PDF, EPS and a 1000 dpi PNG at its print width (190 mm; Fig. 1
140 mm). Text >= 7 pt at print size, TrueType fonts embedded (pdf.fonttype = ps.fonttype = 42),
Okabe-Ito colours with a distinct marker per method, no titles inside the figures. Infeasible
runs (HF > 0) are drawn with hollow markers. Fig. 1 is drawn with matplotlib; its Graphviz source
is written next to it (fig1_algorithm.dot) for use with `dot -Tpdf`.
"""

import os
import re
import statistics

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # EPS: fixed %%CreationDate (reproducible files)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paper_assets as PA  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

import common as V  # noqa: E402

MM = 1 / 25.4
DPI = 1000


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Arial", "Liberation Sans", "DejaVu Sans"):
        if n in names:
            return n
    return "DejaVu Sans"


FONT = _font()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "grid.linewidth": 0.4,
        "grid.color": "#d9d9d9",
        "lines.linewidth": 1.2,
        "savefig.dpi": DPI,
    }
)
# Okabe-Ito palette; one colour and one marker per method
STYLE = {
    "SC-only": ("#E69F00", "D"),
    "SDIF": ("#56B4E9", "o"),
    "Max-RISWU": ("#009E73", "s"),
    "Min-RISWU": ("#0072B2", "^"),
    "SMSE": ("#D55E00", "v"),
    "MSU-MIU": ("#CC79A7", "P"),
}
INK = "#000000"
GREY = "#555555"


def lighten(c, f=0.78):
    """Opaque tint of colour c (EPS has no transparency)."""
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        meta = {"CreationDate": None} if ext in ("pdf", "eps") else None
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=DPI, metadata=meta)
    sizes = [
        t.get_fontsize()
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    w, h = fig.get_size_inches() / MM
    plt.close(fig)
    print(f"  {name}: {w:.0f} x {h:.0f} mm, smallest text {min(sizes):.1f} pt")
    return {"name": name, "width_mm": w, "height_mm": h, "min_font_pt": min(sizes)}


def plain_log(axis, ticks, scale=1.0):
    """Log axis labelled with plain numbers (a 10^k label has a superscript below 7 pt)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / scale:g}"))
    axis.set_minor_formatter(NullFormatter())


def method_handles(infeasible, note, mew, ms):
    """One entry per method; right after it, the hollow marker as drawn for its infeasible
    runs (only for the methods in `infeasible`, i.e. with such a point in the figure)."""
    hs = []
    for m in PA.METHODS:
        c, mk = STYLE[m]
        hs.append(plt.Line2D([], [], color=c, marker=mk, ms=4.5, ls="none", label=m))
        if m in infeasible:
            hs.append(
                plt.Line2D(
                    [],
                    [],
                    color=c,
                    marker=mk,
                    mec=c,
                    mfc="white",
                    mew=mew,
                    ms=ms,
                    ls="none",
                    label=f"{m}, {note}",
                )
            )
    return hs


# ------------------------------------------------------------------ F_time
def f_time(out):
    U = {(u["dataset"], u["method"]): u for u in PA.main_units()}
    fig, ax = plt.subplots(figsize=(190 * MM, 72 * MM))
    n = len(PA.METHODS)
    bad = set()  # methods with an infeasible (hollow) point
    for i, ds in enumerate(V.DS):
        for j, m in enumerate(PA.METHODS):
            u = U[(ds, m)]
            c, mk = STYLE[m]
            x = i + (j - (n - 1) / 2) * 0.12
            if not PA.is_feasible(u):
                bad.add(m)
            ax.plot(
                [x],
                [PA.san_time(u)],
                marker=mk,
                ms=5,
                mec=c,
                mfc=c if PA.is_feasible(u) else "white",
                mew=1.1,
                ls="none",
                zorder=3,
            )
        if i:
            ax.axvline(i - 0.5, color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_yscale("log")
    plain_log(ax.yaxis, [0.1, 0.3, 1, 3, 10, 30, 100, 300])
    ax.set_ylim(0.09, 300)
    ax.set_xticks(range(len(V.DS)))
    ax.set_xticklabels([PA.NAME[d] for d in V.DS])
    ax.set_xlim(-0.55, len(V.DS) - 0.45)
    ax.set_ylabel("Sanitization time (s)")
    ax.grid(axis="y", which="major")
    ax.tick_params(axis="x", length=0)
    hs = method_handles(bad, "infeasible (HF > 0)", 1.1, 5)
    ax.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=len(hs),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.1)
    return save(fig, out, "F_time")


# ------------------------------------------------------------------ F_scal
def f_scal(out):
    U = PA.scal_units()
    wl = PA.scal_workload()
    H, dh = 88, 5.08  # dh: extra legend row above the axes (axes keep their size in mm)
    hollow_ms = 5  # hollow markers: larger and on top (a thicker edge fills a small hollow "P")
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, (H + dh) * MM))
    bad = set()
    for ax, ds in zip(axes, ("kosarak", "chainstore")):
        for m in PA.METHODS:
            c, mk = STYLE[m]
            xs, mean, lo, hi, feas = [], [], [], [], []
            for pc in range(10, 101, 10):
                L = [u for u in U if u["dataset"] == ds and u["method"] == m and u["_pct"] == pc]
                t = [PA.san_time(u) for u in L]
                xs.append(int(wl[(ds, 0, pc)]["n_tx"]))
                mean.append(statistics.fmean(t))
                lo.append(min(t))
                hi.append(max(t))
                feas.append(all(PA.is_feasible(u) for u in L))
            ax.fill_between(xs, lo, hi, color=lighten(c), lw=0, zorder=1)
            ax.plot(xs, mean, color=c, lw=1.0, zorder=2)
            if not all(feas):
                bad.add(m)
            for x, y, f in zip(xs, mean, feas):
                ax.plot(
                    [x],
                    [y],
                    marker=mk,
                    ms=4 if f else hollow_ms,
                    mec=c,
                    mfc=c if f else "white",
                    mew=1.0,
                    ls="none",
                    zorder=3 if f else 4,  # hollow on top of overlapping filled markers
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log(ax.xaxis, [1e5, 2e5, 5e5, 1e6], scale=1e6)
        plain_log(ax.yaxis, [0.5, 1, 2, 5, 10, 20, 50])
        ax.set_xlabel(f"|Q| ({PA.NAME[ds]}, million transactions)")
        ax.grid(which="major")
        ax.text(0.02, 0.97, PA.NAME[ds], transform=ax.transAxes, va="top", fontsize=7.5)
    axes[0].set_ylabel("Sanitization time (s)")
    fig.legend(
        handles=method_handles(bad, "≥ 1 of 3 orders infeasible (HF > 0)", 1.0, hollow_ms)
        + [plt.Rectangle((0, 0), 1, 1, color=lighten(GREY), label="min–max over 3 orders")],
        loc="upper center",
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=1 - (H * (1 - 0.84) + dh) / (H + dh),
        bottom=H * 0.13 / (H + dh),
        wspace=0.18,
    )
    return save(fig, out, "F_scal")


# ------------------------------------------------------------------ Fig. 1
# Node texts; the line numbers refer to Algorithm 1 and its procedure HideOne.
N = {
    "in": "Input: Q, S, ξ, strategy",
    "p1": "Phase 1 (lines 1-4):\ntwu($T_j$), sumtwu; SC, NSC, Wt\n(once, on Q)",
    "p2": "Phase 2 (lines 5-6):\ninitial victim $v_0$(X) on Q\n(Sub-Alg. 1);\n"
    "order S by decreasing\n"
    "RISWU($v_0$(X), X),\nties by item identifier",
    "l7": "line 7: HideOne(X)\nfor each X in this order",
    "p3": "Phase 3 (lines 8-11):\nR = {X ∈ S : wus(X) ≥ ξ}\nat the start of each round;\n"
    "HideOne(X) for X ∈ R\nin the order of line 6;\nrepeat until R = ∅",
    "out": "Return Q' (line 12)",
    "h1": "(1) wus(X) < ξ ?",
    "h2": "(2) victim re-selected on the\ncurrent database (Sub-Alg. 1),\nkept for the call",
    "w": "(4-16) while\nwus(X) ≥ ξ ?",
    "tn": "$T_X$ and N (lines 6-7)",
    "a": "α (Prop. 3, line 8)",
    "pos": "α > 0:\ncontributions (Sub-Alg. 2)\n→ allocation (Sub-Alg. 3, Prop. 4)\n"
    "→ apply: iu, S($T_j$), twu($T_j$),\nsumtwu (lines 10-12)",
    "zero": "α = 0 (wus = ξ):\nescalation\n(Sub-Alg. 4,\nline 14)",
    "ret": "return",
}


def write_dot(path):
    def q(s):
        """dot label: quoted string, or HTML-like label when it has a subscript ($T_X$)."""
        if "$" not in s:
            return '"' + s.replace("\n", "\\n") + '"'
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\$(\w)_(\w)\$", r"\1<SUB>\2</SUB>", s)
        return "<" + s.replace("\n", "<BR/>") + ">"

    L = [
        "// Fig. 1: Algorithm 1 and its procedure HideOne. Render: dot -Tpdf fig1_algorithm.dot",
        "// Line numbers inside cluster_hide are those of procedure HideOne.",
        "digraph alg1 {",
        '  graph [rankdir=TB, fontname="Arial", fontsize=9, nodesep=0.3, ranksep=0.3];',
        '  node [shape=box, style=rounded, fontname="Arial", fontsize=9];',
        '  edge [fontname="Arial", fontsize=8];',
    ]
    for k in ("in", "p1", "p2", "l7", "p3", "out"):
        L.append(f"  {k} [label={q(N[k])}];")
    L += [
        '  subgraph cluster_hide { label="HideOne(X)"; style="rounded,dashed";',
        f'    h1 [shape=diamond, style="", label={q(N["h1"])}];',
        f'    h2 [label={q(N["h2"])}];',
        f'    w [shape=diamond, style="", label={q(N["w"])}];',
        f'    tn [label={q(N["tn"])}];',
        f'    a [label={q(N["a"])}];',
        f'    pos [label={q(N["pos"])}];',
        f'    zero [label={q(N["zero"])}];',
        '    ret [shape=oval, style="", label="return"];',
        "  }",
        "  in -> p1 -> p2 -> l7 -> p3 -> out;",
        '  p3 -> p3 [label="R ≠ ∅"];',
        '  l7 -> h1 [style=dashed, label="calls"];',
        '  p3 -> h1 [style=dashed, label="calls"];',
        '  h1 -> ret [label="yes"];',
        '  h1 -> h2 [label="no"];',
        "  h2 -> w;",
        '  w -> tn [label="yes"];',
        '  w -> ret [label="no"];',
        "  tn -> a;",
        '  a -> pos [label="α > 0"];',
        '  a -> zero [label="α = 0"];',
        '  pos -> w [label="back to the while test"];',
        "  zero -> w;",
        "}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def fig1(out):
    W, H, Y0 = 140, 150, 13
    fig = plt.figure(figsize=(W * MM, (H - Y0) * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")
    fs = 7.0

    def box(x, y, w, h, text, kind="box"):
        if kind == "diamond":
            ax.add_patch(
                Polygon(
                    [(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)],
                    closed=True,
                    fc="white",
                    ec=INK,
                    lw=0.7,
                )
            )
        else:
            rs = h / 2 if kind == "term" else 1.6
            ax.add_patch(
                FancyBboxPatch(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    boxstyle=f"round,pad=0,rounding_size={rs}",
                    fc="white",
                    ec=INK,
                    lw=0.7,
                )
            )
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.25)
        return {"x": x, "y": y, "w": w, "h": h}

    def arrow(p, q, text=None, tpos=None, dashed=False, ha="left"):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.7,
                color=INK,
                linestyle=(0, (3, 2)) if dashed else "-",
                shrinkA=0,
                shrinkB=0,
            )
        )
        if text:
            ax.text(*tpos, text, fontsize=7, ha=ha, va="center", color=INK)

    def poly(pts, head=True):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=INK, lw=0.7, solid_capstyle="butt")
        if head:
            arrow(pts[-2], pts[-1])

    top = lambda b: (b["x"], b["y"] + b["h"] / 2)  # noqa: E731
    bot = lambda b: (b["x"], b["y"] - b["h"] / 2)  # noqa: E731
    lft = lambda b: (b["x"] - b["w"] / 2, b["y"])  # noqa: E731
    rgt = lambda b: (b["x"] + b["w"] / 2, b["y"])  # noqa: E731

    # main flow (left column)
    cx, w = 28, 44
    b_in = box(cx, 143, w, 7, N["in"], kind="term")
    b_p1 = box(cx, 128, w, 14, N["p1"])
    b_p2 = box(cx, 103, w, 27, N["p2"])
    b_l7 = box(cx, 80, w, 10, N["l7"])
    b_p3 = box(cx, 50, w, 30, N["p3"])
    b_out = box(cx, 24, w, 7, N["out"], kind="term")
    for a_, b_ in ((b_in, b_p1), (b_p1, b_p2), (b_p2, b_l7), (b_l7, b_p3), (b_p3, b_out)):
        arrow(bot(a_), top(b_))
    xl = cx - w / 2 - 3
    poly([(cx - w / 2, 42), (xl, 42), (xl, 58), (cx - w / 2, 58)])
    ax.text(xl - 1.6, 50, "R ≠ ∅", fontsize=7, ha="center", va="center", rotation=90)

    # HideOne frame (right column)
    fx0, fx1, fy0, fy1 = 54, 139, 14, 148
    ax.add_patch(
        FancyBboxPatch(
            (fx0, fy0),
            fx1 - fx0,
            fy1 - fy0,
            boxstyle="round,pad=0,rounding_size=2.5",
            fc="#f2f2f2",
            ec=GREY,
            lw=0.7,
            linestyle=(0, (4, 2)),
        )
    )
    ax.text(
        fx0 + 2.5, fy1 - 3.5, "HideOne(X)", fontsize=8, ha="left", va="center", fontweight="bold"
    )
    ax.text(
        fx0 + 2.5,
        fy1 - 7.5,
        "(line numbers of HideOne)",
        fontsize=7,
        ha="left",
        va="center",
        color=GREY,
    )
    hx = 96
    b_h1 = box(hx, 132, 36, 12, N["h1"], kind="diamond")
    b_ret = box(129, 132, 13, 7, N["ret"], kind="term")
    b_h2 = box(hx, 114, 46, 14, N["h2"])
    b_w = box(hx, 94, 36, 15, N["w"], kind="diamond")
    b_tn = box(hx, 78, 36, 7, N["tn"])
    b_a = box(hx, 67, 36, 7, N["a"])
    b_pos = box(83, 43, 44, 23, N["pos"])
    b_zero = box(122, 43, 29, 23, N["zero"])
    arrow(rgt(b_h1), lft(b_ret), "yes", (rgt(b_h1)[0] + 0.8, 134.3))
    arrow(bot(b_h1), top(b_h2), "no", (hx + 1.2, 123.5))
    arrow(bot(b_h2), top(b_w))
    arrow(bot(b_w), top(b_tn), "yes", (hx + 1.2, 84))
    poly([rgt(b_w), (129, 94), bot(b_ret)])
    ax.text(116, 96, "no", fontsize=7, ha="left", va="center")
    arrow(bot(b_tn), top(b_a))
    poly([bot(b_a), (hx, 59.5), (83, 59.5), top(b_pos)])
    poly([(hx, 59.5), (122, 59.5), top(b_zero)])
    ax.text(84, 61.5, "α > 0", fontsize=7, ha="left", va="center")
    ax.text(110, 61.5, "α = 0", fontsize=7, ha="left", va="center")
    # back to the while test
    yb = 26
    poly([bot(b_zero), (122, yb), (58, yb)], head=False)
    poly([bot(b_pos), (83, yb)], head=False)
    poly([(58, yb), (58, 94), lft(b_w)])
    ax.text(90, yb - 2.5, "back to the while test", fontsize=7, ha="center", va="center")
    # calls
    arrow(rgt(b_l7), (fx0, 80), "calls", (cx + w / 2 + 0.6, 82.3), dashed=True)
    arrow(rgt(b_p3), (fx0, 50), "calls", (cx + w / 2 + 0.6, 52.3), dashed=True)

    write_dot(os.path.join(out, "fig1_algorithm.dot"))
    return save(fig, out, "fig1_algorithm")


def main(out):
    print("figures ->", os.path.relpath(out, V.REPO), f"(font: {FONT})")
    return [f_time(out), f_scal(out), fig1(out)]
