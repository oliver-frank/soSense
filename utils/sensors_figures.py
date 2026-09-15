import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Polygon
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Polygon


def plot_network_time_slices(
    graphs,
    patient_nodes=None,
    titles=None,
    figsize=(7.2, 3.8),
    layout_seed=1,
    panel_width=1.15,
    panel_height=2.35,
    panel_gap=1.45,
    top_slant=0.35,
    bottom_slant=-0.10,
    node_size=30,
    patient_marker="o",
    other_marker="o",
    save_path=None
):
    """
    Plot weighted networks on perspective-like time-slice panels
    using a clean JAMA-style palette and publication layout.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import networkx as nx

    # JAMA-inspired palette from ggsci
    blue = "#374E55"
    teal = "#0073C2"
    orange = "#EFC000"
    red = "#CD534C"
    gray = "#7E6148"
    light_panel = "#FAFAFA"      # almost white
    edge_light = "#AEB6B8"       # lighter network edges
    text_col = "#2B2B2B"

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if titles is None:
        titles = [rf"$t_{{{i+1}}}$" for i in range(len(graphs))]

    if patient_nodes is None:
        patient_nodes = set()

    def get_patient_set(i):
        if isinstance(patient_nodes, dict):
            return set(patient_nodes.get(i, []))
        return set(patient_nodes)

    def normalize_pos(pos):
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])

        xnorm = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
        ynorm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)

        return {node: (x, y) for node, x, y in zip(pos.keys(), xnorm, ynorm)}

    def map_to_panel(x, y, corners, margin=0.15):
        x = margin + x * (1 - 2 * margin)
        y = margin + y * (1 - 2 * margin)

        bl, br, tr, tl = [np.array(c) for c in corners]

        point = (
            (1 - x) * (1 - y) * bl +
            x * (1 - y) * br +
            x * y * tr +
            (1 - x) * y * tl
        )
        return tuple(point)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    n_slices = len(graphs) + 1

    for i in range(n_slices):
        is_assessment_slice = i == len(graphs)
        G = None if is_assessment_slice else graphs[i]
        ox = i * panel_gap
        oy = 0

        bl = (ox, oy)
        br = (ox + panel_width, oy + bottom_slant)
        tr = (ox + panel_width, oy + panel_height + top_slant)
        tl = (ox, oy + panel_height)
        corners = [bl, br, tr, tl]

        panel = Polygon(
            corners,
            closed=True,
            facecolor=light_panel,
            edgecolor="#D9DCDD",
            linewidth=0.6,
            zorder=0
        )
        ax.add_patch(panel)

        # subtle top accent line

        if is_assessment_slice:
            panel.set_facecolor("#FCFCFC")
            panel.set_edgecolor("#C8CCCD")
        
            ax.text(
                ox + panel_width/2,
                oy + panel_height/2,
                "BPRS\nAssessment",
                ha="center",
                va="center",
                fontsize=10,
                color="#6F7678",
                weight="medium"
            )

            continue

        pos = nx.spring_layout(G, seed=layout_seed)
        pos = normalize_pos(pos)
        pos_panel = {n: map_to_panel(x, y, corners) for n, (x, y) in pos.items()}

        weights = np.array([G[u][v].get("weight", 1) for u, v in G.edges()])
        max_w = weights.max() if len(weights) else 1

        for u, v in G.edges():
            x1, y1 = pos_panel[u]
            x2, y2 = pos_panel[v]
            w = G[u][v].get("weight", 1)

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edge_light,
                linewidth=0.35 + 1.35 * w / max_w,
                alpha=0.65,
                zorder=2
            )

        patient_set = get_patient_set(i)
        other_nodes = [n for n in G.nodes if n not in patient_set]
        patient_nodes_i = [n for n in G.nodes if n in patient_set]

        if other_nodes:
            ax.scatter(
                [pos_panel[n][0] for n in other_nodes],
                [pos_panel[n][1] for n in other_nodes],
                s=node_size,
                marker=other_marker,
                facecolor="white",
                edgecolor=blue,
                linewidth=0.8,
                zorder=4
            )

        if patient_nodes_i:
            ax.scatter(
                [pos_panel[n][0] for n in patient_nodes_i],
                [pos_panel[n][1] for n in patient_nodes_i],
                s=node_size * 1.15,
                marker=patient_marker,
                facecolor=red,
                edgecolor="white",
                linewidth=0.8,
                zorder=5
            )

        ax.text(
            ox + 0.10,
            oy + panel_height - 0.16,
            titles[i],
            ha="left",
            va="top",
            fontsize=10.5,
            fontstyle="italic",
            color=text_col
        )

    ax.scatter([], [], s=node_size * 1.15, marker=patient_marker,
               facecolor=red, edgecolor="white", linewidth=0.8,
               label="Patient node")

    ax.scatter([], [], s=node_size, marker=other_marker,
               facecolor="white", edgecolor=blue, linewidth=0.8,
               label="Staff node")

    ax.plot([], [], color=edge_light, linewidth=1.8, alpha=0.8,
            label="Stronger connection")

    ax.plot([], [], color=edge_light, linewidth=0.55, alpha=0.8,
            label="Weaker connection")

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=7.8,
        handlelength=1.8,
        columnspacing=1.3,
        handletextpad=0.6
    )

    ax.set_aspect("equal")
    ax.axis("off")

    xmin = -0.25
    xmax = (n_slices - 1) * panel_gap + panel_width + 0.35
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.35, panel_height + top_slant + 0.25)

    plt.tight_layout(pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, ax
def plot_one_html_per_participant(
    df,
    code_col="code",
    date_col="date",
    variables=None,
    ma_variables=None,
    ma_window=7,
    raw=True,
    bprs_col="bprs",
    out_dir="participant_html"
):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([code_col, date_col])

    if variables is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        variables = [c for c in numeric_cols if c not in [code_col]]

    if ma_variables is None:
        ma_variables = [v for v in variables if v != bprs_col]

    os.makedirs(out_dir, exist_ok=True)

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
    color_map = {var: palette[i % len(palette)] for i, var in enumerate(variables)}

    for participant, sub in df.groupby(code_col):
        sub = sub.sort_values(date_col).copy()

        for var in ma_variables:
            if var in sub.columns:
                sub[f"{var}_ma"] = sub[var].rolling(window=ma_window, min_periods=1).mean()

        fig = go.Figure()

        for i, var in enumerate([v for v in variables if v in sub.columns]):
            axis_name = "y" if i == 0 else f"y{i+1}"
            axis_layout = "yaxis" if i == 0 else f"yaxis{i+1}"
            color = color_map[var]

            if raw:
                if var == bprs_col:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[date_col],
                            y=sub[var],
                            mode="markers+lines",
                            name=var,
                            marker=dict(size=10, symbol="diamond"),
                            line=dict(width=1.5, color=color),
                            yaxis=axis_name
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[date_col],
                            y=sub[var],
                            mode="lines",
                            name=f"{var} raw",
                            line=dict(width=1.2, color=color, dash="dot"),
                            opacity=0.45,
                            yaxis=axis_name
                        )
                    )

            if var != bprs_col and var in ma_variables:
                fig.add_trace(
                    go.Scatter(
                        x=sub[date_col],
                        y=sub[f"{var}_ma"],
                        mode="lines",
                        name=f"{var} MA({ma_window})",
                        line=dict(width=2.5, color=color),
                        yaxis=axis_name
                    )
                )

            if i == 0:
                fig.update_layout(**{
                    axis_layout: dict(title=dict(text=var, font=dict(color=color)), tickfont=dict(color=color))
                })
            else:
                fig.update_layout(**{
                    axis_layout: dict(
                        title=dict(text=var, font=dict(color=color)),
                        tickfont=dict(color=color),
                        overlaying="y",
                        side="right" if i % 2 == 1 else "left",
                        anchor="free",
                        position=min(0.85 + 0.05 * i, 1.0) if i % 2 == 1 else max(0.15 - 0.05 * i, 0.0)
                    )
                })

        fig.update_layout(
            template="plotly_white",
            title=f"Participant {participant}",
            xaxis=dict(title="Date"),
            hovermode="x unified",
            width=1300,
            height=700
        )

        filename = os.path.join(out_dir, f"participant_{participant}.html")
        fig.write_html(filename, auto_open=False)
        


def plot_mixedlm(
    df,
    x,
    y,
    participant,
    result,
    ax=None,
    n_points=100,
    alpha_points=0.25,
    alpha_lines=0.8,
    save_path=None
):
    sns.set_theme(style="white")

    plt.rcParams.update({
        # "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
    else:
        fig = ax.figure

    data = df[[x, y, participant]].replace([np.inf, -np.inf], np.nan).dropna()

    ids = np.sort(data[participant].unique())

    # Generate distinguishable grayscale values (avoid pure black)
    n_ids = len(ids)
    grays = np.linspace(0.25, 0.75, n_ids)  # 0=black, 1=white

    colors = {pid: str(gray) for pid, gray in zip(ids, grays)}

    x_grid = np.linspace(data[x].min(), data[x].max(), n_points)

    fixed_intercept = result.fe_params["Intercept"]
    fixed_slope = result.fe_params[x]

    # Participant-specific lines + scatter
    for pid in ids:
        d = data[data[participant] == pid]
        color = colors[pid]

        ax.scatter(
            d[x],
            d[y],
            s=16,
            color=color,
            alpha=alpha_points,
            linewidth=0,
            zorder=1
        )

        re = result.random_effects[pid]

        rand_intercept = re.get("Group", re.get("Intercept", re.get("const", 0)))
        rand_slope = re.get(x, 0)

        y_hat_i = (
            fixed_intercept + rand_intercept
            + (fixed_slope + rand_slope) * x_grid
        )

        ax.plot(
            x_grid,
            y_hat_i,
            color=color,
            linewidth=0.9,
            alpha=alpha_lines,
            zorder=2
        )

    # Fixed effect line (thick black)
    y_hat_fixed = fixed_intercept + fixed_slope * x_grid

    ax.plot(
        x_grid,
        y_hat_fixed,
        color="black",
        linewidth=2.5,
        zorder=5,
        label="Fixed effect"
    )

    # Axis styling (JAMA-like)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)

    ax.legend(frameon=False, fontsize=8, loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, ax
 