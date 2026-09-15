import numpy as np
import pandas as pd
import networkx as nx
import re

###############################################################################
## Survey-Network Analysis
###############################################################################

# FIXME: To which date does BPRS correspond? Timestamp panss or timestamp cgi?
# --> timestamp panss
def get_survey_df(path):
    
    # read in .csv file and set appropriate dtypes
    df = pd.read_csv(path)
    df = df.rename(columns={'record_id':'subject'})
    df = df.set_index('subject')
    df['timestamp_panss'] =  pd.to_datetime(df['timestamp_panss'])
    df['timestamp_cgi'] =  pd.to_datetime(df['timestamp_cgi'])
    
    # get helper columns
    df['date_bprs'] = df['timestamp_panss'].dt.date
    df['year_bprs'] = df['timestamp_panss'].dt.year
    df['week_bprs'] = df['timestamp_panss'].dt.isocalendar().week
    df['weekday_bprs'] = df['timestamp_panss'].dt.day_name()
    df['t'] = df['redcap_repeat_instance'].fillna(0)
    
    # get variable that assigns group
    df['group'] = df['redcap_event_name'].str.extract(r"(employees|patients)")
    
    # drop unneeded variables
    drop_columns = ['redcap_event_name','redcap_repeat_instrument']
    drop_columns.extend([
        'koerperbezogenheit', 'angst', 'zur_ckgezogenheit',
        'zerfall_denkprozesse', 'schuldgef_hle', 'gespanntheit',
        'manieriertheit', 'groessenideen', 'depressive_stimmung',
        'feindseligkeit', 'misstrauen', 'halluzinationen', 'verlangsamung',
        'unkooperative_verhalten', 'ungewoehnliche_denkinhalte',
        'affektive_abstumpfung', 'erregung', 'orientierungsstoerungen'
    ])
    drop_columns.append('bprs_complete')
    drop_columns.append('timestamp_cgi')
    drop_columns.append('timestamp_panss')
    drop_columns.append('redcap_repeat_instance')
    
    # rename certain variables
    df = df.rename(columns={"bprs_sum":"bprs"})
    
    
    df = df.drop(columns=drop_columns)
    
    return df

def get_meta_summary(df_meta):
    """
    Create demographic and diagnosis summary tables from metadata.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Must contain:
        - birth_date
        - start_study
        - female (0/1)
        - Dx

    Returns
    -------
    df_dx : pd.DataFrame
        Processed dataframe with age and primary_dx columns.
    summary : pd.DataFrame
        Demographic summary table.
    dx_summary : pd.DataFrame
        Diagnosis frequency summary table.
    """

    df_dx = df_meta.copy()

    # Dates
    df_dx["birth_date"] = pd.to_datetime(
        df_dx["birth_date"], errors="coerce"
    )
    df_dx["start_study"] = pd.to_datetime(
        df_dx["start_study"], errors="coerce"
    )

    # Age at study start
    df_dx["age"] = (
        df_dx["start_study"] - df_dx["birth_date"]
    ).dt.days / 365.25

    # Female: assumes 1 = female, 0 = male
    df_dx["female"] = pd.to_numeric(
        df_dx["female"], errors="coerce"
    )

    # Extract primary Dx: Fxx before any "ND ..."
    def extract_primary_dx(x):
        if pd.isna(x):
            return pd.NA

        x = str(x)

        # Remove secondary diagnoses
        x = re.split(r"\bND\b", x, flags=re.IGNORECASE)[0]

        match = re.search(
            r"f\d{2}(?:\.\d+)?",
            x,
            flags=re.IGNORECASE
        )

        return match.group(0).upper() if match else pd.NA

    df_dx["primary_dx"] = df_dx["Dx"].apply(extract_primary_dx)

    # Helper functions
    def mean_sd(x):
        return f"{x.mean():.1f} ({x.std():.1f})"

    def n_pct(x):
        n = int(x.sum())
        pct = 100 * x.mean()
        return f"{n} ({pct:.1f})"

    # Demographic summary
    summary = pd.DataFrame({
        "Characteristic": [
            "Patients, No.",
            "Age, mean (SD), y",
            "Female sex, No. (%)"
        ],
        "Total": [
            len(df_dx),
            mean_sd(df_dx["age"].dropna()),
            n_pct(df_dx["female"].dropna())
        ]
    })

    # Diagnosis summary
    dx_counts = (
        df_dx["primary_dx"]
        .value_counts(dropna=False)
        .reset_index()
    )

    dx_counts.columns = ["Characteristic", "n"]

    dx_counts["Characteristic"] = (
        "Diagnosis " +
        dx_counts["Characteristic"].fillna("Unknown").astype(str)
    )

    dx_counts["Total"] = dx_counts["n"].apply(
        lambda n: f"{n} ({100 * n / len(df_dx):.1f})"
    )

    dx_summary = dx_counts[["Characteristic", "Total"]]

    return df_dx, summary, dx_summary
def fill_missing_dates(group):
    pid = group.name  # the current participant code
    
    # remove code from the group so we don't depend on it being present
    group = group.drop(columns="code", errors="ignore")
    group['date'] = pd.to_datetime(group['date']).dt.date
    # full daily date range for this participant
    full_dates = pd.DataFrame({
        "date": pd.date_range(group["date"].min(), group["date"].max(), freq="D")
    })
    full_dates['date'] = pd.to_datetime(full_dates['date']).dt.date
    # merge only on date
    out = full_dates.merge(group, on="date", how="left")
    
    # add code back as a normal column
    out["code"] = pid
    
    return out

def build_average_graph(adj_dict, patient_ids = {12,18,2,19,34,37,13,7,51,53,56,50,52,57,43,44,3,8,99,85,
                9,83,61,62,71,73,74,79,80,81,84,91,92,94,95,96,97,98}, 
                        threshold_staff=0.2, 
                        threshold_mixed=0.05,
                        threshold_patient=0.01):
    """
    adj_dict: dict[date -> adjacency matrix (numpy array)]
    patient_ids: list or set of patient node indices
    thresholds:
        staff-staff edges use threshold_staff
        patient-patient edges use threshold_patient
        mixed edges use threshold_mixed
    """

    # --- stack matrices ---
    matrices = list(adj_dict.values())
    A_avg = np.mean(matrices, axis=0)
    A_avg = pd.DataFrame(data=A_avg,    # values
             index=matrices[0].index,    # 1st column as index
             columns=matrices[0].index)
    n = A_avg.shape[0]
    G = nx.Graph()

    # --- add nodes with type ---
    for i in matrices[0].index:
        node_type = "patient" if i in patient_ids else "staff"
        G.add_node(i, role=node_type)

    # --- add edges with type-specific thresholds ---
    for i in matrices[0].index:
        for j in matrices[0].index:
            if j != i:
                weight = A_avg.loc[i, j]
                if weight <= 0:
                    continue

                role_i = G.nodes[i]['role']
                role_j = G.nodes[j]['role']

                # choose threshold
                if role_i == "staff" and role_j == "staff":
                    threshold = threshold_staff
                elif role_i == "patient" and role_j == "patient":
                    threshold = threshold_patient
                else:
                    threshold = threshold_mixed

                if weight >= threshold:
                    G.add_edge(i, j, weight=weight)
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    return G, A_avg

def compute_graph_metrics(G):
    """
    Computes node-level and global metrics.
    """

    strength = dict(G.degree(weight='weight'))
    clustering = nx.clustering(G, weight='weight')
    centrality = nx.degree_centrality(G)

    # attach to graph
    nx.set_node_attributes(G, strength, "strength")
    nx.set_node_attributes(G, clustering, "clustering")
    nx.set_node_attributes(G, centrality, "centrality")

    summary = {
        "avg_strength": np.mean(list(strength.values())) if strength else 0,
        "avg_clustering": np.mean(list(clustering.values())) if clustering else 0,
        "avg_centrality": np.mean(list(centrality.values())) if centrality else 0,
    }

    return summary


# ============================================================
# GERMAN NEO-FFI-30 SCORING
# Assumes item columns are labeled 1..30 (or "1".."30" in CSV)
# ============================================================

# Factor definitions using ascending item labels 1..30
FACTOR_ITEMS = {
    "Neurotizismus":       np.arange(1,7),
    "Extraversion":        np.arange(7,13),
    "Offenheit":           np.arange(13,19),
    "Vertraeglichkeit":    np.arange(19,25),
    "Gewissenhaftigkeit":  np.arange(25,31),
}

# Reverse-coded items, using the same 1..30 labeling
REVERSE_ITEMS = {13, 15, 17, 19, 20,21, 22, 24, 30}

MIN_RESPONSE = 1
MAX_RESPONSE = 5


def reverse_code(series, min_val=1, max_val=5):
    """Reverse-code a Likert item."""
    return max_val + min_val - series


def score_neo_ffi_30(df, id_column=None, min_valid_items_per_factor=4):
    """
    Compute factor scores for the German NEO-FFI-30.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing item columns labeled 1..30.
        Column names may be integers (1, 2, ..., 30) or strings ("1", ..., "30").
    id_column : str or None
        Optional participant ID column to retain in the output.
    min_valid_items_per_factor : int
        Minimum number of non-missing items required to compute a factor score.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per participant and one factor score per factor.
    """
    data = df.copy()

    # Make sure item columns can be accessed as strings "1".."30"
    data.columns = [str(c) for c in data.columns]

    required_cols = [str(i) for i in range(1, 31)]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing item columns: {missing_cols}")

    # Convert item columns to numeric
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Check response range
    for col in required_cols:
        invalid = data[col].dropna()
        if not invalid.between(MIN_RESPONSE, MAX_RESPONSE).all():
            raise ValueError(f"Column {col} contains values outside the range {MIN_RESPONSE}-{MAX_RESPONSE}")

    # Create scored item columns
    for item in range(1, 31):
        col = str(item)
        scored_col = f"{col}_scored"

        if item in REVERSE_ITEMS:
            data[scored_col] = reverse_code(data[col], MIN_RESPONSE, MAX_RESPONSE)
        else:
            data[scored_col] = data[col]

    # Build output with factor scores
    output = pd.DataFrame(index=data.index)

    if id_column is not None:
        if str(id_column) not in data.columns:
            raise ValueError(f"ID column '{id_column}' not found in DataFrame")
        output[id_column] = data[str(id_column)]

    for factor, items in FACTOR_ITEMS.items():
        scored_cols = [f"{i}_scored" for i in items]

        valid_counts = data[scored_cols].notna().sum(axis=1)
        factor_score = data[scored_cols].mean(axis=1, skipna=True)

        # Set to missing if too few valid items
        factor_score = factor_score.where(valid_counts >= min_valid_items_per_factor, np.nan)

        output[factor] = factor_score

    return output
def attach_windows_to_bprs(df, bprs_events, windows):
    df_pred = df.drop(columns=["bprs"], errors="ignore").copy()
    df_pred["date"] = pd.to_datetime(df_pred["date"])

    bprs_events = bprs_events.copy()
    bprs_events["bprs_date"] = pd.to_datetime(bprs_events["bprs_date"])

    out = []

    for start_day, end_day in windows:
        tmp = df_pred.merge(
            bprs_events,
            on="code",
            how="inner"
        )

        tmp["rel_day"] = (
            tmp["date"] - tmp["bprs_date"]
        ).dt.days

        tmp = tmp.loc[
            tmp["rel_day"].between(start_day, end_day)
        ].copy()

        tmp["window_start"] = start_day
        tmp["window_end"] = end_day
        tmp["window"] = f"{start_day}_to_{end_day}"

        out.append(tmp)

    return pd.concat(out, ignore_index=True)
def make_bprs_windows(df_attached, agg_map):
    out = (
        df_attached
        .groupby(["code", "bprs_id", "window", "window_start", "window_end"], as_index=False)
        .agg(**agg_map)
        .sort_values(["code", "date", "window_start"])
    )

    n = (
        df_attached
        .groupby(["code", "bprs_id", "window", "window_start", "window_end"])
        ["date"]
        .count()
        .reset_index(name="n_instances")
    )

    out = out.merge(
        n,
        on=["code", "bprs_id", "window", "window_start", "window_end"],
        how="left"
    )

    return out
def plot_window_effects(
    p_vals,
    coef_col="coef",
    p_col="p",
    start_col="start_day",
    end_col="end_day",
    alpha=0.05,
    predictor="clus_c_z",
    ax=None,
):
    """
    Plot mixed model effect sizes across time windows.

    Parameters
    ----------
    p_vals : pd.DataFrame
        DataFrame containing coefficients and p-values.
    coef_col : str
        Column containing effect sizes.
    p_col : str
        Column containing p-values.
    start_col : str
        Window start day.
    end_col : str
        Window end day.
    alpha : float
        Significance threshold.
    predictor : str
        Predictor name for title.
    ax : matplotlib axis, optional
    """

    df = p_vals.copy()

    df["window_mid"] = (
        df[start_col] + df[end_col]
    ) / 2

    df = df.sort_values("window_mid")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sig = df[p_col] < alpha

    ax.plot(
        df[end_col],
        df[coef_col],
        marker="o",
        linewidth=2,
    )

    ax.scatter(
        df.loc[sig, end_col],
        df.loc[sig, coef_col],
        marker="*",
        s=120,
        label=f"p < {alpha}",
    )

    ax.axhline(0, linestyle="--", alpha=.7)

    ax.set_xlabel("Days relative to BPRS (last day of 7 day time window)")
    ax.set_ylabel(f"Coefficient ({predictor})")
    ax.set_title(f"Temporal effect of {predictor}")
    ax.legend()

    return ax
###############################################################################
## Archived / Potentially Deprecated
###############################################################################

# Hannes: Why do we need this? 
# Paul: Might use this later for quality control but do not use it atm
# "at_lunch": Idea: If somebody is not at lunch, which is mandatoiry, their sensor data fror that day might be faulty
def add_lunch(df,hour_of_interest,minutes_thr_lunch):

    df = df.copy()
    
    # create helper columns
    df["contact_day"] = df["contact_start"].dt.floor("D")
    df['contact_duration_minutes'] = df["contact_duration"].dt.total_seconds() / 60
    
    # lunch window: contacts starting between 12:00 and 12:59 (exclude battery)
    contact_lunch_candidate = (df["contact_start"].dt.hour == hour_of_interest) & (df["id_tgt"] != 0)
    
    contact_lunch_src = (
        df.loc[contact_lunch_candidate]
          .groupby(["id_src", "contact_day"])["contact_duration_minutes"]
          .sum()
          .gt(minutes_thr_lunch)
    )
    contact_lunch_src.name = "id_src_joined_lunch"
    
    contact_lunch_tgt = (
        df.loc[contact_lunch_candidate]
          .groupby(["id_tgt", "contact_day"])["contact_duration_minutes"]
          .sum()
          .gt(minutes_thr_lunch)
    )
    contact_lunch_tgt.name = "id_tgt_joined_lunch"
    
    df = df.merge(contact_lunch_src, left_on=["id_src", "contact_day"], right_index=True, how="left")
    df = df.merge(contact_lunch_tgt, left_on=["id_tgt", "contact_day"], right_index=True, how="left")
    
    df["id_src_joined_lunch"] = df["id_src_joined_lunch"].astype("boolean").fillna(False)
    df["id_tgt_joined_lunch"] = df["id_tgt_joined_lunch"].astype("boolean").fillna(False)
    
    # drop helper columns
    df = df.drop(columns=["contact_day","contact_duration_minutes"])

    return df
def mixedlm_r2(model, data):
    """
    Compute Nakagawa-style marginal and conditional R²
    for a random-intercept statsmodels MixedLM.

    Parameters
    ----------
    model : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted MixedLM result.
    data : pandas.DataFrame
        Data used for prediction.

    Returns
    -------
    dict
        {
            "marginal_r2": float,
            "conditional_r2": float,
            "var_fixed": float,
            "var_random": float,
            "var_residual": float,
        }
    """
    # Fixed-effects predictions
    fixed_pred = model.predict(data)

    # Variance components
    var_fixed = np.var(fixed_pred, ddof=1)
    var_random = model.cov_re.iloc[0, 0]
    var_residual = model.scale

    total_var = var_fixed + var_random + var_residual

    return {
        "marginal_r2": var_fixed / total_var,
        "conditional_r2": (var_fixed + var_random) / total_var,
        "var_fixed": var_fixed,
        "var_random": var_random,
        "var_residual": var_residual,
    }

###############################################################################
## Plotting
###############################################################################


    

def plot_mixed_model(model, result, df, predictor, outcome):
    # -------------------------
    # Extract fixed effects
    # -------------------------
    fixed_intercept = result.fe_params["Intercept"]
    fixed_slope = result.fe_params[predictor]

    conf = result.conf_int()
    intercept_ci = conf.loc["Intercept"]
    slope_ci = conf.loc[predictor]

    random_effects = result.random_effects

    # -------------------------
    # Styling
    # -------------------------
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # participant colors
    codes = df["code"].unique()
    palette = sns.color_palette("husl", len(codes))
    color_dict = dict(zip(codes, palette))

    # x-range
    x_vals = np.linspace(df[predictor].min(), df[predictor].max(), 200)

    fig, ax = plt.subplots(figsize=(8,6))

    # -------------------------
    # Scatter data
    # -------------------------
    for code in codes:
        sub = df[df["code"] == code]

        ax.scatter(
            sub[predictor],
            sub[outcome],
            color=color_dict[code],
            alpha=0.25,
            s=35
        )

    # -------------------------
    # Participant regression lines
    # -------------------------
    for code in codes:

        re = random_effects[code]

        intercept = fixed_intercept + re["Group"]
        slope = fixed_slope + re[predictor]

        y_vals = intercept + slope * x_vals

        ax.plot(
            x_vals,
            y_vals,
            color=color_dict[code],
            linewidth=1.8,
            alpha=0.8
        )

    # -------------------------
    # Population regression line
    # -------------------------
    y_mean = fixed_intercept + fixed_slope * x_vals

    ax.plot(
        x_vals,
        y_mean,
        color="black",
        linewidth=4,
        label="Population effect"
    )

    # -------------------------
    # Confidence band (fixed effect)
    # -------------------------
    # y_low = intercept_ci[0] + slope_ci[0] * x_vals
    # y_high = intercept_ci[1] + slope_ci[1] * x_vals

    # ax.fill_between(
    #     x_vals,
    #     y_low,
    #     y_high,
    #     color="black",
    #     alpha=0.15,
    #     label="95% CI (fixed effect)"
    # )

    # -------------------------
    # Labels and layout
    # -------------------------
    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
    
def plot_graph(G, figsize=(8, 8), layout_seed=42):
    """
    Creates a clean, publication-style network plot.
    """

    plt.figure(figsize=figsize)

    # --- layout ---
    pos = nx.spring_layout(G, seed=layout_seed, k=0.3)

    # --- node styling ---
    patient_nodes = [n for n, d in G.nodes(data=True) if d['role'] == 'patient']
    staff_nodes = [n for n, d in G.nodes(data=True) if d['role'] == 'staff']

    # soft, professional palette
    patient_color = "#4C72B0"  # muted blue
    staff_color = "#DD8452"    # muted orange

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=patient_nodes,
        node_color=patient_color,
        node_size=120,
        alpha=0.9,
        label="Patients"
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=staff_nodes,
        node_color=staff_color,
        node_size=120,
        alpha=0.9,
        label="Staff"
    )

    # --- edge styling ---
    edges = G.edges(data=True)
    weights = np.array([d['weight'] for (_, _, d) in edges])

    if len(weights) > 0:
        # normalize weights
        w_min, w_max = weights.min(), weights.max()
        norm_weights = (weights - w_min) / (w_max - w_min + 1e-9)

        widths = 0.5 + 4 * norm_weights
        alphas = 0.2 + 0.6 * norm_weights
    else:
        widths = []
        alphas = []

    for ((u, v, d), w, a) in zip(edges, widths, alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=w,
            alpha=a,
            edge_color="#4a4a4a"
        )

    # --- labels (optional, often removed for publication clarity) ---
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # --- styling ---
    plt.title("Average Daily Interaction Network", fontsize=14)
    plt.legend(frameon=False)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def lighten_color(color, amount=0.5):
    """
    Lighten a color by blending it toward white.
    amount=0 -> original color
    amount=1 -> white
    """
    c = np.array(to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)


def _get_fe_params(result, x_col):
    """Extract fixed intercept and slope from a statsmodels MixedLM result."""
    fe = result.fe_params

    # intercept
    if "Intercept" in fe.index:
        intercept = fe["Intercept"]
    elif "const" in fe.index:
        intercept = fe["const"]
    else:
        intercept = 0.0

    # slope
    if x_col in fe.index:
        slope = fe[x_col]
    else:
        raise ValueError(f"Could not find fixed slope for '{x_col}' in result.fe_params")

    return intercept, slope


def _get_re_intercept_slope(re_series, x_col):
    """
    Extract random intercept and slope from one group's random effects.
    Works for common MixedLM naming patterns.
    """
    idx = list(re_series.index)

    # random intercept candidates
    intercept_candidates = ["Intercept", "const", "Group", "Group Var"]
    re_intercept = 0.0
    for cand in intercept_candidates:
        if cand in re_series.index:
            re_intercept = re_series[cand]
            break

    # if no obvious intercept name, fall back carefully
    if re_intercept == 0.0:
        # if the random effects only contain one term and it's not x_col,
        # that is likely the intercept
        if len(idx) == 1 and idx[0] != x_col:
            re_intercept = re_series.iloc[0]
        elif len(idx) > 1:
            non_x = [k for k in idx if k != x_col and "Var" not in k]
            if len(non_x) > 0:
                re_intercept = re_series[non_x[0]]

    # random slope
    re_slope = re_series[x_col] if x_col in re_series.index else 0.0

    return re_intercept, re_slope

def plot_mixed_interaction_subsets(
    df_low,
    df_high,
    res_low,
    res_high,
    x="clus_c",
    y="bprs",
    group="code",
    low_label="Low Salience Network Within-Network Connectivity",
    high_label="Salience Network Within-Network Connectivity",
    low_color="#0072B2",   # Okabe-Ito blue
    high_color="#D55E00",  # Okabe-Ito vermillion
    figsize=(10, 7),
    scatter_size=36,
    scatter_alpha=0.45,
    random_line_alpha=0.75,
    random_line_width=1.2,
    fixed_line_width=4.0,
    dpi=300,
):
    """
    Plot two mixed-model fits on the same axes:
    - bold fixed-effect line for each subset
    - light scatter for observed data
    - light random-effects lines for each group ('code')
    """

    # --- style ---
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Clean publication-style background
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    # Combined x-range for comparability
    x_min = min(df_low[x].min(), df_high[x].min())
    x_max = max(df_low[x].max(), df_high[x].max())
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1)
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 200)

    def draw_subset(df, result, base_color, label):
        # Fixed effects
        fe_intercept, fe_slope = _get_fe_params(result, x)
        y_fixed = fe_intercept + fe_slope * x_grid

        # Unique group shades
        groups = pd.Index(df[group].dropna().unique())
        groups = groups.sort_values()

        # spread shades from moderately light to very light
        shade_levels = np.linspace(0.45, 0.78, max(len(groups), 2))
        color_map = {
            g: lighten_color(base_color, amount=shade_levels[i if len(groups) > 1 else 0])
            for i, g in enumerate(groups)
        }

        # Scatter by group
        for g in groups:
            dfg = df[df[group] == g]
            ax.scatter(
                dfg[x],
                dfg[y],
                s=scatter_size,
                color=color_map[g],
                edgecolor="none",
                alpha=scatter_alpha,
                zorder=2,
            )

        # Random effects lines by group
        re_dict = result.random_effects
        for g in groups:
            if g not in re_dict:
                continue

            re_intercept, re_slope = _get_re_intercept_slope(re_dict[g], x)
            y_group = (fe_intercept + re_intercept) + (fe_slope + re_slope) * x_grid

            ax.plot(
                x_grid,
                y_group,
                color=color_map[g],
                linewidth=random_line_width,
                alpha=random_line_alpha,
                zorder=1,
            )

        # Fixed effect line
        ax.plot(
            x_grid,
            y_fixed,
            color=base_color,
            linewidth=fixed_line_width,
            label=label,
            zorder=3,
        )

    # Draw both subsets
    draw_subset(df_low, res_low, low_color, low_label)
    draw_subset(df_high, res_high, high_color, high_label)

    # Labels
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=low_color, lw=fixed_line_width, label=low_label),
        Line2D([0], [0], color=high_color, lw=fixed_line_width, label=high_label),
        Line2D([0], [0], color="0.6", lw=1.5, alpha=0.8, label="Random effects"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.75",
               markersize=7, alpha=0.7, label="Observed data"),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc="best")

    plt.tight_layout()
    return fig, ax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mixedlm_participant_lines(
    df,
    model_fit,
    predictor="clus_c_z",
    outcome="bprs",
    group_col="code",
    figsize=(7.2, 5.2),
    show_points=True,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sns.set_style("white")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    b0 = model_fit.fe_params["Intercept"]
    b1 = model_fit.fe_params[predictor]

    x = np.linspace(df[predictor].min(), df[predictor].max(), 100)

    # fig, ax = plt.subplots(figsize=figsize)

    # Participant-specific lines: fixed slope + random intercept
    for pid, re in model_fit.random_effects.items():
        intercept_i = b0 + re.iloc[0]
        ax.plot(
            x,
            intercept_i + b1 * x,
            color="#BDBDBD",
            lw=0.8,
            alpha=0.35,
            zorder=1
        )

    # Raw observations
    if show_points:
        ax.scatter(
            df[predictor],
            df[outcome],
            s=12,
            color="#4D4D4D",
            alpha=0.15,
            zorder=2
        )

    # Fixed-effect / population line
    ax.plot(
        x,
        b0 + b1 * x,
        color="#003A70",
        lw=3.5,
        zorder=3,
        label="Population estimate"
    )

    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    return fig, ax

def plot_mixedlm_forest(
    model_fit,
    outcome_label="BPRS",
    figsize=(7, 5),
    title="Mixed-effects model estimates",
    term_labels=None,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sns.set_style("white")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    coefs = (
        pd.DataFrame({
            "term": model_fit.fe_params.index,
            "beta": model_fit.fe_params.values,
            "lower": model_fit.conf_int().loc[model_fit.fe_params.index, 0],
            "upper": model_fit.conf_int().loc[model_fit.fe_params.index, 1],
        })
        .query("term != 'Intercept'")
        .sort_values("beta")
    )

    if term_labels is not None:
        coefs["term"] = coefs["term"].replace(term_labels)

    # fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        coefs["beta"],
        coefs["term"],
        xerr=[
            coefs["beta"] - coefs["lower"],
            coefs["upper"] - coefs["beta"]
        ],
        fmt="o",
        color="#003A70",
        ecolor="#003A70",
        elinewidth=1.5,
        markersize=5,
        capsize=3,
        capthick=1.2,
        zorder=3
    )

    ax.axvline(
        0,
        color="#7A7A7A",
        lw=1,
        ls="--",
        zorder=1
    )

    ax.set_xlabel(f"Standardized effect on {outcome_label}")
    ax.set_ylabel("")
    ax.set_title(title)

    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax, coefs
def plot_mixedlm_figure(
    df,
    model_fit,
    predictor="clus_c_z",
    outcome="bprs",
    group_col="code",
    term_labels=None,
):

    sns.set_style("white")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(12, 5))

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.3],
        wspace=0.30
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plot_mixedlm_forest(
        model_fit=model_fit,
        outcome_label="BPRS",
        term_labels=term_labels,
        ax=ax1
    )

    plot_mixedlm_participant_lines(
        df=df,
        model_fit=model_fit,
        predictor=predictor,
        outcome=outcome,
        group_col=group_col,
        ax=ax2
    )

    # Panel labels
    ax1.text(
        -0.15, 1.03, "A",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold"
    )

    ax2.text(
        -0.12, 1.03, "B",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig, (ax1, ax2)