import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.sensors_figures import plot_network_time_slices
from utils.sensors_figures import plot_one_html_per_participant
from utils.sensors_figures import plot_mixedlm
from utils.sensors_survey_network import get_survey_df
import pickle
import datetime as dt
import networkx as nx
import statsmodels.formula.api as smf
from utils.sensors_survey_network import fill_missing_dates


with open('../data/exchange/matrix_dict_all.pickle', 'rb') as handle:
    matrix_dict = pickle.load(handle)
path_survey = '../data/survey/SoSenseStudieZurUnte-Bprs_DATA_2024-08-30_1102.csv'
df_dyads = pd.read_pickle('../data/exchange/df_contacts.pkl')
df_network = pd.read_pickle('../data/exchange/df_network0.pkl')
df_bprs = pd.read_pickle('../data/exchange/df_bprs.pkl')

df_network = df_network[~df_network["code"].str.startswith("therapist")]
df_dyads = df_dyads.loc[df_dyads['id_tgt']!=0]
df_network['study_start'] = df_network.groupby('code')['date'].transform('min')
df_network['study_end'] = df_network.groupby('code')['date'].transform('max')
df_network['time'] = pd.to_timedelta((df_network['date']-df_network['study_start'])).dt.days
df_network['dyadic']= df_network['dyadic_interaction_staff_minutes']+df_network['dyadic_interaction_patient_minutes']
df_network['group']= df_network['group_interaction_staff_minutes']+df_network['group_interaction_no_staff_minutes']
df_network['dyad_ratio'] = df_network['dyadic']/(df_network['group']+df_network['dyadic'])
df_network['dyad_ratio_pat'] = df_network['dyadic_interaction_patient_minutes']/(df_network['group']+df_network['dyadic'])
df_network['group_ratio'] = df_network['group']/(df_network['group']+df_network['dyadic'])
df_network['group_pat_ratio'] = df_network['group_interaction_no_staff_minutes']/df_network['contact_span_minutes']
df_network['mean_stren'] = df_network.groupby('date')['strengths'].transform('mean')
df_network['clus'] = df_network['clustering-coefficient']
df_network['cent'] = df_network['eigenvector-centrality']
df_network['staff_ratio']= (df_network['group_interaction_staff_minutes']+df_network['dyadic_interaction_staff_minutes'])/(df_network['contact_span_minutes'])

# filter by quality criteria
df_network = df_network.loc[df_network['ids_present']>=8]
df_network = df_network.loc[df_network['ids_present_pat_x']>=3]
df_network = df_network.loc[df_network['contact_span_minutes']>60]

# %%
df_filled = (
    df_network.groupby("code", group_keys=False)
      .apply(fill_missing_dates)
      .reset_index(drop=True)
)
df_filled = df_filled.sort_values(["code", "date"])
df_filled['clus'] =  df_filled['clustering-coefficient']
df_filled['cent'] =  df_filled['eigenvector-centrality']
df_filled['cent_pat'] =  df_filled['eigenvector-centrality_pat']

df_filled['conc_pat'] =  df_filled['partner-concentration_pat']
df_filled['conc'] =  df_filled['partner-concentration']


exclude = ["id", "code", "date"]
cols = [c for c in df_filled.columns if c not in exclude]

df_filled[[f"{c}_prev" for c in cols]] = (
    df_filled.groupby("code")[cols].shift(1)
)

df_filled = df_filled[~df_filled["code"].str.startswith("therapist")]
df_network["date"] = pd.to_datetime(df_network["date"]).dt.date


df_filled['date'] = pd.to_datetime(df_filled['date']).dt.date
df_survey = get_survey_df(path_survey)
df_survey = df_survey.reset_index()
df_survey.rename(columns={'subject':'code','date_bprs':'date'}, inplace=True)
df_survey['date'] = pd.to_datetime(df_survey['date']).dt.date

df_merged = pd.merge(df_filled, df_survey, on = ['code','date'], how='left')
sns.set_theme(style="white")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Edit this list: each histogram gets its own dataframe + variable
plot_specs = [
    {"var": "time_of_day", "df": df_dyads,   "source": "Dyads",   "log": False},
    {"var": "duration_minutes", "df": df_dyads,   "source": "Dyads",   "log": True},
    # {"var": "staff_ratio", "df": df_network,   "source": "Network",   "log": False},
    # {"var": "dyad_ratio", "df": df_network, "source": "Network", "log": False},
    # {"var": "partner-concentration", "df": df_network, "source": "Network", "log": False},
]

panel_labels = ["A", "B", "C"]
hatches = ["", "//", "\\\\", "xx", ".."]

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 2.4)
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.10,
    top=0.92,
    wspace=0.35,
    hspace=0.45
)

axes = axes.flatten()

# Panel A: illustration placeholder
ax = axes[0]
ax.text(
    0.5, 0.5,
    "Illustration\nadded later",
    ha="center",
    va="center",
    fontsize=8
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Illustration", pad=4)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("black")

ax.text(
    -0.12, 1.08, panel_labels[0],
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top"
)

# Histogram panels
for i, spec in enumerate(plot_specs):
    ax = axes[i + 1]

    var = spec["var"]
    df = spec["df"]
    use_log = spec["log"]
    source = spec["source"]

    data = df[var].replace([np.inf, -np.inf], np.nan).dropna()

    if use_log:
        data = data[data > 0]

        if data.empty or data.nunique() < 2:
            ax.text(
                0.5, 0.5,
                "Insufficient\npositive data",
                ha="center",
                va="center",
                fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(var.replace("_", " ").title() + " (log scale)", pad=4)
            continue

        xmin = data.min()
        xmax = data.max()

        bins = np.logspace(np.log10(xmin), np.log10(xmax), 20)

        sns.histplot(
            x=data,
            bins=bins,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    else:
        if data.empty:
            ax.text(
                0.5, 0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(var.replace("_", " ").title(), pad=4)
            continue

        sns.histplot(
            x=data,
            bins=20,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )

    for patch in ax.patches:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_hatch(hatches[i])

    title = var.replace("_", " ").title()
    if use_log:
        title += " (log scale)"

    ax.set_title(title, pad=4)
    ax.set_xlabel(source)
    ax.set_ylabel("No. of observations")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", length=3, width=0.8)

    ax.text(
        -0.12, 1.08, panel_labels[i + 1],
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top"
    )

# plt.savefig("jama_style_mixed_df_figure.pdf", bbox_inches="tight")
# plt.savefig("jama_style_mixed_df_figure.tiff", dpi=600, bbox_inches="tight")
# plt.show()

# G0 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,5,30)])
# isolates = list(nx.isolates(G0))
# if isolates:
#     G0.remove_nodes_from(isolates)
# G1 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,5,31)])
# isolates = list(nx.isolates(G1))
# if isolates:
#     G1.remove_nodes_from(isolates)
G2 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,1)])
isolates = list(nx.isolates(G2))
if isolates:
    G2.remove_nodes_from(isolates)
G3 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,2)])
isolates = list(nx.isolates(G3))
if isolates:
    G3.remove_nodes_from(isolates)
G4 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,5)])
isolates = list(nx.isolates(G4))
if isolates:
    G4.remove_nodes_from(isolates)
G5 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,6)])
isolates = list(nx.isolates(G5))
if isolates:
    G5.remove_nodes_from(isolates)
G6 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,7)])
isolates = list(nx.isolates(G6))
if isolates:
    G6.remove_nodes_from(isolates)
# G7 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,8)])
# isolates = list(nx.isolates(G7))
# if isolates:
#     G7.remove_nodes_from(isolates)
fig, ax = plot_network_time_slices(
    graphs=[G2, G3, G4, G5, G6],
    patient_nodes=[12,18,2,19,34,37,13,7,51,53,56,50,52,57,43,44,3,8,99,85,
                     9,83,61,62,71,73,74,79,80,81,84,91,92,94,95,96,97,98],
    titles=["-7 days","-4 days", "-3 days", "-2 days","-1 day"],
    save_path="../figures/network_sheets_jama.tiff"
)

plot_one_html_per_participant(
    df_merged,
    code_col="code",
    date_col="date",
    variables=["clus","bprs"],
    ma_variables=["clus"],
    ma_window=7,
    bprs_col="bprs"
)
md1 = smf.mixedlm("bprs ~ clus_c_z+ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1+clus_c_z', missing='drop') 
mdf1 = md1.fit(reml = True)
fig, ax = plot_mixedlm(
    df=df_bprs,
    x="clus_c_z",
    y="bprs",
    participant="code",
    result=mdf1,
    save_path="mixedlm_jama_plot.tiff"
)