import os
import sys
import numpy as np
import pandas as pd
from utils.settings import data_dir
from utils.sensors_survey_network import get_survey_df
from utils.sensors_survey_network import get_meta_summary
from utils.sensors_survey_network import fill_missing_dates
from utils.sensors_survey_network import plot_mixedlm_participant_lines
from utils.sensors_survey_network import attach_windows_to_bprs
from utils.sensors_survey_network import make_bprs_windows
from utils.sensors_survey_network import mixedlm_r2
from utils.sensors_survey_network import plot_mixedlm_figure
from utils.sensors_survey_network import plot_window_effects
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from scipy.stats import iqr
# from analysis_sensor_network_measures import df_network_long as df_network
from preprocessing_sensors import df_cleaned as df_dyads


# get name of script
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# set output directory
if not os.path.isdir(f"../output/figures/{script_name}"):
    os.makedirs(f"../output/figures/{script_name}")
    
# patients with SSD or no psychosis at all
no_psy = np.array([7,52,99,71,79,80,81,84,92,94,95])
ssd = np.array([18,2,13,51,56,50,3,8,9,98])

# path to survey data
path_survey = os.path.join(data_dir,'survey','SoSenseStudieZurUnte-Bprs_DATA_2024-08-30_1102.csv')
# path_meta = os.path.join(data_dir,'survey','sosense_dx.csv')
path_meta = os.path.join('../data','survey','sosense_dx.csv')
path_orbis = '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/data/survey/20260623_MNST_Station_Station Tag_01042023bis3006204.xlsx'
path_psp = os.path.join(data_dir,'survey','SoSenseStudieZurUnte-Psp_DATA_2024-08-30_1103.csv')
path_panss = os.path.join(data_dir,'survey','SoSenseStudieZurUnte-Panss_DATA_2024-06-14_1609.csv')
# matrix threshold
thr = 0

###############################################################################
## Load data
###############################################################################

# get PANSS survey
df_panss = pd.read_csv(path_panss).rename(columns={'record_id': 'code','panss_sum':'panss'})
df_panss['date'] = pd.to_datetime(df_panss['timestamp_panss']).dt.date

# get bprs survey
df_survey = get_survey_df(path_survey)
df_survey = df_survey.reset_index()
df_survey.rename(columns={'subject':'code','date_bprs':'date'}, inplace=True)

# load metadata
df_meta = pd.read_csv(path_meta)

# daily stats in long format, including overall summary stats and network measures
df_network = pd.read_pickle("../data/exchange/df_network" + str(thr) +".pkl")

# for later iterating over different time windows
p_vals = []

###############################################################################
## Prepare analysis
###############################################################################

df_network['study_start'] = df_network.groupby('code')['date'].transform('min')
df_network['study_end'] = df_network.groupby('code')['date'].transform('max')
df_network['time'] = pd.to_timedelta((df_network['date']-df_network['study_start'])).dt.days
df_network['dyadic']= df_network['dyadic_interaction_staff_minutes']+df_network['dyadic_interaction_patient_minutes']
df_network['group']= df_network['group_interaction_staff_minutes']+df_network['group_interaction_no_staff_minutes']
df_network['dyad_ratio'] = df_network['dyadic']/(df_network['group']+df_network['dyadic'])
df_network['dyad_ratio_pat'] = df_network['dyadic_interaction_patient_minutes']/(df_network['group']+df_network['dyadic'])
df_network['rel_dyad'] = df_network['dyadic']/df_network['contact_span_minutes']
df_network['dyad_index'] = (df_network['dyadic']-df_network['group'])/(df_network['group']+df_network['dyadic'])
df_network['group_ratio'] = df_network['group']/(df_network['group']+df_network['dyadic'])
df_network['group_pat_ratio'] = df_network['group_interaction_no_staff_minutes']/df_network['contact_span_minutes']
df_network['dyad_pat_ratio'] = df_network['dyadic_interaction_patient_minutes']/df_network['contact_span_minutes']
df_network['dt'] = pd.to_datetime(df_network['date'])
df_network["calendar_week"] = df_network["dt"].dt.isocalendar().week
df_network["calendar_year"] = df_network["dt"].dt.isocalendar().year
df_network['mean_stren'] = df_network.groupby('date')['strengths'].transform('mean')
df_network['mean_clus'] = df_network.groupby('date')['clustering-coefficient'].transform('mean')
df_network['clus'] = df_network['clustering-coefficient']
df_network['cent'] = df_network['eigenvector-centrality']
df_network['rssi_comm_in'] = df_network['rssi_comm_in'] + 90
df_network['rssi_comm_out'] = df_network['rssi_comm_out'] + 90
df_network['rssi_ratio'] = df_network['rssi_comm_in']/df_network['rssi_comm_out']
df_network['rssi_comm_in_pat'] = df_network['rssi_comm_in_pat'] + 90
df_network['rssi_comm_out_pat'] = df_network['rssi_comm_out_pat'] + 90
df_network['rssi_ratio_pat'] = df_network['rssi_comm_in_pat']/df_network['rssi_comm_out_pat']
df_network['rel_staff']= (df_network['group_interaction_staff_minutes']+df_network['dyadic_interaction_staff_minutes'])/(df_network['contact_span_minutes'])
df_network['abs_staff']= df_network['group_interaction_staff_minutes']+df_network['dyadic_interaction_staff_minutes']
df_network['dyad_staff_ratio']= (df_network['dyadic_interaction_staff_minutes'])/(df_network['contact_span_minutes'])
df_network['group_staff_ratio']= (df_network['group_interaction_staff_minutes'])/(df_network['contact_span_minutes'])
df_network['rel_pat'] = df_network['dyad_pat_ratio'] + df_network['group_pat_ratio'] 
df_network['abs_pat'] = df_network['dyadic_interaction_patient_minutes'] + df_network['group_interaction_no_staff_minutes'] 
df_network['staff_index'] = (df_network['abs_staff']-df_network['abs_pat'])/(df_network['abs_staff']+df_network['abs_pat'])
df_network['staff_ratio'] = df_network['abs_staff']/(df_network['abs_staff']+df_network['abs_pat'])
df_network['ids_pres_staff'] = df_network['ids_present'] - df_network['ids_present_pat_x']
df_orbis = pd.read_excel(path_orbis)
df_orbis = df_orbis.iloc[39:,1:].reset_index(drop=True)
df_orbis.columns = df_orbis.iloc[0]
df_orbis = df_orbis.iloc[1:]
df_orbis = df_orbis.reset_index(drop=True)
df_orbis['date'] = pd.to_datetime(df_orbis['Tag']).dt.date
df_orbis['patients_present'] = df_orbis['Aufenthalt  um 12:00 Uhr']
df_orbis = df_orbis.loc[:,['date','patients_present']]

df_network = df_network.merge(df_orbis,on = 'date', how = 'left')
df_network['patient_sensor_ratio'] = df_network['ids_present_pat_x']/df_network['patients_present']
# sns.histplot(data = df_network, x = 'patient_sensor_ratio')

fig, ax1 = plt.subplots(figsize=(16, 10))

# Left y-axis
sns.lineplot(
    data=df_network,
    x="date",
    y="patient_sensor_ratio",
    ax=ax1,
    label="patient_sensor_ratio",
    color="tab:blue",
    marker="o",
    legend=None
)
ax1.set_ylabel("patient_sensor_ratio", color="tab:blue")

# Right y-axis
ax2 = ax1.twinx()

sns.lineplot(
    data=df_network,
    x="date",
    y="ids_present_pat_x",
    ax=ax2,
    label="ids_present_pat",
    color="tab:red",
    marker="o",
    legend=None
)
ax2.set_ylabel("ids_present_pat", color="tab:red")

ax1.set_xlabel("Time")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()



# filter by quality criteria
df_network = df_network.loc[df_network['ids_present']>=8]
df_network = df_network.loc[df_network['ids_present_pat_x']>=3]
df_network = df_network.loc[df_network['contact_span_minutes']>60]
# df_network = df_network.loc[df_network['patient_sensor_ratio']>0.4]


df_dyads = df_dyads.loc[df_dyads['id_tgt'] != 0]
# sns.histplot(data = df_network, x = 'patient_sensor_ratio')









df_network_incl_staff = df_network.copy()
df_network = df_network[~df_network["code"].str.startswith("therapist")]
#FIXME outer or left??
df_merged = pd.merge(df_network, df_survey, on = ['code','date'], how='outer')

##############################
# testing start
##############################
df = df_merged.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"])

# One row per BPRS assessment
bprs_events = (
    df.loc[df["bprs"].notna(), ["code", "date", "bprs", "weekday_bprs"]]
      .rename(columns={"date": "bprs_date", "weekday_bprs": "weekday"})
      .sort_values(["code", "bprs_date"])
      .reset_index(drop=True)
)

agg_map = {
    "bprs": ("bprs", "last"),
    "date": ("bprs_date", "last"),
    "weekday": ("weekday", "last"),
    "idx": ("id", "first"),

    "total_interaction": ("duration_minutes_sum", "mean"),
    "total_interaction_pat": ("duration_minutes_sum_pat", "mean"),
    "clus": ("clustering-coefficient", "mean"),
    "avg_clus": ("mean_clus", "mean"),
    "betweenness_centrality": ("betweenness_centrality", "mean"),
    "betweenness_centrality_pat": ("betweenness_centrality_pat", "mean"),
    "clus_pat": ("clustering-coefficient_pat", "mean"),
    "cent": ("eigenvector-centrality", "mean"),
    "rel_pat": ("rel_pat", "mean"),
    "rel_staff": ("rel_staff", "mean"),
    "rel_dyad": ("rel_dyad", "mean"),
    "stren": ("strengths", "mean"),
    "stren_pat": ("strengths_pat", "mean"),
    "ids_pres": ("ids_present", "mean"),
    "density": ("density", "mean"),
    "entropy": ("entropy", "mean"),
    "entropy_pat": ("entropy_pat", "mean"),
    "degree": ("degree", "mean"),
    "degree_pat": ("degree_pat", "mean"),
    "jac": ("weighted-jaccard-prev-day", "mean"),
    "jac_pat": ("weighted-jaccard-prev-day_pat", "mean"),
}

bprs_events["bprs_id"] = bprs_events.groupby("code").cumcount() + 1
windows = [
    (-14, -8),
    (-12, -6),
    (-10, -4),
    (-8, -2),
    (-6, 0),
    (-4, 2),
    (-2, 4),
    (2, 8),
    (4, 10),
]
df_attached = attach_windows_to_bprs(
    df=df,
    bprs_events=bprs_events,
    windows=windows
)
df_bprs_all_windows = make_bprs_windows(df_attached, agg_map)
p_vals = []

for window, df_bprs in df_bprs_all_windows.groupby("window"):

    df_bprs = df_bprs.copy()

    df_bprs["clus_c"] = (
        df_bprs["clus"] -
        df_bprs.groupby("code")["clus"].transform("mean")
    )

    df_bprs["clus_c_z"] = (
        df_bprs["clus_c"] - df_bprs["clus_c"].mean()
    ) / df_bprs["clus_c"].std()

    try:
        md = smf.mixedlm(
            "bprs ~ clus_c_z",
            df_bprs,
            groups=df_bprs["code"],
            re_formula="1",
            missing="drop"
        )

        mdf = md.fit(reml=True)

        p_vals.append({
            "window": window,
            "start_day": df_bprs["window_start"].iloc[0],
            "end_day": df_bprs["window_end"].iloc[0],
            "predictor": "clus_c_z",
            "coef": mdf.params.get("clus_c_z", np.nan),
            "p": mdf.pvalues.get("clus_c_z", np.nan),
            "n_obs": int(mdf.nobs),
            "n_participants": df_bprs["code"].nunique()
        })

    except Exception as e:
        p_vals.append({
            "window": window,
            "start_day": df_bprs["window_start"].iloc[0],
            "end_day": df_bprs["window_end"].iloc[0],
            "predictor": "clus_c_z",
            "coef": np.nan,
            "p": np.nan,
            "error": str(e),
            "n_obs": len(df_bprs),
            "n_participants": df_bprs["code"].nunique()
        })

p_vals = pd.DataFrame(p_vals)

plot_window_effects(
    p_vals,
    predictor="clus_c_z"
)

plt.tight_layout()
plt.show()

##############################
# testing end
##############################
 # for t_win in [-14,-12,-10,-8,-6,-4,-2,1,3,5,7]:
df_merged['new_bprs'] = ~np.isnan(df_merged['bprs'].shift())
 # df_merged['new_window'] = ~np.isnan(df_merged['bprs'].shift(t_win))
 # df_merged["window_id"] = df_merged.groupby(["code"])['new_window'].cumsum()

df_merged["bprs_id"] = df_merged.groupby(["code"])['new_bprs'].cumsum()
 # FIXME: make sure that no values of more than 7 days before are included
 # df_last7 = df_merged.groupby(['code','bprs_id']).tail(14)
df_merged['latest'] = df_merged.sort_values('date').groupby(['code','bprs_id'])['date'].transform('last')
df_merged['bprs'] = df_merged.sort_values('date').groupby(['code','bprs_id'])['bprs'].transform('last')
 # df_merged['bprs'] = df_merged.sort_values('date').groupby(['code','window_id'])['bprs'].transform('last')

 # compute differenceNeoFFI30
df_merged['diff_to_latest'] =pd.to_timedelta(df_merged['latest']  - df_merged['date']).dt.days
df_merged = df_merged.loc[df_merged['diff_to_latest'] <8]
df_merged = df_merged.loc[df_merged['diff_to_latest'] >= 1]
df_merged['n_instances'] = df_merged.sort_values('date').groupby(['code','bprs_id'])['date'].transform('count')
# df_merged = df_merged.loc[df_merged['n_instances'] > 1]

 # aggregate over the 7 day window  'calendar_year','calendar_week', 'bprs_id'
 #     
df_bprs = df_merged.groupby(['code','bprs_id'], as_index=False).agg(
         bprs = ('bprs','last'),
         idx = ('id','first'),
         date=("date", "last"),
         total_interaction=("duration_minutes_sum", "mean"),
         total_interaction_pat=("duration_minutes_sum_pat", "mean"),
         clus = ('clustering-coefficient','mean'),
         avg_clus = ('mean_clus','mean'),
         betweenness_centrality =  ('betweenness_centrality','mean'),
         betweenness_centrality_pat =   ('betweenness_centrality_pat','mean'),
         clus_pat = ('clustering-coefficient_pat','mean'),
         cent = ('eigenvector-centrality','mean'),
         rel_pat = ('rel_pat','mean'),
         rel_staff = ('rel_staff','mean'),
         rel_dyad = ('rel_dyad','mean'),

         stren = ('strengths','mean'),
         stren_pat = ('strengths_pat','mean'),
         dyadic_staff = ('dyadic_interaction_staff_minutes','mean'),
         dyadic_staff_dur = ('dyadic_interaction_staff_median_minutes','mean'),
         group_staff_dur = ('group_interaction_staff_median_minutes','mean'),
         dyadic_pat_dur = ('dyadic_interaction_patient_median_minutes','mean'),
         group_pat_dur = ('group_interaction_no_staff_median_minutes','mean'),

         dyadic_pat = ('dyadic_interaction_patient_minutes','mean'),
         group_pat = ('group_interaction_no_staff_minutes','mean'),
         group_staff = ('group_interaction_staff_minutes','mean'),
         staff_share = ('staff_interaction_share','mean'),
         cent_pat = ('eigenvector-centrality_pat','mean'),

         study_end = ('study_end','last'),
         study_start = ('study_start','last'),
         ids_pres = ('ids_present','mean'),
         dyad_ratio = ('dyad_ratio','mean'),
         dyad_index = ('dyad_index','mean'),
         rssi_comm_in = ('rssi_comm_in','mean'),
         rssi_comm_out = ('rssi_comm_out','mean'),
         rssi_ratio = ('rssi_ratio','mean'),
         rssi_comm_in_pat = ('rssi_comm_in_pat','mean'),
         rssi_comm_out_pat = ('rssi_comm_out_pat','mean'),
         rssi_ratio_pat = ('rssi_ratio_pat','mean'),

         group_ratio = ('group_ratio','mean'),
         staff_ratio = ('staff_ratio','mean'),
         staff_index = ('staff_index','mean'),

         dyad_ratio_pat = ('dyad_ratio_pat','mean'),
         # group_pat_ratio = ('group_pat_ratio','mean'),
         ids_pres_pat = ('ids_present_pat_x','mean'),
         ids_pres_staff = ('ids_pres_staff','mean'),

         time = ('time','last'),
         rel_int = ('duration_minutes_relative','mean'),
         density = ('density', 'mean'),

         mean_stren = ('mean_stren','mean'),
         entropy = ('entropy','mean'),
         entropy_pat = ('entropy_pat','mean'),
         iei = ('intercontact_interval_minutes_median','mean'),
         any_iei =('any_intercontact_interval_minutes_median','mean'),
         iei_pat = ('intercontact_interval_minutes_median_pat','mean'),
         any_iei_pat =('any_intercontact_interval_minutes_median_pat','mean'),

         any_burstiness = ('any_burstiness','mean'),
         any_burstiness_pat = ('any_burstiness_pat','mean'),

         dur = ('duration_minutes_median','mean'),
         dur_pat = ('duration_minutes_median_pat','mean'),

         any_dur = ('any_duration_minutes_median','mean'),
         any_dur_pat = ('any_duration_minutes_median_pat','mean'),

         rssi= ('rssi_mean_weighted','mean'),
         rssi_pat= ('rssi_mean_weighted_pat','mean'),

         conc = ('partner-concentration','mean'),
         conc_pat = ('partner-concentration_pat','mean'),
         burstiness = ('burstiness','mean'),
         burstiness_pat = ('burstiness_pat','mean'),
         dyad_staff_ratio = ('dyad_staff_ratio','mean'),
         group_staff_ratio = ('group_staff_ratio','mean'),
         dyad_pat_ratio = ('dyad_pat_ratio','mean'),
         group_pat_ratio = ('group_pat_ratio','mean'),

         degree = ('degree','mean'),
         degree_pat = ('degree_pat','mean'),
         contact_span = ('contact_span_minutes','mean'),
         weekday = ("weekday_bprs",'last'),
         jac = ('weighted-jaccard-prev-day','mean'),
         jac_pat = ('weighted-jaccard-prev-day_pat','mean'),
         rich_club = ('rich_club','mean'),
         k_core = ('k_core','mean'),
         rich_club_pat = ('rich_club_pat','mean'),
         k_core_pat = ('k_core_pat','mean'),
         comm_ratio = ('community_ratio','mean'),
         comm_ratio_pat = ('community_ratio_pat','mean'),
         modularity = ('modularity','mean'),
         modularity_pat = ('modularity_pat','mean'),
         n_comms = ('n_comms','mean'),
         n_comms_pat = ('n_comms_pat','mean')    
         )
# df_bprs['weekday'] = df_bprs['weekday'].astype('category')

df_bprs = df_bprs.sort_values(["code","date"])
exclude = ["id", "code", "date"]
num_cols = df_bprs.select_dtypes(include='number').columns

df_bprs[[f"{c}_prev" for c in num_cols]] = (
    df_bprs.groupby("code")[num_cols].shift(1)
)
df_bprs[[f"{c}_next" for c in num_cols]] = (
    df_bprs.groupby("code")[num_cols].shift(-1)
)
df_bprs[[f"{c}_next_next" for c in num_cols]] = (
    df_bprs.groupby("code")[num_cols].shift(-2)
)
df_bprs[[f"{c}_dx" for c in num_cols]] = (
    df_bprs[num_cols] - df_bprs.groupby('code')[num_cols].shift(1)
)

df_bprs = pd.merge(df_bprs, df_meta, on='code', how= 'left')
df_bprs['study_start'] =pd.to_datetime(df_bprs['start_study']).dt.date
df_bprs['date_mri']=pd.to_datetime(df_bprs['date_mri']).dt.date
df_bprs['age'] = (pd.to_datetime(df_bprs['study_start']) -pd.to_datetime(df_bprs['birth_date'])).dt.days/365
# df_bprs = pd.merge(df_bprs, df_panss, on = ['code','date'], how='left')
df_bprs = df_bprs.sort_values(["code", "date"])


# previous date per participant
df_bprs["prev_date"] = df_bprs.groupby("code")["date"].shift(1)
# means
num_cols = df_bprs.select_dtypes(include='number').columns
df_bprs[[f"{col}_mean" for col in num_cols]] = (
    df_bprs.groupby('code')[num_cols].transform('mean')
)
# z-scoring and centering
num_cols = df_bprs.select_dtypes(include='number').columns
df_bprs[[f"{col}_c" for col in num_cols]] = (
    df_bprs[num_cols] - df_bprs.groupby('code')[num_cols].transform('mean')
)


num_cols = df_bprs.select_dtypes(include='number').columns
df_bprs[[f"{col}_z" for col in num_cols]] = (
    (df_bprs[num_cols] - df_bprs[num_cols].mean()) / df_bprs[num_cols].std()
)


df_ssd = df_bprs.loc[df_bprs.idx.isin(ssd)]
df_psy = df_bprs.loc[~df_bprs.idx.isin(no_psy)]
df_bprs.to_csv('~/Desktop/df.csv')
df_bprs['weekday']=df_bprs['weekday'].astype('category')
df_bprs = df_bprs.drop_duplicates(subset=["code", "date"])
## this for later implementing a histogram clus_effect_size~time_window
# md2 = smf.mixedlm("bprs~ +clus_c_z+ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
# mdf2 = md2.fit(reml = True)
# p_vals.append({
#     'predictor': 'clus_c_z',
#     'win': t_win,
#     'p': mdf2.pvalues['clus_c_z'],
#     'coef': mdf2.params['clus_c_z'],
#     'n': mdf2.model.n_groups,
#     'outcome': 'bprs'
# })            

# p_vals = pd.DataFrame(p_vals)
###############################################################################
## Stats
###############################################################################

md1 = smf.mixedlm("bprs ~ rssi_comm_out_pat_c_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
#  individual
md1 = smf.mixedlm("bprs ~ dur_c_z+iei_c_z+dyad_ratio_c_z+rel_int_c_z+staff_ratio_c_z+ids_pres_z+female+age_z+time_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
# Get summary table
# table1 = mdf1.summary().tables[1].reset_index()

# Save
# table.to_csv("../data/exchange/indiv_summary.csv", index=False)
r2 = mixedlm_r2(mdf1, df_bprs)

print(f"Marginal R²: {r2['marginal_r2']:.3f}")
print(f"Conditional R²: {r2['conditional_r2']:.3f}")

print(f"Marginal R²: {r2['marginal_r2']:.3f}")
print(f"Conditional R²: {r2['conditional_r2']:.3f}")
term_labels = {
    "dur_c_z": "mean interaction duration",
    "iei_c_z": "inter-event interval",
    "dyad_ratio_c_z": "Dyadic Ratio",
    "rel_int_c_z": "Relative interaction",
    "staff_ratio_c_z": "Staff ratio",
    "ids_pres_z": "network size",
    "female": "Female sex",
    "age_z": "Age",
    "time_z": "Time",
}

fig, axes = plot_mixedlm_figure(
    df=df_bprs,
    model_fit=mdf1,
    predictor="dur_c_z",
    outcome="bprs",
    group_col="code",
    term_labels=term_labels
)

#     plt.ylim
# individual <--> network
md2 = smf.mixedlm("bprs ~ clus_c_z+cent_c_z+comm_ratio_c_z+rich_club_c_z+ids_pres_z+time_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf2 = md2.fit(reml = True)
print(mdf2.summary())
# Get summary table
# table2 = mdf2.summary().tables[1].reset_index()
# k-core?
r2 = mixedlm_r2(mdf2, df_bprs)

print(f"Marginal R²: {r2['marginal_r2']:.3f}")
print(f"Conditional R²: {r2['conditional_r2']:.3f}")
term_labels = {
    "clus_c_z": "Clustering coefficient",
    "cent_c_z": "Eigenvector centrality",
    "comm_ratio_c_z": "Community ratio",
    "rich_club_c_z": "Rich club",
    "k_core_c_z": "K-core",
    "ids_pres_z": "network size",
    "female": "Female sex",
    "age_z": "Age",
    "time_z": "Time",
}

fig, axes = plot_mixedlm_figure(
    df=df_bprs,
    model_fit=mdf2,
    predictor="clus_c_z",
    outcome="bprs",
    group_col="code",
    term_labels=term_labels
)

#     plt.ylim([18, 55]) (18 is BPRS minimum (18 items))
plt.show()
# Save
# table2.to_csv("../data/exchange/clus_summary.csv", index=False)
# network
md3 = smf.mixedlm("bprs ~ modularity_z+density_z +ids_pres_z +time_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf3 = md3.fit(reml = True)
print(mdf3.summary())
# Get summary table
# table3 = mdf3.summary().tables[1].reset_index()

term_labels = {
    "modularity_z": "mean interaction duration",
    "density_z": "inter-event interval",
    "ids_pres_z": "network size",
    "female": "Female sex",
    "age_z": "Age",
    "time_z": "Time",
}

fig, axes = plot_mixedlm_figure(
    df=df_bprs,
    model_fit=mdf3,
    predictor="modularity_z",
    outcome="bprs",
    group_col="code",
    term_labels=term_labels
)

# # Save
# table3.to_csv("../data/exchange/group_summary.csv", index=False)


r2 = mixedlm_r2(mdf3, df_bprs)

print(f"Marginal R²: {r2['marginal_r2']:.3f}")
print(f"Conditional R²: {r2['conditional_r2']:.3f}")

df_bprs.to_pickle('../data/exchange/df_bprs.pkl')
df_bprs.to_csv('../data/exchange/df_bprs.csv')

#
####################################################################################
# df slopes
####################################################################################
md4 = smf.mixedlm("bprs ~ clus_c_z+ids_pres_z+age_z+female_z+time_z", df_bprs, groups=df_bprs["code"], re_formula='1+clus_c_z', missing='drop') 
mdf4 = md4.fit(reml = True)
print(mdf4.summary())
cov = mdf4.cov_re

r = cov.iloc[0, 1] / (
    cov.iloc[0, 0] * cov.iloc[1, 1]
) ** 0.5

print(f"Random intercept–slope correlation = {r:.3f}")
slope_clus = []
slope_clus_re = []
slope_dur = []
slope_dur_re = []
keys = []
for record_id in np.unique(df_bprs.code):
    # try:
    dat_x = df_bprs.loc[df_bprs.code==record_id].copy()
    try:
        mod = smf.ols(formula='bprs ~ clus_z+ids_pres_z+time_z', data=dat_x)
        res_x = mod.fit()
        slope_clus.append(res_x.params['clus_z'])
        print(record_id)
        print(dat_x[['code','clus_z','bprs']])
        print(res_x.params['clus_z'])
    except:
        slope_clus.append(np.nan)
        print('insufficient data for '+record_id)
    try:
        slope_clus_re.append(mdf4.random_effects[record_id]['clus_c_z'])
    except:
        slope_clus_re.append(np.nan)
    # try:
    #     mod = smf.ols(formula='bprs ~ cent', data=dat_x)
    #     res_x = mod.fit()
    #     slope_dur.append(res_x.params['cent'])
    # except:
    #     slope_dur.append(np.nan)
    # try:
    #     slope_dur_re.append(mdf1.random_effects[record_id]['cent_z'])
    # except:
    #     slope_dur_re.append(np.nan)
    keys.append(record_id)
    # print(res_x.summary())
    # except:
    #     # slope.append(np.nan)
    #     slope_re.append(np.nan)
slope_dict = pd.DataFrame({'code':keys,
                           # 'slope_cent_pat_ols':slope_cent,
                           # 'slope_cent_pat_re':slope_cent_re,
                           'slope_clus_ols':slope_clus,
                           'slope_clus_re':slope_clus_re}) 
df_slope = pd.merge(df_bprs, slope_dict,on='code', how='outer')


df_slope = df_slope.groupby('code').agg(
    slope_clus_ols = ('slope_clus_ols','last'),
    slope_clus_re = ('slope_clus_re','last'),
    study_start = ('study_start','last'),
    study_end = ('study_end','last'),
    date_mri = ('date_mri','last'),
    whitecat = ('whitecat','last'),
    # dur = ('dur','mean'),
    # cent = ('cent','mean'),
    # dyad_ratio = ('dyad_ratio','mean'),
    # dyadic_staff_dur = ('dyadic_staff_dur','mean'),
    # dyadic_pat_dur = ('dyadic_pat_dur','mean'),
    # conc = ('conc','mean'),
    bprs = ('bprs','mean'),
    # rich_club = ('rich_club','mean'),
    # comm_ratio = ('comm_ratio','mean'),
    # staff_ratio = ('staff_ratio','mean'),
    # clus = ('clus','mean'),
    # ids_pres = ('ids_pres','mean'),
    # rel_int = ('rel_int','mean'),
    # rssi = ('rssi_mean_weighted_mean','mean)
    age = ('age','last'),
    female = ('female','last'),
    # iei = ('iei','mean')
    ).reset_index()
# df_slope['date'] = df_slope['date_mri']
df_slope = pd.merge(df_network, df_slope,on=['code'], how='outer')

# Example network columns
network_cols = [
    "duration_minutes_sum", 
    "duration_minutes_sum_pat", 
    "clustering-coefficient",
    "eigenvector-centrality",
    "dyadic_interaction_staff_median_minutes",
    "dyadic_interaction_patient_median_minutes",
    "ids_present",
    # "study_start",
    # "study_end",
    "dyad_ratio",
    "staff_ratio",
    "ids_present_pat_x",
    "ids_pres_staff",
    "duration_minutes_relative",
    "entropy",
    "intercontact_interval_minutes_median",
    "rssi_comm_in",
    "rssi_comm_out",
    "rssi_ratio",
    "duration_minutes_median",
    "rssi_mean_weighted",
    "rssi_mean_weighted_pat",
    "partner-concentration",
    "partner-concentration_pat",
    "burstiness",
    "weighted-jaccard-prev-day",
    "community_ratio",
    "community_ratio_pat",
]

# Make sure dates are datetime
df_slope["date_mri"] = pd.to_datetime(df_slope["date_mri"]).dt.date
df_network["date"] = pd.to_datetime(df_network["date"]).dt.date

# -----------------------------
# 1. Network values exactly at MRI date
# -----------------------------
# -------------------------
# Base table: one row per participant
# -------------------------
df_final = (
    df_slope[["code", "date_mri",'age','female','bprs','slope_clus_ols','slope_clus_re','study_start_x','study_end_x','whitecat']]
    .drop_duplicates("code")
)
df_final['study_start'] = df_final['study_start_x']
df_final['study_end'] = df_final['study_end_x']
df_final=df_final.drop(columns=['study_start_x','study_end_x'])
# -------------------------
# Network values on MRI date
# -------------------------
mri_values = (
    df_network[["code", "date"] + network_cols]
    .merge(
        df_final,
        left_on=["code", "date"],
        right_on=["code", "date_mri"],
        how="inner",
    )
    .drop(columns="date")
    .rename(columns={c: f"{c}_mri" for c in network_cols})
)

df_final = df_final.merge(mri_values, on=["code", "date_mri"], how="left")

# -------------------------
# Mean across whole study
# -------------------------
mean_all = (
    df_network
    .groupby("code")[network_cols]
    .mean()
    .rename(columns=lambda c: f"{c}_mean_all")
    .reset_index()
)

df_final = df_final.merge(mean_all, on="code", how="left")

# -------------------------
# Mean ±3 days around MRI
# -------------------------
tmp = df_network.merge(df_final[["code", "date_mri"]], on="code")

week = tmp[
    (tmp["date"] >= tmp["date_mri"] - pd.Timedelta(days=3)) &
    (tmp["date"] <= tmp["date_mri"] + pd.Timedelta(days=3))
]

mean_week = (
    week
    .groupby("code")[network_cols]
    .mean()
    .rename(columns=lambda c: f"{c}_mean_week")
    .reset_index()
)

df_final = df_final.merge(mean_week, on="code", how="left")
# mod = smf.ols(formula='slope_clus_re ~ slope_clus_ols', data=df_slope)
# res_y = mod.fit()
# print(res_y.summary())
df_final = df_final.dropna(subset=["date_mri"])
df_final['study_start'] = df_final['study_start_x']
df_final['study_end'] = df_final['study_end_x']
df_final['female'] = df_final['female_x']
df_final['whitecat'] = df_final['whitecat_x']
df_final['age'] = df_final['age_x']
df_final['bprs'] = df_final['bprs_x']
df_final['slope_clus_ols'] = df_final['slope_clus_ols_x']
df_final['slope_clus_re'] = df_final['slope_clus_re_x']
df_final['clus'] = df_final['clustering-coefficient_mean_all']
df_final = df_final.drop(columns=['slope_clus_ols_x', 'slope_clus_re_x','study_start_x','bprs_x','bprs_y','study_end_x','study_start_y','study_end_y','age_x','age_y','female_x','female_y','whitecat_x','whitecat_y'])
df_final = df_final.set_index("code")

# df_final.to_pickle('../data/exchange/df_slope.pkl')
# df_final.to_csv('../out/df_slope.csv')

df_slope.to_csv('../data/exchange/df_slope.csv')

###########################################################################################
# Tables
###########################################################################################
# Tb. 1: sample characteristic table

df_dx, summary, dx_summary = get_meta_summary(df_meta)
table_summary = pd.concat([summary, dx_summary], ignore_index=True)
# Mean BPRS per participant
bprs_subject = (
    df_bprs
    .groupby("code", as_index=False)["bprs"]
    .mean()
)
# Overall summary statistics
df_network_incl_staff["is_ther"] = df_network_incl_staff["code"].str.startswith("ther", na=False)
df_network_staff = df_network_incl_staff.loc[df_network_incl_staff["is_ther"]]

bprs_mean = bprs_subject["bprs"].mean()
bprs_sd = bprs_subject["bprs"].std()

median_dur_pat = np.nanmedian(df_dyads.loc[((df_dyads['id_src_ispat'])|(df_dyads['id_tgt_ispat'])),'duration_minutes'])
iqr_dur_pat = iqr(df_dyads.loc[((df_dyads['id_src_ispat'])|(df_dyads['id_tgt_ispat'])),'duration_minutes'])
median_dur_staff = np.nanmedian(df_dyads.loc[(~df_dyads['id_src_ispat'])|(~df_dyads['id_tgt_ispat']),'duration_minutes'])
iqr_dur_staff = iqr(df_dyads.loc[(~df_dyads['id_src_ispat'])|(~df_dyads['id_tgt_ispat']),'duration_minutes'])
median_iei_pat = np.nanmedian(df_dyads.loc[(df_dyads['id_src_ispat'])|(df_dyads['id_tgt_ispat'])]
                              .groupby(['id_pair','date'])['time_of_day'].diff()*60)
iqr_iei_pat = iqr(df_dyads.loc[(df_dyads['id_src_ispat'])|(df_dyads['id_tgt_ispat'])]
                  .groupby(['id_pair','date'])['time_of_day'].diff()*60, nan_policy="omit")
median_iei_staff = np.nanmedian(df_dyads.loc[(~df_dyads['id_src_ispat'])|(~df_dyads['id_tgt_ispat'])]
                                .groupby(['id_pair','date'])['time_of_day'].diff()*60)
iqr_iei_staff = iqr(df_dyads.loc[(~df_dyads['id_src_ispat'])|(~df_dyads['id_tgt_ispat'])]
                    .groupby(['id_pair','date'])['time_of_day'].diff()*60, nan_policy="omit")

def mean_from_df_network(df, var):   
    df_var = (
        df
        .groupby("code", as_index=False)[var]
        .mean()
    )
    mean_var = np.nanmean(df_var[var])
    sd_var = np.nanstd(df_var[var])
    return {'mean_'+str(var):mean_var,'sd_'+str(var):sd_var}
rel_int_pat = mean_from_df_network(df_network, 'duration_minutes_relative')
rel_int_staff = mean_from_df_network(df_network_staff, 'duration_minutes_relative')
dyad_index_pat = mean_from_df_network(df_network, 'dyad_index')
dyad_index_staff = mean_from_df_network(df_network_staff, 'dyad_index')
staff_index_pat = mean_from_df_network(df_network, 'staff_index')
staff_index_staff = mean_from_df_network(df_network_staff, 'staff_index')
clus_pat = mean_from_df_network(df_network, 'clustering-coefficient')
clus_staff = mean_from_df_network(df_network_staff, 'clustering-coefficient')
cent_pat = mean_from_df_network(df_network, 'eigenvector-centrality')
cent_staff = mean_from_df_network(df_network_staff, 'eigenvector-centrality')
community_ratio_pat = mean_from_df_network(df_network, 'community_ratio')
community_ratio_staff = mean_from_df_network(df_network_staff, 'community_ratio')

total_dyads = len(df_dyads)
total_dyads_pat= len(df_dyads.loc[(df_dyads['id_src_ispat'])|(df_dyads['id_tgt_ispat'])])
total_dyads_staff=len(df_dyads.loc[(~df_dyads['id_src_ispat'])|(~df_dyads['id_tgt_ispat'])])

# Add to summary table
rows = [
    ["Dyads, total", total_dyads, total_dyads_pat, total_dyads_staff],
    ["Duration, median (IQR)",
     "",
     f"{median_dur_pat:.1f} ({iqr_dur_pat:.1f})",
     f"{median_dur_staff:.1f} ({iqr_dur_staff:.1f})"],
    ["IEI, median (IQR)",
     "",
     f"{median_iei_pat:.1f} ({iqr_iei_pat:.1f})",
     f"{median_iei_staff:.1f} ({iqr_iei_staff:.1f})"],
]

for label, pat, staff in [
    ("Relative interaction, mean (SD)", rel_int_pat, rel_int_staff),
    ("Dyad index, mean (SD)", dyad_index_pat, dyad_index_staff),
    ("Staff index, mean (SD)", staff_index_pat, staff_index_staff),
    ("Clustering coefficient, mean (SD)", clus_pat, clus_staff),
    ("Eigenvector centrality, mean (SD)", cent_pat, cent_staff),
    ("Community ratio, mean (SD)", community_ratio_pat, community_ratio_staff),
]:
    var_pat = next(k.replace("mean_", "") for k in pat if k.startswith("mean_"))
    var_staff = next(k.replace("mean_", "") for k in staff if k.startswith("mean_"))

    rows.append([
        label,
        "",
        f"{pat['mean_'+var_pat]:.2f} ({pat['sd_'+var_pat]:.2f})",
        f"{staff['mean_'+var_staff]:.2f} ({staff['sd_'+var_staff]:.2f})",
    ])

network_summary = pd.DataFrame(
    rows,
    columns=["Characteristic", "Total", "Patient", "Staff"]
)

table_summary = pd.concat([table_summary, network_summary], ignore_index=True)
dx_mask = table_summary["Characteristic"].str.startswith("Diagnosis", na=False)

# Move counts from Total -> Patient
table_summary.loc[dx_mask, "Patient"] = table_summary.loc[dx_mask, "Total"]
table_summary.loc[dx_mask, "Total"] = ""
# Append to existing table
print(table_summary)

# Optional export
table_summary.to_csv("../data/exchange/clinical_summary_table.csv", index=False)
# table_summary.to_excel("clinical_summary_table.xlsx", index=False)





#  network
md6= smf.mixedlm("clus ~ time_z+ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf6 = md6.fit(reml = True)
print(mdf6.summary())
sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(10, 5))


sns.lineplot(data = df_bprs, x = 'time',y = 'clus', hue = 'code', ci = None, legend = False, alpha = 0.5)

plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.grid(axis="x", visible=False)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

# 
md1 = smf.mixedlm("bprs ~ clus_mean_z + clus_c_z +ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
# Get summary table
# table1 = mdf1.summary().tables[1].reset_index()
md1 = smf.mixedlm("bprs ~ dyad_ratio_mean_z + dyad_ratio_c_z +ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
md1 = smf.mixedlm("bprs ~ rssi_ratio_mean_z+rssi_ratio_c_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())

md1 = smf.mixedlm("bprs ~ comm_ratio_mean_z + comm_ratio_c_z +ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
md1 = smf.mixedlm("clus ~ entropy_c_z + ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
md1 = smf.mixedlm("comm_ratio ~ time_z+ids_pres_z", df_bprs, groups=df_bprs["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
# Save
# table.to_csv("../data/exchange/indiv_summary.csv", index=False)
r2 = mixedlm_r2(mdf1, df_bprs)

print(f"Marginal R²: {r2['marginal_r2']:.3f}")
print(f"Conditional R²: {r2['conditional_r2']:.3f}")   
# % avg graph
# adj_dict = pd.read_pickle('../data/exchange/adj2026-03-16.pkl')
# patient_ids = {12,18,2,19,34,37,13,7,51,53,56,50,52,57,43,44,3,8,99,85,
#                 9,83,61,62,71,73,74,79,80,81,84,91,92,94,95,96,97,98}  

# G_avg, A_avg = build_average_graph(
#     adj_dict,
#     patient_ids,
#     threshold_staff=0.7,
#     threshold_mixed=0.7,
#     threshold_patient=0.7
# )

# plot_graph(G_avg)

# # % dur & Time 
# df = pd.read_pickle(('../data/exchange/df_contacts.pkl'))
# df = df.loc[df['id_tgt']!=0]
# sns.histplot(data = df, x  = 'time_of_day')
# plt.plot()
# sns.histplot(data = df, x  = 'duration_minutes', log_scale=True)


# 
df_daily_sums = df_dyads.groupby(['id_pair','date']).agg(duration_minutes=('duration_minutes','sum'), rssi=('rssi_mean','mean')).reset_index()
df_daily_sums["duration_minutes_prev"] = df_daily_sums.groupby("id_pair")["duration_minutes"].shift(1)
df_daily_sums["rssi_prev"] = df_daily_sums.groupby("id_pair")["rssi"].shift(1)

df_daily_sums['first_date'] =  df_daily_sums.groupby('id_pair')['date'].transform('min')
df_daily_sums['time'] = pd.to_timedelta(df_daily_sums['date']-df_daily_sums['first_date']).dt.days
# means
num_cols = df_daily_sums.select_dtypes(include='number').columns
df_daily_sums[[f"{col}_mean" for col in num_cols]] = (
    df_daily_sums.groupby('id_pair')[num_cols].transform('mean')
)
# z-scoring and centering
num_cols = df_daily_sums.select_dtypes(include='number').columns
df_daily_sums[[f"{col}_c" for col in num_cols]] = (
    df_daily_sums[num_cols] - df_daily_sums.groupby('id_pair')[num_cols].transform('mean')
)

num_cols = df_daily_sums.select_dtypes(include='number').columns
df_daily_sums[[f"{col}_z" for col in num_cols]] = (
    (df_daily_sums[num_cols] - df_daily_sums[num_cols].mean()) / df_daily_sums[num_cols].std()
)

# 
md2 = smf.mixedlm("duration_minutes~duration_minutes_prev_c_z + duration_minutes_mean_z+time_z", df_daily_sums, groups=df_daily_sums["id_pair"], re_formula='1', missing='drop') 
mdf2 = md2.fit(reml = True)
print(mdf2.summary())
md6 = smf.mixedlm("duration_minutes~rssi_c_z + rssi_mean_z+time_z", df_daily_sums, groups=df_daily_sums["id_pair"], re_formula='1', missing='drop') 
mdf6 = md6.fit(reml = True)
print(mdf6.summary())
# Get summary table
# table2 = mdf2.summary().tables[1].reset_index()
r2 = mixedlm_r2(mdf2, df_daily_sums)
example_pair = df_daily_sums.loc[df_daily_sums['id_pair']== '2_6']
sns.lineplot(example_pair,x='date', y='rssi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
sns.scatterplot(data = df_daily_sums, x = 'rssi', y= 'duration_minutes')