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

"""
Predict survey measures from network measures. For each survey assesssment
we need to find the network measure for the week before.

This script must output df_slope.csv in /data/exchange
"""

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
df_network = pd.read_pickle("../data/exchange/df_network_rssi_75_above_5_min_dur.pkl")
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



# filter by quality criteria
df_network = df_network.loc[df_network['ids_present']>=8]
df_network = df_network.loc[df_network['ids_present_pat_x']>=3]
df_network = df_network.loc[df_network['contact_span_minutes']>60]
# df_network = df_network.loc[df_network['patient_sensor_ratio']>0.3]


df_dyads = df_dyads.loc[df_dyads['id_tgt'] != 0]
# sns.histplot(data = df_network, x = 'patient_sensor_ratio')









df_network_incl_staff = df_network.copy()
df_network = df_network[~df_network["code"].str.startswith("therapist")]
df_panss['pos']= df_panss['panss_p1']+df_panss['panss_p2']+df_panss['panss_p3']+df_panss['panss_p4']+df_panss['panss_p5']+df_panss['panss_p6']+df_panss['panss_p7']
# FIXME: Should I use "outer in the other script?"
df_merged = pd.merge(df_network, df_panss, on = ['code','date'], how='outer')



# %%
##############################
# testing end
##############################
 # for t_win in [-14,-12,-10,-8,-6,-4,-2,1,3,5,7]:
df_merged['new_panss'] = ~np.isnan(df_merged['panss'].shift())
 # df_merged['new_window'] = ~np.isnan(df_merged['panss'].shift(t_win))
 # df_merged["window_id"] = df_merged.groupby(["code"])['new_window'].cumsum()

df_merged["panss_id"] = df_merged.groupby(["code"])['new_panss'].cumsum()
 # FIXME: make sure that no values of more than 7 days before are included
 # df_last7 = df_merged.groupby(['code','panss_id']).tail(14)
df_merged['latest'] = df_merged.sort_values('date').groupby(['code','panss_id'])['date'].transform('last')
df_merged['panss'] = df_merged.sort_values('date').groupby(['code','panss_id'])['panss'].transform('last')
 # df_merged['panss'] = df_merged.sort_values('date').groupby(['code','window_id'])['panss'].transform('last')
df_merged['pos'] = df_merged.sort_values('date').groupby(['code','panss_id'])['pos'].transform('last')

 # compute differenceNeoFFI30
df_merged['diff_to_latest'] =pd.to_timedelta(df_merged['latest']  - df_merged['date']).dt.days
df_merged = df_merged.loc[df_merged['diff_to_latest'] <8]
df_merged = df_merged.loc[df_merged['diff_to_latest'] >= 1]
df_merged['n_instances'] = df_merged.sort_values('date').groupby(['code','panss_id'])['date'].transform('count')
 # df_merged = df_merged.loc[df_merged['n_instances'] > 1]

 # aggregate over the 7 day window  'calendar_year','calendar_week', 'panss_id'
 #     
df_panss = df_merged.groupby(['code','panss_id'], as_index=False).agg(
         panss = ('panss','last'),
         pos = ('pos','last'),

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
         # weekday = ("weekday_panss",'last'),
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
# df_panss['weekday'] = df_panss['weekday'].astype('category')

df_panss = df_panss.sort_values(["code","date"])
exclude = ["id", "code", "date"]
num_cols = df_panss.select_dtypes(include='number').columns

df_panss[[f"{c}_prev" for c in num_cols]] = (
    df_panss.groupby("code")[num_cols].shift(1)
)
df_panss[[f"{c}_next" for c in num_cols]] = (
    df_panss.groupby("code")[num_cols].shift(-1)
)
df_panss[[f"{c}_next_next" for c in num_cols]] = (
    df_panss.groupby("code")[num_cols].shift(-2)
)
df_panss[[f"{c}_dx" for c in num_cols]] = (
    df_panss[num_cols] - df_panss.groupby('code')[num_cols].shift(1)
)

df_panss = pd.merge(df_panss, df_meta, on='code', how= 'left')
df_panss['study_start'] =pd.to_datetime(df_panss['start_study']).dt.date
df_panss['date_mri']=pd.to_datetime(df_panss['date_mri']).dt.date
df_panss['age'] = (pd.to_datetime(df_panss['study_start']) -pd.to_datetime(df_panss['birth_date'])).dt.days/365
df_panss = df_panss.sort_values(["code", "date"])


# previous date per participant
df_panss["prev_date"] = df_panss.groupby("code")["date"].shift(1)
# means
num_cols = df_panss.select_dtypes(include='number').columns
df_panss[[f"{col}_mean" for col in num_cols]] = (
    df_panss.groupby('code')[num_cols].transform('mean')
)
# z-scoring and centering
num_cols = df_panss.select_dtypes(include='number').columns
df_panss[[f"{col}_c" for col in num_cols]] = (
    df_panss[num_cols] - df_panss.groupby('code')[num_cols].transform('mean')
)


num_cols = df_panss.select_dtypes(include='number').columns
df_panss[[f"{col}_z" for col in num_cols]] = (
    (df_panss[num_cols] - df_panss[num_cols].mean()) / df_panss[num_cols].std()
)


df_ssd = df_panss.loc[df_panss.idx.isin(ssd)]
df_psy = df_panss.loc[~df_panss.idx.isin(no_psy)]
df_panss.to_csv('~/Desktop/df.csv')
# df_panss['weekday']=df_panss['weekday'].astype('category')
df_panss = df_panss.drop_duplicates(subset=["code", "date"])
## this for later implementing a histogram clus_effect_size~time_window
# md2 = smf.mixedlm("panss~ +clus_c_z+ids_pres_z", df_panss, groups=df_panss["code"], re_formula='1', missing='drop') 
# mdf2 = md2.fit(reml = True)
# p_vals.append({
#     'predictor': 'clus_c_z',
#     'win': t_win,
#     'p': mdf2.pvalues['clus_c_z'],
#     'coef': mdf2.params['clus_c_z'],
#     'n': mdf2.model.n_groups,
#     'outcome': 'panss'
# })            

# p_vals = pd.DataFrame(p_vals)
###############################################################################
## Stats
###############################################################################
# %%
md1 = smf.mixedlm("panss ~ rssi_c_z+rssi_mean_z", df_psy, groups=df_psy["code"], re_formula='1', missing='drop') 
mdf1 = md1.fit(reml = True)
print(mdf1.summary())
# %% ttest
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Compute participant means
df_mean = (
    df_network
    .groupby("code")[["rssi_comm_in", "rssi_comm_out"]]
    .mean()
    .reset_index()
)

# 2. Paired t-test
t_stat, p_value = ttest_rel(
    df_mean["rssi_comm_in"],
    df_mean["rssi_comm_out"],
    nan_policy="omit"
)

print(f"t = {t_stat:.3f}")
print(f"p = {p_value:.4f}")
# Convert to long format for plotting
df_long = df_mean.melt(
    id_vars="code",
    value_vars=["rssi_comm_in", "rssi_comm_out"],
    var_name="Condition",
    value_name="RSSI"
)

plt.figure(figsize=(6, 5))

# Connect each participant's values
sns.lineplot(
    data=df_long,
    x="Condition",
    y="RSSI",
    hue="code",
    estimator=None,
    alpha=0.4,
    legend=False
)

# Overlay participant means
sns.pointplot(
    data=df_long,
    x="Condition",
    y="RSSI",
    errorbar=("ci", 95),
    color="black"
)

plt.title(f"Paired comparison\nPaired t-test: p = {p_value:.3f}")
plt.tight_layout()
plt.show()
# TODO: ANOVA w/ SSD/Psy/no-psy/therapeuts

df_mean_staff = (
    df_network_incl_staff
    .groupby("code")[["rssi_mean_weighted"]]
    .mean()
    .reset_index()
)
df_mean_staff["staff"] = df_mean_staff["code"].str.startswith("therapist").astype('category')
df_mean_pat = df_mean_staff.loc[~df_mean_staff["code"].str.startswith("therapist")]

df_mean_ther =  df_mean_staff.loc[df_mean_staff["code"].str.startswith("therapist")]
t_stat, p_value = ttest_ind(
    df_mean_pat["rssi_mean_weighted"],
    df_mean_ther["rssi_mean_weighted"],
    nan_policy="omit"
)
sns.boxplot(data = df_mean_staff,
            x = "rssi_mean_weighted",
            y = "staff")