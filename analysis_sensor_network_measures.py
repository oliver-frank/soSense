import os
import sys
import pandas as pd
from preprocessing_sensors import df_cleaned as df
from utils.settings import data_dir
from utils.sensors_network_measures import get_local_measures_df
from utils.sensors_network_measures import keep_clean_patient_patient
from utils.sensors_network_measures import get_matrix_dict
from utils.sensors_network_measures import compute_daily_interaction_stats
from utils.sensors_network_measures import plot_contacts_per_day_and_week
from utils.sensors_network_measures import get_active_ids_by_time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# set output_dir
if not os.path.isdir(f"../output/figures/{script_name}"):
    os.makedirs(f"../output/figures/{script_name}")
    
###############################################################################
## General settings
###############################################################################

# define a threshold (every value lower than this will be set to 0)
matrix_threshold = 0

# include all patient-patient-interactions, or only when there is no staff involved?
strict_patients_only = False

# should the matrices be normalized?
normalization = False # can be False or min-max

# define badge outcome
outcome = 'duration_minutes'  # duration_minutes vs. rssi_mean
# outcome = 'iei'  # duration_minutes vs. rssi_mean

# set unit of time that defines time window to aggregate contact information between dyads
unit_time = 'date' # date vs. year_week
aggregation_minutes = None

# define a list of measures that you want to compute
measures = ['strengths',
            'degree',
            'weighted-jaccard-prev-day',
            'clustering-coefficient',
            'partner-concentration',
            'eigenvector-centrality',
            'entropy',
            'k_core',
            'rich_club',
            'n_comms',
            'modularity',
            'community_ratio',
            'betweenness_centrality',
            'density',
            'network_size',
            'rssi_comm_in',
            'rssi_comm_out',
            'rssi_comm_ratio'
            ]

###############################################################################
## Compute subject-focused network stats
###############################################################################

df_copy = df.copy()
if strict_patients_only:
    df_pats = keep_clean_patient_patient(df)
else:
    df_pats = df.loc[((df['id_src_ispat'] == True) & (df['id_tgt_ispat'] == True))]
df_daily_stats_pat = compute_daily_interaction_stats(df_pats)
df_daily_stats = compute_daily_interaction_stats(df_copy)
cols_to_keep = ['id', 'date']
# I need different column names for '_pat' for the statistics later (PG):
df_daily_stats_pat = df_daily_stats_pat.rename(
    columns={col: f"{col}_pat" for col in df_daily_stats_pat.columns if col not in cols_to_keep}
)
df_daily_stats = pd.merge(df_daily_stats, df_daily_stats_pat, on= ['id','date'],how='outer')

###############################################################################
## Get adjacency matrices for each unit of time and compute local topological
## measures
###############################################################################

# drop battery contacts
df = df.loc[df['id_tgt'] != 0]
# TODO: threshold? (PG: for now, no explicit threshold)
# for matrix_threshold in[0,2.5,5,7.5,10,15,20]:
df_copy = df.copy()
results = []
# get matrix dictionary using different subsets of badges
# I added a long format dataframe
for group_context in ['all','patients_only']: #,    
    if group_context == 'patients_only':
        if strict_patients_only:
            df_copy = keep_clean_patient_patient(df)
        else:
            df_copy = df.loc[((df['id_src_ispat'] == True) & (df['id_tgt_ispat'] == True))]
        df_copy['duration_seconds'] = (df_copy['contact_end']-df_copy['contact_start']).dt.total_seconds()
        df_copy['duration_minutes'] = (df_copy['contact_end']-df_copy['contact_start']).dt.total_seconds()/60
    matrix_dict = get_matrix_dict(df=df_copy,unit_time=unit_time,outcome=outcome,aggregation_minutes=aggregation_minutes)
    with open('../data/exchange/matrix_dict_'+group_context+'.pickle', 'wb') as handle:
        pickle.dump(matrix_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # compute all measures, store both in wide and long format
    df_measures = []
    df_measures_long = []

    for measure in measures:   
        
        active_ids_5min = get_active_ids_by_time(df, aggregation_minutes=aggregation_minutes)
        
        df_measure = get_local_measures_df(
            df,
            matrix_dict,
            measure_name=measure,
            aggregation_minutes=aggregation_minutes,
            active_ids_by_time=active_ids_5min,
            n_jobs=-1,
        )
        # df_measure = get_local_measures_df(matrix_dict,measure,matrix_threshold=matrix_threshold,)
        df_measure_long = df_measure.reset_index().melt(
            id_vars='id',
            var_name='date',
            value_name=measure
        ).dropna(subset=measure)
        df_measures.append(df_measure)
        if len(df_measures_long)>0:
            df_measures_long = df_measures_long.merge(df_measure_long,on = ['id','date'], how='outer')
        else:
            df_measures_long = df_measure_long
            
        # just plot as matrix
        plt.figure(figsize=(15,15))
        ax = sns.heatmap(df_measure,cmap="viridis")
        ax.set(xlabel=unit_time,ylabel="Badge",title=f"Group Context: {group_context}, Measure: {measure}")
        plt.tight_layout()
        plt.show()

    df_network = pd.concat(df_measures,keys=measures,names=['topological_measure',unit_time],axis='columns')
    df_network = df_network.melt(ignore_index=False,value_name='topological_value').reset_index()
    df_network['group_context'] = group_context
    if group_context == 'all':
        df_network_long = df_measures_long
    else:
        cols_to_keep = ['id', 'date']
        df_measures_long = df_measures_long.rename(
            columns={col: f"{col}_pat" for col in df_measures_long.columns if col not in cols_to_keep}
        )
        df_network_long =  df_network_long.merge(df_measures_long,on = ['id','date'], how='outer')
        

    results.append(df_network)

df_network = pd.concat(results)

###############################################################################
# Merge daily stats and network measures and replace sensor ids 
# with subject ids
###############################################################################

df_network["time_window_start"] = df_network['date']
df_network["date"] = pd.to_datetime(df_network["date"]).dt.date
df_daily_stats["date"] = pd.to_datetime(df_daily_stats["date"]).dt.date
df_network_long["time_window_start"] = df_network_long['date']
df_network_long["date"] = pd.to_datetime(df_network_long["date"]).dt.date
# df_network = df_network.merge(df_daily_stats,on=['id','date','group_context'],how='outer')
df_network_long = df_network_long.merge(df_daily_stats,on=['id','date'],how='outer')

path = os.path.join(data_dir,'sensors','rec2id.csv')
df_id_mapper = pd.read_csv(path)
# df_network = df_network.merge(df_id_mapper,left_on='id',right_on='current_id',how='outer')
df_network_long = df_network_long.merge(df_id_mapper,left_on='id',right_on='current_id',how='outer')

# mask_nan_code = df_network["code"].isna()
# df_network.loc[mask_nan_code,"code"] = "therapist_" + df_network.loc[mask_nan_code,"id"].astype(str)
mask_nan_code = df_network_long["code"].isna()
df_network_long.loc[mask_nan_code,"code"] = "therapist_" + df_network_long.loc[mask_nan_code,"id"].astype(str)

###############################################################################
## Plotting
###############################################################################

# Plot outcome measure on y-axis, time of contact start on x-axis day as column, study week as row
if __name__ == "__main__":
    fig = plot_contacts_per_day_and_week(df,outcome)
    fig.write_html(f"../output/figures/{script_name}/{outcome}_per_day_and_week.html",auto_open=True)

###############################################################################
## Save social network df
###############################################################################

df_network_long.to_pickle('../data/exchange/df_network' + '_rssi_75_above_5_min_dur.pkl')