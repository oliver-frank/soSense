import os
import glob
from utils.settings import n_jobs
from utils.settings import data_dir
from utils.sensors_preprocessing import get_logfile_df
from utils.sensors_preprocessing import get_contact_df
from utils.sensors_preprocessing import remove_faulty_interactions
from utils.sensors_preprocessing import plot_contacts_3d
from utils.sensors_preprocessing import set_patient_ids

###############################################################################
## Get inputs
###############################################################################

# get directory to logfiles and get all .json files as list
dir_sensor_logs = os.path.join(data_dir,'sensors','sensor_logs/')
logfiles = glob.glob(dir_sensor_logs + '*.json')
path_to_patient_id = os.path.join(data_dir,'sensors','patient_ids.csv')

# TODO: delete this faulty recording
faulty_recording = 'Conlog_ID2_2023-07-06T05-12-39.995Z.json'
logfiles = [logfile for logfile in logfiles if faulty_recording not in logfile]

# TODO: I changed the file in my folder, has to be changed in other folders as well - PG
# path to .tsv file that informs about id switches
# TODO: We should eventually track this via git (JW)
path_id_switches = os.path.join(data_dir,'sensors','id_switches.tsv')

###############################################################################
## Settings for preprocessing
###############################################################################

config_contacts = dict(
    must_be_after = "9:00", 
    must_be_before = "20:00",
    exclude_days = ['Saturday','Sunday'],
    gap_person = 75,
    gap_batt = 400,
    min_detections = 2,
    min_duration_seconds = 4,
    # -75 for networks, -85 for RSSI 
    threshold_lower = -75,
    threshold_upper = -40, 
    threshold_hq_lower = -75,
    threshold_hq_upper = -50,
    lag_autocorr = 1,
    path_id_switches=path_id_switches,
    )

# parameters that define idle/faulty contacts. If longer than minutes_thr + 
# any of the other measures is true, a contact is defined as faulty
config_faulty_contacts = dict(
    rssi_mean_thr = -60,
    rssi_autocorr_thr = 0.9,
    faulty_min_minutes=10,
    unlikely_thr = 0.8
    )

###############################################################################
## Sensor preprocessing
###############################################################################

# get all logfiles as one large data frame
df = get_logfile_df(logfiles,n_jobs)

# convert to contacts
df = get_contact_df(df,config_contacts)

###############################################################################
## Remove faulty interactions
###############################################################################

df_cleaned = remove_faulty_interactions(df,config_faulty_contacts)

# just for rssi analysis
df_cleaned = df_cleaned.loc[df_cleaned['duration_minutes']>5]

###############################################################################
## Add helper columns for upcoming analyses
###############################################################################

# add two columns that show if either source or target id is a patient
df_cleaned = set_patient_ids(df_cleaned, path_to_patient_id)

################################################################################
# Plotting
###############################################################################

if __name__ == "__main__":
    
    fig = plot_contacts_3d(df.loc[(df['id_tgt']!=0)&(df['duration_minutes']>1)],title='Contacts before removing faulty contacts')
    fig.show()
    
    fig = plot_contacts_3d(df_cleaned.loc[(df_cleaned['id_tgt']!=0)&(df_cleaned['duration_minutes']>1)],title='Contacts after removing faulty contacts')
    fig.show()
    
    # save output as pickle file
    df_cleaned.to_pickle('../data/exchange/df_contacts.pkl')