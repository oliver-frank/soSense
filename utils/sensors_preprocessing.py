from .settings import memory
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

def get_df_from_logfile(path_logfile,timezone='Europe/Berlin'):
    '''Reads in a single sensor logfile as dataframe.
    1. Removes duplicate rows.
    2. Ensures appropriate datatypes.
    3. Converts timestamps to datetimes in respective timezone
    4. Drops unneeded columns & renames certain columns.
    '''

    # read in logfile as dataframe
    df = pd.read_json(path_logfile)

    # drop duplicate rows
    df = df.drop_duplicates()

    # dtypes for non-datetime columns
    df = df.astype({"id":"Int64","idname":"string","rssi": "Int64","tx": "Int64","myID": "Int64"})

    # parse the timestamp column and convert to datetime in defined timezone
    df["datetime"] = pd.to_datetime(df["timestamp"],errors="coerce",utc=True).dt.tz_convert(timezone)

    # delete unnecessary columns
    df = df.drop(columns=["idname","tx","timestamp"])

    # rename columns
    df = df.rename(columns={"myID":"id_src","id":"id_tgt"})

    return df

@memory.cache
def get_logfile_df(paths_logfiles,n_jobs):
    '''Reads in all logfiles as dataframes and concatenates them to one dataframe.
    Having a separate function for this is useful for developing (if you just
    want to test out for a small number of logfiles)'''

    # first read in all logfiles
    with joblib.Parallel(n_jobs=n_jobs,verbose=10) as parallel:
        dfs = parallel(joblib.delayed(get_df_from_logfile)(filepath) for filepath in paths_logfiles)

    # concatenate them
    df = pd.concat(dfs)

    return df

def switch_ids(df,path_id_switches):
    """During the study, sometimes a new sensor badge id had to be assigned
    when a badge was lost or broken. Therefore, replace sensor ids according
    to predefined switch intervals.

    For each switch rule (id_old → id_new), the replacement is applied to both
    `id_src` and `id_tgt` if the row's datetime satisfies:

    start_new_id <= datetime <= end_new_id
    """

    # read in the dataframe from file
    df_id_switches = pd.read_csv(path_id_switches,sep="\t",parse_dates=["start_new_id","end_new_id"])
    df_id_switches["start_new_id"] = df_id_switches["start_new_id"].dt.tz_localize("Europe/Berlin")
    df_id_switches["end_new_id"]   = df_id_switches["end_new_id"].dt.tz_localize("Europe/Berlin")

    for id_old, id_new, start, end in df_id_switches.itertuples(index=False):

        time_mask = (df["datetime"] >= start) & (df["datetime"] < end)
        df.loc[time_mask & (df["id_src"] == id_old), "id_src"] = id_new
        df.loc[time_mask & (df["id_tgt"] == id_old), "id_tgt"] = id_new

    return df

def exclude_faulty_days(df):
    """
    Remove all rows on days where a sensor had a self-contact (id_src == id_tgt).

    A "day" is defined as df["datetime"] floored to midnight in the same timezone.
    If a sensor has any self-contact on a given day, then all rows on that day
    involving that sensor (as id_src or id_tgt) are removed.
    """

    df = df.copy()
    df["contact_day"] = df["datetime"].dt.floor("D")

    self_contact_mask = df["id_src"].eq(df["id_tgt"])
    faulty_days = df.loc[self_contact_mask,["id_src","contact_day"]].rename(columns={"id_src":"id"})

    # gives you a multiindex with (id,contact_day) pairs for faulty days
    faulty_index = pd.MultiIndex.from_frame(faulty_days.drop_duplicates())

    src_index = pd.MultiIndex.from_frame(pd.DataFrame({"id": df["id_src"], "contact_day": df["contact_day"]}))
    tgt_index = pd.MultiIndex.from_frame(pd.DataFrame({"id": df["id_tgt"], "contact_day": df["contact_day"]}))

    drop_src = src_index.isin(faulty_index)
    drop_tgt = tgt_index.isin(faulty_index)

    return df.loc[~(drop_src | drop_tgt)].drop(columns=["contact_day"]).reset_index(drop=True)

def autocorr_lag(series_rssi,lag):
    '''Helper function to compute the auto-correlation for a rssi time series'''

    values = series_rssi.dropna().astype(float).to_numpy()

    if values.size <= lag:
        return np.nan

    values_current = values[lag:]
    values_previous = values[:-lag]

    # correlation undefined if either has zero std (constant)
    if np.std(values_current) == 0 or np.std(values_previous) == 0:
        return np.nan

    # result is a symmetric matrix, so we can just take the upper left value
    r = float(np.corrcoef(values_current,values_previous)[0,1])

    return r

@memory.cache
def get_contact_df(df,config):
    """
    Aggregate raw RSSI detections into contact episodes.

    The input `df` contains timestamped detections where a sensor (`id_src`)
    observes another device (`id_tgt`) with a received signal strength (`rssi`)
    at a given `datetime`.

    A contact episode is defined per (id_src,id_tgt) as a sequence of detections
    where consecutive timestamps are no more than `gap` seconds apart
    (for normal contacts). If the time difference between consecutive
    detections exceeds the gap, a new episode starts.

    The device with `id_tgt == 0` represents the battery and is treated as a
    special contact: it is always kept regardless of RSSI thresholding and
    uses a separate gap (`gap_batt`) for episode splitting.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw detections. Must contain at least:
        - 'id_src' (int): identifier of the sensing device
        - 'id_tgt' (int): identifier of the detected device (0 == battery)
        - 'datetime' (datetime-like): time of detection
        - 'rssi' (int/float): RSSI in dBm (less negative == stronger/closer)

    must_be_after: str
        Timestamp of detection must be after this time

    must_be_before: str
        Timestamp of detection must be before this time

    exclude_days: list of str
        Timestamps on days that should be ignored

    gap_person : int
        Maximum allowed time gap (seconds) between consecutive detections
        to belong to the same contact episode for non-battery contacts
        (id != 0).

    gap_batt : int
        Maximum allowed time gap (seconds) between consecutive detections
        to belong to the same episode for battery detections (id == 0).

    min_detections: int
        Minimum number of a sequence of consecutive detections to be considered a contact

    min_duration_seconds: int
        Minimum timespan for a sequence of consecutive detections to be considered a contact.
        If no specific hypthosesis, this is set to 2 times the sample rate, i.e.
        two consecutive detections. Having this argument in addition to min_detections helps
        us to get rid of contact pairs where two badges detected each other at the exact same
        time, but only once. In this case, number of detections for this
        contact pair will be 2 but duration will be 0.

    threshold_lower : int
        Minimum RSSI (dBm) required for a non-battery detection (id != 0)
        to be included when forming episodes. Battery detections (id == 0)
        are included regardless of RSSI.

    threshold_upper : int
        Maximum RSSI (dBm) for a non-battery detection (id != 0)
        to be included when forming episodes. Battery detections (id == 0)
        are included regardless of RSSI. Values higher than that are considered
        too close.

    threshold_hq_lower : int
        Lower RSSI bound (dBm) for counting "high-quality" detections in
        `rssi_pct_hq`.

    threshold_hq_upper : int
        Upper RSSI bound (dBm) used for two purposes:
        - values stronger than this (rssi > threshold_hq_upper) count toward `rssi_pct_too_close`
        - values between (threshold_hq_lower, threshold_hq_upper) count toward `rssi_pct_hq`

    lag_autocorr : int
        Lag (in number of samples) for computing RSSI autocorrelation
        (`rssi_auto_corrcoef`) within each episode.

    path_id_switches: str, optional
        If provided, then switch ids for this id during that time window

    Returns
    -------
    pandas.DataFrame
        One row per contact episode per (id_src,id_tgt), typically including:
        - 'contact_start', 'contact_end' (datetime): episode boundaries
        - 'duration' (timedelta): contact_end - contact_start
        - 'n_detections' (int): number of detections in the episode

        RSSI summaries:
        - 'rssi_pct_hq' (float): fraction with threshold_hq_lower < rssi < threshold_hq_upper
        - 'rssi_pct_too_close' (float): fraction with rssi > threshold_hq_upper
        - 'rssi_auto_corrcoef' (float): lag-l autocorrelation of rssi

    Notes
    -----
    - RSSI is assumed to be in dBm; higher (less negative) values indicate
      stronger signals / closer proximity.
    - Autocorrelation may be NaN for short or constant RSSI sequences.
    """

    # first include only detections that fullfill the following criteria.
    # Battery detections will always be included
    include_detections_person = (
        (df["rssi"] > config['threshold_lower'])
        & (df["rssi"] < config['threshold_upper'])
        & (df["datetime"].dt.time >= pd.to_datetime(config['must_be_after']).time())
        & (df["datetime"].dt.time <= pd.to_datetime(config['must_be_before']).time())
        & (~df["datetime"].dt.day_name().isin(config['exclude_days']))
        & (~df["id_tgt"].eq(0))
     )
# TODO: double-check time-zone handling
    include_detections_battery = (
        (df["datetime"].dt.time >= pd.to_datetime(config['must_be_after']).time())
        & (df["datetime"].dt.time <= pd.to_datetime(config['must_be_before']).time())
        & (~df["datetime"].dt.day_name().isin(config['exclude_days']))
        & (df["id_tgt"].eq(0))
    )

    include_detections = (include_detections_person) | (include_detections_battery)
    df = df.loc[include_detections]

    # some people have switched ids during study
    if config['path_id_switches']:
        switch_ids(df,config['path_id_switches'])

    # exclude faulty days
    df = exclude_faulty_days(df)

    # create a new column for each unique id-pair (make it undirected, i.e.
    # does not matter if src-tgt or tgt-src)
    id_min = df[["id_src","id_tgt"]].min(axis=1)
    id_max = df[["id_src","id_tgt"]].max(axis=1)
    df["id_pair"] = id_min.astype(str) + "_" + id_max.astype(str)
    df = df.sort_values(["id_pair","datetime"])

    # a contact is defined as signal detection without interruption within
    # the predefined time window. We can create a new boolean column 'new_contact'
    # that is 1 whenever the gap to previous detection exceeds threshold and
    # 0 when it is still the same contact. We can then use cumsum()
    # (i.e. current value is the sum of all values before
    # which for a boolean array automatically gives us a counter)
    df['time_diff'] = df.groupby(["id_pair"])["datetime"].diff()
    df['allowed_gap'] = pd.to_timedelta(np.where(df["id_tgt"].eq(0),config['gap_batt'],config['gap_person']),unit='s')
    df["contact_new"] = df['time_diff'].isna() | (df['time_diff'] > df['allowed_gap'])
    df["contact_id"] = df.groupby(["id_pair"])["contact_new"].cumsum().astype("Int64")

    # get the number of continuous detections per contact and drop if there are less detections
    # for this contact as required per input argument
    df["n_detections"] = df.groupby(["id_pair","contact_id"]).transform("size")
    df = df.loc[df["n_detections"] >= config['min_detections']]

    # now we can compute quality measures using the rssi values for each
    # unique src-tgt pair
    # added some more measures
    df = (df.groupby(["id_pair","contact_id"], as_index=False).agg(
            id_src = ('id_src','max'),
            id_tgt = ('id_tgt','min'),
            contact_start=("datetime", "first"),
            contact_end=("datetime", "last"),
            n_detections=("datetime","size"),
            rssi_mean=("rssi","mean"),
            rssi_min=("rssi","min"),
            rssi_max=("rssi","max"),
            rssi_std=("rssi","std"),
            rssi_pct_hq=("rssi", lambda s: ((s > config['threshold_hq_lower']) & (s < config['threshold_hq_upper'])).mean()),
            rssi_pct_too_close=("rssi", lambda s: (s > config['threshold_hq_upper']).mean()),
            rssi_auto_corrcoef=("rssi", lambda s: autocorr_lag(s, lag=config['lag_autocorr'])),
            rssi_mad=("rssi", lambda s:s.diff().abs().mean()),
            
            )
    )
    df['rssi_range']= df['rssi_max']-df['rssi_mean']
    # get helper columns for following analyses
    iso = df["contact_start"].dt.isocalendar()
    duration_seconds = (df["contact_end"] - df["contact_start"]).dt.total_seconds()
    hour = df["contact_start"].dt.hour
    minute = df["contact_start"].dt.minute
    second = df["contact_start"].dt.second
    
    # get day of the week as string
    contact_day = df['contact_start'].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    contact_day = pd.Categorical(contact_day,categories=day_order,ordered=True)

    df = df.assign(
        year=iso.year,
        week=iso.week,
        year_week=iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2),
        date=df["contact_start"].dt.date,
        contact_day=contact_day,
        hour=hour,
        minute=minute,
        second=second,
        time_of_day=hour + minute / 60 + second / 3600,
        duration_seconds=duration_seconds,
        duration_minutes=duration_seconds / 60,
    )

    # drop contacts that do not have minimum duration
    df = df.loc[df['duration_seconds'] >= config['min_duration_seconds']]

    return df

def remove_faulty_interactions(df, config):
    """
    Remove faulty badge interactions and any overlapping interactions involving
    the same badges.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of badge interactions. Must contain at least the columns
        `id_src`, `id_tgt`, `contact_start`, `contact_end`,
        `rssi_mean`, `rssi_auto_corrcoef`, and `duration_minutes`.

    config : dict
        Dictionary with threshold values. Required keys are:
        - `"rssi_mean_thr"` : float
            Threshold above which `rssi_mean` is considered suspicious.    # get helper columns for following analyses
    df['year'] = df['contact_start'].dt.year
    df['week'] = df['contact_start'].dt.isocalendar().week
        - `"rssi_autocorr_thr"` : float
            Threshold above which `rssi_auto_corrcoef` is considered suspicious.
        - `"faulty_min_minutes"` : float
            Minimum interaction duration in minutes for a contact to be treated
            as faulty.
    """

    # get mask that detects faulty contacts
    # added a "likely unworn" score based on several features, and kept a hard flag for 'too_close'
    # make a distinction between faulty and long faulty contacts: faulty means, we want to delete them, 
    # long faulty means, we want to also delete all other contacts the badge had during the long faulty contact
    # mask_too_high_autocorr = df["rssi_auto_corrcoef"] > config['rssi_autocorr_thr']
    mask_non_battery = df['id_tgt'] != 0
    mask_too_close = (df["rssi_mean"] > config['rssi_mean_thr']) & mask_non_battery
    mask_too_long = df['duration_minutes'] > config['faulty_min_minutes']
    mask_short = df['duration_minutes'] < 1
    df['acf_s'] = normalize(df['rssi_auto_corrcoef'],mask_non_battery)
    df['std_s'] = normalize(df['rssi_std'],mask_non_battery, invert=True)
    df['mad_s'] = normalize(df['rssi_mad'], mask_non_battery,invert=True)
    df['range_s'] = normalize(df['rssi_range'], mask_non_battery,invert=True)
    df['dur_s'] = normalize(df['duration_seconds'],mask_non_battery)
    df['high_frac_s'] = normalize(df['rssi_pct_too_close'],mask_non_battery)
    df['unworn_score'] = (
        0.3 * df['acf_s'] +
        0.25 * df['std_s'] +
        0.20 * df['mad_s'] +
        0.10 * df['range_s'] +
        0.15 * df['dur_s']    
    )
    # for short, i.e. <1 minute contacts the above params are unreliable, thus we only use a rssi threshold
    mask_likely_unworn = (df['unworn_score']>config['unlikely_thr'])  & mask_non_battery & ~mask_short
    mask_long_faulty_contacts = (mask_too_close | mask_likely_unworn) & mask_too_long & mask_non_battery
    mask_faulty_contacts = (mask_too_close | mask_likely_unworn) & mask_non_battery

    # dataframe with only faulty contacts
    df_faulty_contacts = df[mask_long_faulty_contacts]


    # now find other time windows where interactions with these badges should be cleared
    for idx,row_faulty in df_faulty_contacts.iterrows():

        # interactions involving either participant
        ids_faulty = {row_faulty['id_src'], row_faulty['id_tgt']}
        participant_mask = ((df["id_src"].isin(ids_faulty)) | (df["id_tgt"].isin(ids_faulty))) & (df['id_tgt'] != 0)

        # check if this contact has any overlap with faulty time window
        overlap_mask = ((df["contact_start"] < row_faulty["contact_end"]) & (df["contact_end"] > row_faulty["contact_start"]))

        # update mask
        mask_faulty_contacts |= participant_mask & overlap_mask

    # drop faulty contacts
    df = df.loc[~mask_faulty_contacts].reset_index(drop=True)

    return df
def normalize(series, mask=None, invert=False):
    if mask is not None:
        s_ref = series[mask]   # only use valid data for stats
    else:
        s_ref = series

    q05 = s_ref.quantile(0.05)
    q95 = s_ref.quantile(0.95)

    s = (series - q05) / (q95 - q05)
    s = s.clip(0, 1)

    return 1 - s if invert else s

def set_patient_ids(df, path_to_csv):
    '''Creates two columns that show if either id_src or id_tgt belongs to
    patient group'''

    ids_patients = [12,18,2,19,34,37,13,7,51,53,56,50,52,57,43,44,3,8,99,85,
                    9,83,61,62,71,73,74,79,80,81,84,91,92,94,95,96,97,98]
    # add if id was from a patient or not
    # not using this due to problems with the csv file (wrong numbers)
    # ids_patients = list(pd.read_csv(path_to_csv))
    df['id_src_ispat'] = df['id_src'].isin(ids_patients)
    df['id_tgt_ispat'] = df['id_tgt'].isin(ids_patients)

    return df

def plot_contacts_3d(df,title):
    '''Plot contact df as 3d scatterplot'''
    
    df = df.copy()

    # exclude battery contacts
    df_plot = df.copy()
    df_plot = df_plot.loc[df['id_src'] != 0]
    df_plot = df_plot.loc[df['id_tgt'] != 0]

    # plot in 3D
    fig = px.scatter_3d(df_plot,
                        x='duration_minutes',
                        y='rssi_mean',
                        z='rssi_auto_corrcoef',
                        color='rssi_std',
                        hover_data=['id_src','id_tgt'],
                        title=title)
    
    return fig
