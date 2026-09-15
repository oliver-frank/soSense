from .settings import memory
import numpy as np
import pandas as pd
import bct
import networkx as nx
import plotly.express as px
import joblib

###############################################################################
## Get undirected adjacency matrices for each unit in time
###############################################################################

# backlog idea: Multiply duration with the sigmoid transformed rssi value as a weight
# --> close contacts are weighed higher than distant ones (PG)
# df_agg['duration_minutes'] = df_agg['duration_minutes'] *  rssi_to_weight(df_agg['rssi_mean'])
# def rssi_to_weight(rssi, rssi0=-70, k=0.3):
#    return 1 / (1 + np.exp(-k * (rssi - rssi0)))

# FIXME: This was in get_matrix_dict function but this should happen elsewhere
# The function below should be 'stupid', that is, it should only aggregate
# and output adjacency matrices but nothing more (JW)
# for __, g in df.groupby(groupby): 
#     df_mask = (df['date'].isin(g['date']))&(df['id_pair'].isin(g['id_pair']))
#     df.loc[df_mask, 'rssi_mean'] = (g["rssi_mean"] * g["duration_minutes"]).sum() / g["duration_minutes"].sum()
def get_active_ids_by_time(df, aggregation_minutes=None, unit_time="date"):
    df = df.copy()

    if aggregation_minutes is not None:
        df["contact_start"] = pd.to_datetime(df["contact_start"])
        time_col = f"{aggregation_minutes}min_window"
        df[time_col] = df["contact_start"].dt.floor(f"{aggregation_minutes}min")
    else:
        time_col = unit_time

    active = {}

    for t, g in df.groupby(time_col, observed=True):
        ids = pd.unique(pd.concat([g["id_src"], g["id_tgt"]]))
        active[pd.to_datetime(t)] = set(ids)

    return active
def get_matrix_dict(df,unit_time='date',outcome ='contact_duration_minutes',normalize=False,  aggregation_minutes=None,):
    '''Operates on contact dataframe and aggregates measure of choice over chosen 
    unit of time for each possible dyad pair
    aggregation_minutes:
        None -> use existing unit_time column
        1..60 -> create fixed windows from contact_start, e.g. 5-minute bins
    """
    '''
    # TODO: what is the best aggregation? Most common in the literature is sum opf interaction durations. 
    # Might be biased in our case, as staff are present less per day than patients
    # Alternatives: number of interactions, maximum length, median length, median iei
    
    # for each dyad sum all interactions this dyad had per unit of time. Also, get median iei and mean rssi, in case we want to use them later
    if aggregation_minutes is not None:
        if not (1 <= aggregation_minutes <= 60):
            raise ValueError("aggregation_minutes must be between 1 and 60")
    
        df["contact_start"] = pd.to_datetime(df["contact_start"])
        unit_col = f"{aggregation_minutes}min_window"
        df[unit_col] = df["contact_start"].dt.floor(f"{aggregation_minutes}min")
    else:
        unit_col = unit_time
    df = df.sort_values([unit_col, 'id_pair', 'contact_start'])
    df_agg = df.groupby([unit_col,'id_pair'],as_index=False,observed=True).agg({'duration_minutes':'sum','rssi_mean':'mean'})
    
    # theoretically possible connections between any dyad
    df_agg[['id_1','id_2']] = df_agg['id_pair'].str.split('_',expand=True).astype('int64')
    ids_unique = np.unique(df_agg[['id_1','id_2']].to_numpy())
    matrix_template = pd.DataFrame(data=0,index=ids_unique,columns=ids_unique,dtype=np.float64)
    matrix_template.index.name = 'id'
    matrix_template.columns.name = 'id'
    
    # iterate over chosen unit of time and fill adjacency matrix
    matrix_dict = {}
    
    for time_bin, df_time_bin in df_agg.groupby(unit_col, observed=True):        
        matrix = matrix_template.copy()
        
        for idx,row in df_time_bin.iterrows():
            
            id1 = row['id_1']
            id2 = row['id_2']
            measure = row[outcome]
            matrix.loc[id1,id2] = measure
            matrix.loc[id2,id1] = measure
            
            if normalize == True:
                matrix = bct.normalize(matrix)
        
        matrix_dict[time_bin] = matrix
    
    return matrix_dict

###############################################################################
## Compute local topological measures from adjacency matrices
###############################################################################


# TODO: We can probably use the package antropy for that? (JW)
def node_entropy(G, weight="weight", base=np.e):
    """
    Compute interaction entropy per node in a weighted undirected graph.

    Parameters
    ----------
    G : networkx.Graph
        Weighted undirected graph.
    weight : str
        Edge attribute containing weights.
    base : float
        Log base (np.e for natural log, 2 for bits).

    Returns
    -------
    dict
        {node: entropy}
    """

    entropy_dict = {}

    for node in G.nodes():

        # collect edge weights connected to the node
        weights = []
        for _, _, data in G.edges(node, data=True):
            w = data.get(weight, 1.0)
            if w > 0:
                weights.append(w)

        if len(weights) == 0:
            entropy_dict[node] = 0.0
            continue

        weights = np.array(weights, dtype=float)

        # convert to probabilities
        p = weights / weights.sum()

        # compute entropy
        entropy = -np.sum(p * np.log(p))

        if base != np.e:
            entropy /= np.log(base)

        entropy_dict[node] = float(entropy)

    return entropy_dict


def normalize_graph_minmax(G):
    """
    Min-max normalize edge weights of G in place to [0, 1]. 
    Assumes all edges have a "weight" attribute.
    """
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    w_min, w_max = min(weights), max(weights)
    for _, _, d in G.edges(data=True):
        d["weight"] = (d["weight"] - w_min) / (w_max - w_min)

    return G

def get_local_measure(df, unit_time,matrix,measure_name,matrix_threshold=None,normalize=False, prev_matrix=None,active_ids=None,aggregation_minutes=None):
    """
    Compute one local topological measure for a single adjacency matrix.
    Each sub-function must return a dictionary with the node ids as 
    keys and the local topological values as values.

    Returns
    -------
    tuple (unit_time, pandas.Series)
        Series index matches matrix.index.
    """

    # get series of NaNs to return if measure cannot be computed
    index = matrix.index
    nan_result = pd.Series(np.nan, index=index, dtype=float)
    matrix = matrix.copy().astype(float).fillna(0)    # threshold matrix if specified
    if active_ids is not None:
        active_ids = set(active_ids)
        inactive_ids = [node for node in matrix.index if node not in active_ids]
        matrix.loc[inactive_ids, :] = 0
        matrix.loc[:, inactive_ids] = 0
    # Build graph once
    G = nx.from_pandas_adjacency(matrix)

    # Remove idle nodes
    # isolates = list(nx.isolates(G))
    
    if matrix_threshold is not None:
        arr = matrix.to_numpy(copy=True)
        mask = np.triu(np.ones(arr.shape, dtype=bool), k=1)
        vals = arr[mask]
        vals = vals[np.isfinite(vals) & (vals != 0)]
    
        if len(vals) == 0:
            return unit_time, nan_result
    
        cutoff = np.nanpercentile(vals, matrix_threshold)
        matrix = matrix.mask(matrix < cutoff, 0)

    if normalize == 'min-max':
        G = normalize_graph_minmax(G)
    # a second time in case a node loses its edges due to thresholding, but shlould still be present 
    G = nx.from_pandas_adjacency(matrix)
    # if isolates:
    #     G.remove_nodes_from(isolates)
    if active_ids is not None:
       G.remove_nodes_from([node for node in G.nodes if node not in active_ids])

    # return series of nans if nothing to compute
    if G.number_of_nodes() == 0:
        return unit_time, nan_result

    # Louvain communities
    comms = nx.community.louvain_communities(
        G,
        weight='weight',
        resolution=1,
        threshold=1e-07,
        max_level=None,
        seed=None
    )

    # node -> community lookup
    node_to_comm = {
        node: i
        for i, comm in enumerate(comms)
        for node in comm
    }

    # global modularity
    modularity_val = nx.community.modularity(
        G,
        comms,
        weight='weight'
    )
    for u, v, data in G.edges(data=True):
        data['distance'] = 1 / data['weight']
    def duration_weighted_rssi_by_comm(df, unit_time, G, node_to_comm, aggregation_minutes=None):
        if aggregation_minutes is None:
            start = pd.to_datetime(unit_time).tz_localize("Europe/Berlin")
            end = start.normalize() + pd.Timedelta(hours=20)
        else:
            start = pd.to_datetime(unit_time).tz_localize("Europe/Berlin")
            end = start + pd.Timedelta(minutes=aggregation_minutes)
    
        contacts = df.copy()
        contacts["contact_start"] = pd.to_datetime(contacts["contact_start"])
        contacts["contact_end"] = pd.to_datetime(contacts["contact_end"])
    
        contacts = contacts[
            (contacts["contact_start"] < end) &
            (contacts["contact_end"] > start)
        ]
    
        contacts = contacts.dropna(subset=["id_src", "id_tgt", "rssi_mean", "duration_minutes"])
    
        nodes = set(G.nodes)
        in_vals = {n: [] for n in G.nodes}
        out_vals = {n: [] for n in G.nodes}
    
        for row in contacts.itertuples(index=False):
            u = row.id_src
            v = row.id_tgt
    
            if u not in nodes or v not in nodes:
                continue
    
            if not G.has_edge(u, v):
                continue
    
            rssi = float(row.rssi_mean)
            dur = float(row.duration_minutes)
    
            if dur <= 0:
                continue
    
            same_comm = node_to_comm[u] == node_to_comm[v]
    
            if same_comm:
                in_vals[u].append((rssi, dur))
                in_vals[v].append((rssi, dur))
            else:
                out_vals[u].append((rssi, dur))
                out_vals[v].append((rssi, dur))
    
        def weighted_mean(vals):
            if not vals:
                return np.nan
            rssis = np.array([x[0] for x in vals], dtype=float)
            durs = np.array([x[1] for x in vals], dtype=float)
            return np.average(rssis, weights=durs)
    
        rssi_in = {n: weighted_mean(in_vals[n]) for n in G.nodes}
        rssi_out = {n: weighted_mean(out_vals[n]) for n in G.nodes}
    
        rssi_ratio = {}
        for n in G.nodes:
            a = rssi_in[n]
            b = rssi_out[n]
    
            if np.isnan(a) or np.isnan(b):
                rssi_ratio[n] = np.nan
            else:
                # safer for negative RSSI values
                denom = abs(a) + abs(b)
                rssi_ratio[n] = np.nan if denom == 0 else (a - b) / denom
    
        return rssi_in, rssi_out, rssi_ratio
        
    try:

        if measure_name == "strengths":
            d = dict(G.degree(weight="weight"))

        elif measure_name == "degree":
            d = dict(G.degree(weight=None))

        elif measure_name == "clustering-coefficient":
            d = nx.clustering(G, weight="weight")

        elif measure_name == "clustering-coefficient_uw":
            d = nx.clustering(G, weight=None)

        elif measure_name == "weighted-jaccard-prev-day":
            if prev_matrix is None:
                d = nan_result
            else:
                d = temporal_weighted_jaccard(matrix, prev_matrix)

        elif measure_name == "eigenvector-centrality":
            d = nx.eigenvector_centrality(G, weight="weight")
        elif measure_name == "katz_centrality":
            d = nx.katz_centrality(G, weight="weight")
        elif measure_name == "betweenness_centrality":
            d = nx.betweenness_centrality(G, weight="distance", normalized=True, endpoints=False)
        elif measure_name == "closeness_centrality":
            d = nx.closeness_centrality(G, weight="distance")
        elif measure_name == "eigenvector-centrality_uw":
            d = nx.eigenvector_centrality(G, weight=None)
        elif measure_name == "current_flow_closeness_centrality":
            d = nx.current_flow_closeness_centrality(G, weight="weight")
        elif measure_name == "laplacian_centrality":
            d = nx.laplacian_centrality(G,normalized=True, weight="weight")
        elif measure_name == "entropy":
            d = node_entropy(G, weight="weight")

        elif measure_name == "partner-concentration":
            d = partner_concentration_dict(G, weight="weight")

        elif measure_name == "rich_club":
            rc = nx.rich_club_coefficient(G, normalized=False)
            d =  {
                    n: rc.get(G.degree(n) - 1, float("nan"))
                    for n in G.nodes()
                }
        elif measure_name == "network_size":
            d = pd.Series(len(G.nodes), index=G.nodes, dtype=float)

        elif measure_name == "k_core":
            d = nx.core_number(G)
        elif measure_name == "density":
            d = pd.Series(nx.density(G), index=G.nodes, dtype=float)
        elif measure_name == "n_comms":
            d = pd.Series(len(comms), index=G.nodes, dtype=float)

        # -----------------------------
        # COMMUNITY-BASED MEASURES
        # -----------------------------
        elif measure_name in ["rssi_comm_in", "rssi_comm_out", "rssi_comm_ratio"]:
            rssi_in, rssi_out, rssi_ratio = duration_weighted_rssi_by_comm(
                df=df,
                unit_time=unit_time,
                G=G,
                node_to_comm=node_to_comm,
                aggregation_minutes=aggregation_minutes
            )
        
            if measure_name == "rssi_comm_in":
                d = rssi_in
        
            elif measure_name == "rssi_comm_out":
                d = rssi_out
        
            elif measure_name == "rssi_comm_ratio":
                d = rssi_ratio
        elif measure_name == "community-id":
            d = node_to_comm

        elif measure_name == "modularity":
            d = {node: modularity_val for node in G.nodes}

        elif measure_name == "avg_weight_community":

            d = {}

            for node in G.nodes:

                same_weights = []

                for nbr, edge_data in G[node].items():

                    w = edge_data.get("weight", 1)

                    if node_to_comm[nbr] == node_to_comm[node]:
                        same_weights.append(w)

                d[node] = (
                    np.mean(same_weights)
                    if len(same_weights) > 0
                    else np.nan
                )

        elif measure_name == "avg_weight_noncommunity":

            d = {}

            for node in G.nodes:

                other_weights = []

                for nbr, edge_data in G[node].items():

                    w = edge_data.get("weight", 1)

                    if node_to_comm[nbr] != node_to_comm[node]:
                        other_weights.append(w)

                d[node] = (
                    np.mean(other_weights)
                    if len(other_weights) > 0
                    else np.nan
                )

        elif measure_name == "community_ratio":

            d = {}

            for node in G.nodes:

                same_weights = []
                other_weights = []

                for nbr, edge_data in G[node].items():

                    w = edge_data.get("weight", 1)

                    if node_to_comm[nbr] == node_to_comm[node]:
                        same_weights.append(w)
                    else:
                        other_weights.append(w)

                sum_same = (
                    np.nanmean(same_weights)
                    if len(same_weights) > 0
                    else np.nan
                )

                sum_other = (
                    np.nanmean(other_weights)
                    if len(other_weights) > 0
                    else np.nan
                )

                if np.isnan(sum_other) or sum_other == 0:
                    ratio = np.nan
                else:
                    ratio = (sum_same-sum_other) / (sum_same+sum_other)

                d[node] = ratio

    except:
        d = nan_result

    # Reindex to original node list so removed isolated nodes appear as NaN
    series = pd.Series(d, dtype=float).reindex(index)

    return unit_time, series

def get_local_measures_df(
    df,
    matrix_dict,
    measure_name,
    matrix_threshold=None,
    aggregation_minutes=None,
    normalize=False,
    active_ids_by_time=None,
    **joblib_kwargs
):
    matrix_by_time = {
        pd.to_datetime(k): v
        for k, v in matrix_dict.items()
    }

    items = sorted(matrix_by_time.items())

    if aggregation_minutes is None:
        prev_delta = pd.Timedelta(days=1)
    else:
        prev_delta = pd.Timedelta(minutes=aggregation_minutes)

    pairs = []
    for unit_time, matrix in items:
        prev_time = unit_time - prev_delta
        prev_matrix = matrix_by_time.get(prev_time, None)
        active_ids = None
        if active_ids_by_time is not None:
            active_ids = active_ids_by_time.get(unit_time, None)

        pairs.append((unit_time, matrix, prev_matrix, active_ids))

    results = joblib.Parallel(**joblib_kwargs)(
        joblib.delayed(get_local_measure)(
            df,
            unit_time,
            matrix,
            measure_name,
            matrix_threshold,
            normalize,
            prev_matrix,
            active_ids,
            aggregation_minutes
        )
        for unit_time, matrix, prev_matrix, active_ids in pairs
    )

    return pd.DataFrame({unit_time: result for unit_time, result in results})

@memory.cache
def _make_subject_focused_contacts(df):
    """
    Convert undirected dyadic interactions into a subject-focused dataframe.

    Output columns include:
    - id: focal participant
    - partner_id: the other participant
    - partner_ispat: whether the other participant is a patient
    """
    # focal = source, partner = target
    df_src = (
        df.rename(
            columns={
                "id_src": "id",
                "id_tgt": "partner_id",
                "id_tgt_ispat": "partner_ispat",
                "id_src_ispat": "id_ispat",
            }
        )
        .copy()
    )

    # focal = target, partner = source
    df_tgt = (
        df.rename(
            columns={
                "id_tgt": "id",
                "id_src": "partner_id",
                "id_src_ispat": "partner_ispat",
                "id_tgt_ispat": "id_ispat",
            }
        )
        .copy()
    )

    df_stacked = pd.concat([df_src, df_tgt], ignore_index=True)

    return df_stacked


def _compute_concurrent_contact_breakdown(df_group):
    """
    For one focal participant on one day, decompose the observed contact time into:
    - group_interaction_staff_minutes
    - group_interaction_no_staff_minutes
    - dyadic_interaction_staff_minutes
    - dyadic_interaction_patient_minutes

    PLUS:
    - median duration of contributing segments for each category

    Group = at least 2 concurrent partners active.
    Dyadic = exactly 1 partner active.

    Classification is based on the active partner set in each time segment.
    """
    import numpy as np
    import pandas as pd

    empty_result = {
        "group_interaction_staff_minutes": np.nan,
        "group_interaction_no_staff_minutes": np.nan,
        "dyadic_interaction_staff_minutes": np.nan,
        "dyadic_interaction_patient_minutes": np.nan,

        "group_interaction_staff_median_minutes": np.nan,
        "group_interaction_no_staff_median_minutes": np.nan,
        "dyadic_interaction_staff_median_minutes": np.nan,
        "dyadic_interaction_patient_median_minutes": np.nan,
    }

    if df_group.empty:
        return empty_result

    # collect all interval boundaries
    boundaries = pd.Index(
        sorted(
            set(df_group["contact_start"]).union(set(df_group["contact_end"]))
        )
    )

    if len(boundaries) < 2:
        return {
            k: 0.0 for k in empty_result.keys()
        }

    # cumulative sums
    group_staff = 0.0
    group_no_staff = 0.0
    dyadic_staff = 0.0
    dyadic_patient = 0.0

    # store segment durations for medians
    group_staff_segments = []
    group_no_staff_segments = []
    dyadic_staff_segments = []
    dyadic_patient_segments = []

    # iterate over adjacent half-open segments [left, right)
    for left, right in zip(boundaries[:-1], boundaries[1:]):

        seg_minutes = (right - left).total_seconds() / 60

        if seg_minutes <= 0:
            continue

        # active intervals on this segment
        active = df_group[
            (df_group["contact_start"] < right) &
            (df_group["contact_end"] > left)
        ]

        if active.empty:
            continue

        # active unique partners during this segment
        active_partners = active[
            ["partner_id", "partner_ispat"]
        ].drop_duplicates()

        n_active_partners = len(active_partners)

        if n_active_partners >= 2:

            # group interaction
            any_staff = (~active_partners["partner_ispat"].astype(bool)).any()

            if any_staff:
                group_staff += seg_minutes
                group_staff_segments.append(seg_minutes)

            else:
                group_no_staff += seg_minutes
                group_no_staff_segments.append(seg_minutes)

        elif n_active_partners == 1:

            # dyadic interaction
            partner_is_patient = bool(
                active_partners["partner_ispat"].iloc[0]
            )

            if partner_is_patient:
                dyadic_patient += seg_minutes
                dyadic_patient_segments.append(seg_minutes)

            else:
                dyadic_staff += seg_minutes
                dyadic_staff_segments.append(seg_minutes)

    return {
        "group_interaction_staff_minutes": group_staff,
        "group_interaction_no_staff_minutes": group_no_staff,
        "dyadic_interaction_staff_minutes": dyadic_staff,
        "dyadic_interaction_patient_minutes": dyadic_patient,

        "group_interaction_staff_median_minutes":
            np.median(group_staff_segments)
            if group_staff_segments else 0.0,

        "group_interaction_no_staff_median_minutes":
            np.median(group_no_staff_segments)
            if group_no_staff_segments else 0.0,

        "dyadic_interaction_staff_median_minutes":
            np.median(dyadic_staff_segments)
            if dyadic_staff_segments else 0.0,

        "dyadic_interaction_patient_median_minutes":
            np.median(dyadic_patient_segments)
            if dyadic_patient_segments else 0.0,
    }


def compute_daily_interaction_stats(df):
    """
    Compute daily interaction statistics for each participant.

    Metrics are computed from merged interaction intervals so that
    overlapping interactions count only once where appropriate.

    Additional outputs:
    - staff_interaction_share:
        total dyadic duration with staff / total dyadic duration with everyone
        (a strength-share style metric based on summed edge weights)
    - group_interaction_staff_minutes:
        time in concurrent group interaction (>=2 partners) where at least one partner is staff
    - group_interaction_no_staff_minutes:
        time in concurrent group interaction (>=2 partners) where all partners are patients
    - dyadic_interaction_staff_minutes:
        time in dyadic interaction with a staff member and no concurrent other interaction
    - dyadic_interaction_patient_minutes:
        time in dyadic interaction with a patient and no concurrent other interaction

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing undirected dyadic interactions.
        Expected columns:
        ['id_pair', 'id_src', 'id_tgt', 'contact_start', 'contact_end',
         'duration_seconds', 'duration_minutes', 'date',
         'id_src_ispat', 'id_tgt_ispat']
    """

    df = df.copy()

    # get contacts to battery in a separate dataframe
    df_battery = df.loc[df["id_tgt"] == 0].copy()
    df_battery = df_battery.rename(columns={"id_src": "id"})

    # exclude battery rows from social interaction dataframe
    df = df.loc[df["id_tgt"] != 0].copy()

    # convert undirected dyadic interactions to subject-focused contacts
    df_stacked = _make_subject_focused_contacts(df)

    daily_stats = []

    # process each combination of id and day
    for (id_group, day), df_group in df_stacked.groupby(["id", "date"]):
        df_group = df_group.sort_values("contact_start").copy()

        contact_starts = df_group["contact_start"]
        contact_ends = df_group["contact_end"]
        any_dur_median = np.median(df_group['duration_minutes'])
        any_inter_event = df_group['contact_start'].diff().dt.total_seconds().dropna()
        any_burstiness = burstiness(any_inter_event)
        any_intercontact_interval_minutes_median = np.median(any_inter_event)
        # 1) timespan: first to last observed contact
        contact_first = contact_starts.min()
        contact_last = contact_ends.max()
        contact_span_minutes = (contact_last - contact_first).total_seconds() / 60

        # 2) merged intervals ignoring overlap across all contacts
        merged = []
        current_start, current_end = contact_starts.iloc[0], contact_ends.iloc[0]

        for next_start, next_end in zip(contact_starts.iloc[1:], contact_ends.iloc[1:]):
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged.append((current_start, current_end))

        # durations of merged interaction periods
        durations_minutes = [
            (end - start).total_seconds() / 60 for start, end in merged
        ]
        duration_minutes_sum = sum(durations_minutes)
        duration_minutes_median = np.median(durations_minutes)

        # gaps between merged interaction periods
        if len(merged)>1:
            interevent_intervals = [
                (merged[i][0] - merged[i - 1][1]).total_seconds() / 60
                for i in range(1, len(merged))
                ]
        else:
            interevent_intervals = np.nan
        bursty = burstiness(interevent_intervals)
        
        
        intercontact_interval_minutes_median = (
            np.median(interevent_intervals)
            if len(merged) > 1
            else np.nan
        )

        # rssi weighted by raw dyadic duration
        rssi_mean_weighted = (
            (df_group["rssi_mean"] * df_group["duration_minutes"]).sum()
            / df_group["duration_minutes"].sum()
            if df_group["duration_minutes"].sum() > 0
            else np.nan
        )

        # 3) ids present on that day
        ids_present = pd.concat([
            df.loc[df["date"] == day, "id_src"],
            df.loc[df["date"] == day, "id_tgt"]
        ]).nunique()
        ids_present_pat = pd.concat([
            df.loc[(df["date"] == day)&(df["id_src_ispat"] == True), "id_src"],
            df.loc[(df["date"] == day)&(df["id_tgt_ispat"] == True), "id_tgt"]
        ]).nunique()

        # 4) staff interaction share
        # "strength" interpretation: sum of raw dyadic duration with staff /
        # sum of raw dyadic duration with all partners
        total_dyadic_duration = df_group["duration_minutes"].sum()

        staff_dyadic_duration = df_group.loc[
            ~df_group["partner_ispat"].astype(bool), "duration_minutes"
        ].sum()

        staff_interaction_share = (
            staff_dyadic_duration / total_dyadic_duration
            if total_dyadic_duration > 0
            else np.nan
        )

        # 5) concurrent-time decomposition
        concurrent_breakdown = _compute_concurrent_contact_breakdown(df_group)

        daily_stats.append(
            {
                "id": id_group,
                "date": day,
                "contact_span_minutes": contact_span_minutes,
                "duration_minutes_sum": duration_minutes_sum,
                "duration_minutes_median": duration_minutes_median,
                "intercontact_interval_minutes_median": intercontact_interval_minutes_median,
                "ids_present": ids_present,
                "rssi_mean_weighted": rssi_mean_weighted,
                "staff_interaction_share": staff_interaction_share,
                "ids_present_pat":ids_present_pat,
                "any_duration_minutes_median":any_dur_median,
                "any_burstiness":any_burstiness,
                "any_intercontact_interval_minutes_median":any_intercontact_interval_minutes_median,  
                "burstiness":bursty,
                
                **concurrent_breakdown,
            }
        )

    df_daily_stats = pd.DataFrame(daily_stats)

    # add contacts with battery at each day
    df_battery = (
        df_battery.groupby(["id", "date"], as_index=False)
        .agg({"id": "min", "date": "first", "duration_minutes": "sum"})
        .rename(columns={"duration_minutes": "duration_minutes_sum_battery"})
    )

    df_daily_stats = pd.merge(
        df_daily_stats,
        df_battery,
        on=["id", "date"],
        how="left"
    )

    # compute relative interaction time using first-to-last-contact span
    df_daily_stats["duration_minutes_relative"] = (
        df_daily_stats["duration_minutes_sum"] / df_daily_stats["contact_span_minutes"]
    )

    return df_daily_stats

###############################################################################
## Plotting
###############################################################################

@memory.cache
def plot_contacts_per_day_and_week(df,outcome):
    '''Plot contacts as interactive facet grid. Days as columns and study 
    weeks as rows. For each contact pair draw time of the day against outcome. Needs
    columns: year_week, contact_day,contact_start and outcome'''
    
    # get copy of input
    df = df.copy()
    
    # sort by study week then by day
    df = df.sort_values(['year_week','contact_day'])

    # get contact time as seconds that have passed since midnight on that day
    df["time"] = (
    df["contact_start"].dt.hour * 3600
    + df["contact_start"].dt.minute * 60
    + df["contact_start"].dt.second
    )
    
    # plot as line plots
    fig = px.line(
        df,
        x="time",
        y=outcome,
        facet_row="year_week",
        facet_col="contact_day",
        color="id_pair",
        category_orders={"contact_day":["Monday","Tuesday","Wednesday","Thursday","Friday"]},
        facet_row_spacing=0.001
    )
    
    # map total seconds to strings
    start = 8 * 3600
    end   = 20 * 3600
    tickvals = list(range(start,end + 1, 3600))
    ticktext = [f"{h:02d}:00" for h in range(8, 21)]
    
    fig.update_xaxes(
        range=[start,end],           # always 08:00–20:00
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    
    fig.update_yaxes(autorange=True)
    fig.update_layout(height=df["year_week"].nunique() * 140)
    
    return fig

###############################################################################
## Testing
###############################################################################
import matplotlib.pyplot as plt

def draw_weighted_graph_with_patients(
    G,
    patient_nodes,
    weight_attr="weight",
    layout_func=nx.spring_layout,
    patient_color="red",
    other_color="lightblue",
    edge_color="gray",
    min_alpha=0.1,
    max_alpha=1.0,
    min_width=1,
    max_width=5,
    seed=42
):
    """
    Draw weighted graph with patient nodes highlighted.

    Parameters
    ----------
    G : networkx.Graph
    patient_nodes : list or set
        Nodes to highlight (e.g., patients)
    """

    pos = layout_func(G, seed=seed)

    # --- Edge weights ---
    edges = list(G.edges())
    weights = [G[u][v].get(weight_attr, 1) for u, v in edges]

    w_min, w_max = min(weights), max(weights)
    if w_max == w_min:
        norm_w = [1 for _ in weights]
    else:
        norm_w = [(w - w_min) / (w_max - w_min) for w in weights]

    widths = [min_width + (max_width - min_width) * n for n in norm_w]
    alphas = [min_alpha + (max_alpha - min_alpha) * n for n in norm_w]

    # --- Node colors ---
    patient_set = set(patient_nodes)
    node_colors = [
        patient_color if node in patient_set else other_color
        for node in G.nodes()
    ]

    # --- Draw nodes ---
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos)

    # --- Draw edges (with varying alpha) ---
    for (u, v), w, a in zip(edges, widths, alphas):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=w,
            alpha=a,
            edge_color=edge_color
        )

    # --- Legend ---
    import matplotlib.patches as mpatches
    patient_patch = mpatches.Patch(color=patient_color, label="Patient")
    other_patch = mpatches.Patch(color=other_color, label="Other")
    plt.legend(handles=[patient_patch, other_patch])

    plt.axis("off")
    plt.title("Weighted Graph with Patient Nodes Highlighted")
    plt.show()
# only for testing:
def weighted_social_turnover(graphs, weight="weight"):
    """
    Compute weighted social turnover between consecutive graphs.

    Parameters
    ----------
    graphs : dict
        {time: networkx.Graph}
    weight : str
        edge attribute containing weights

    Returns
    -------
    dict
        {(node, time): turnover}
    """

    times = sorted(graphs.keys())
    turnover = {}

    for t_idx in range(1, len(times)):

        t_prev = times[t_idx - 1]
        t_curr = times[t_idx]

        G_prev = graphs[t_prev]
        G_curr = graphs[t_curr]

        nodes = set(G_prev.nodes()).union(G_curr.nodes())

        for node in nodes:

            # collect weights for partners
            w_prev = {}
            if node in G_prev:
                for nbr, data in G_prev[node].items():
                    w_prev[nbr] = data.get(weight, 1.0)

            w_curr = {}
            if node in G_curr:
                for nbr, data in G_curr[node].items():
                    w_curr[nbr] = data.get(weight, 1.0)

            partners = set(w_prev) | set(w_curr)

            num = 0.0
            den = 0.0

            for p in partners:

                wp = w_prev.get(p, 0)
                wc = w_curr.get(p, 0)

                num += min(wp, wc)
                den += max(wp, wc)

            if den == 0:
                value = None
            else:
                value = 1 - num / den

            turnover[(node, t_curr)] = value

    return turnover
from collections import Counter

def plot_degree_distribution_discrete(G):
    degrees = [d for _, d in G.degree()]
    count = Counter(degrees)

    deg, freq = zip(*sorted(count.items()))

    plt.figure()
    plt.scatter(deg, freq)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution (Discrete)")
    plt.grid(True)
    plt.show()

def plot_edge_weight_distribution(G, weight_attr="weight", bins=10, log_scale=False):
    """
    Plot histogram of edge weights.

    Parameters
    ----------
    G : networkx.Graph
    weight_attr : str
        Edge attribute name for weights
    bins : int
        Number of histogram bins
    log_scale : bool
        Whether to use log scale on y-axis
    """

    # Extract weights (default = 1 if missing)
    weights = [G[u][v].get(weight_attr, 1) for u, v in G.edges()]

    plt.figure()
    plt.hist(weights, bins=bins)

    plt.xlabel("Edge Weight")
    plt.ylabel("Frequency")
    plt.title("Edge Weight Distribution")

    if log_scale:
        plt.yscale("log")

    plt.grid(True)
    plt.show()
def keep_clean_patient_patient(
    df: pd.DataFrame,
    start_col: str = "contact_start",
    end_col: str = "contact_end",
    src_col: str = "id_src",
    tgt_col: str = "id_tgt",
    src_ispat_col: str = "id_src_ispat",
    tgt_ispat_col: str = "id_tgt_ispat",
) -> pd.DataFrame:
    """
    Keep only patient-patient interaction segments, cutting out any portions where
    either participant is simultaneously interacting with staff.

    All original columns are preserved. When an interval is split into multiple
    pieces, the non-time columns are copied unchanged into each piece.
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    original_columns = df.columns.tolist()

    # patient-patient interactions to keep/split
    pp = df[df[src_ispat_col] & df[tgt_ispat_col]].copy()

    # patient-staff interactions: exactly one side is patient
    ps = df[df[src_ispat_col] ^ df[tgt_ispat_col]].copy()

    # identify the patient in each patient-staff interaction
    ps["patient"] = ps[src_col].where(ps[src_ispat_col], ps[tgt_col])

    def merge_intervals(intervals):
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]

        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        return [(s, e) for s, e in merged]

    def subtract_intervals(base_start, base_end, cuts):
        if not cuts:
            return [(base_start, base_end)]

        out = []
        cur = base_start

        for s, e in cuts:
            if s > cur:
                out.append((cur, s))
            cur = max(cur, e)

        if cur < base_end:
            out.append((cur, base_end))

        return out

    out_rows = []

    for _, r in pp.iterrows():
        p1, p2 = r[src_col], r[tgt_col]
        start, end = r[start_col], r[end_col]

        overlaps = ps[
            ps["patient"].isin([p1, p2]) &
            (ps[start_col] < end) &
            (ps[end_col] > start)
        ]

        cuts = [
            (max(start, s), min(end, e))
            for s, e in zip(overlaps[start_col], overlaps[end_col])
        ]

        cuts = merge_intervals(cuts)
        kept_parts = subtract_intervals(start, end, cuts)

        for ks, ke in kept_parts:
            if ks < ke:
                new_row = r.copy()   # keeps all original columns
                new_row[start_col] = ks
                new_row[end_col] = ke
                out_rows.append(new_row)

    if out_rows:
        result = pd.DataFrame(out_rows)
        result = result[original_columns]  # preserve original column order
        result = result.sort_values([start_col, end_col, src_col, tgt_col]).reset_index(drop=True)
    else:
        result = df.iloc[0:0][original_columns].copy()

    return result

def burstiness(inter_event_times):
    mu = np.mean(inter_event_times)
    sigma = np.std(inter_event_times)
    return (sigma - mu) / (sigma + mu)
def partner_concentration_dict(G, weight="weight", nan_for_isolates=True):
    """
    Compute partner concentration for each node in a weighted undirected graph.

    partner_concentration_i = max_j w_ij / sum_j w_ij

    Parameters
    ----------
    G : networkx.Graph
        Weighted undirected graph.
    weight : str
        Edge attribute name for the weight.
    nan_for_isolates : bool
        If True, isolates get np.nan.
        If False, isolates get 0.0.

    Returns
    -------
    dict
        {node: partner_concentration}
    """
    out = {}

    for node in G.nodes():
        weights = [
            data.get(weight, 1.0)
            for _, _, data in G.edges(node, data=True)
        ]

        if len(weights) == 0:
            out[node] = np.nan if nan_for_isolates else 0.0
            continue

        total = sum(weights)
        out[node] = max(weights) / total if total > 0 else np.nan

    return out
def temporal_weighted_jaccard(current_matrix, previous_matrix):
    """
    Compute node-wise weighted Jaccard similarity between two adjacency matrices.

    For each node i:
        sum_k min(w_i,k(current), w_i,k(previous))
        ------------------------------------------
        sum_k max(w_i,k(current), w_i,k(previous))

    Returns
    -------
    pandas.Series
        Indexed by node id.
    """
    # union of node sets
    nodes = current_matrix.index.union(previous_matrix.index)

    # align both matrices to same node set
    cur = current_matrix.reindex(index=nodes, columns=nodes, fill_value=0.0).astype(float)
    prev = previous_matrix.reindex(index=nodes, columns=nodes, fill_value=0.0).astype(float)

    # remove self-ties if diagonal exists
    # np.fill_diagonal(cur.values, 0.0)
    # np.fill_diagonal(prev.values, 0.0)

    mins = np.minimum(cur.values, prev.values).sum(axis=1)
    maxs = np.maximum(cur.values, prev.values).sum(axis=1)

    out = np.divide(
        mins,
        maxs,
        out=np.full(len(nodes), np.nan, dtype=float),
        where=maxs > 0
    )

    return pd.Series(out, index=nodes, dtype=float)