import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os
from pathlib import Path
import datetime as dt
os.chdir(Path(__file__).parent)
df = pd.read_pickle('../data/exchange/df_contacts.pkl')
df = df.loc[df['date']==dt.date(2023,6,21)]
df = df.loc[df['id_tgt']!=0]


# G1 = nx.from_pandas_adjacency(matrix_dict[dt.date(2023,6,21)])
# isolates = list(nx.isolates(G1))
# if isolates:
#     G1.remove_nodes_from(isolates)


# ----------------------------
# PREP
# ----------------------------

# Ensure datetime
df['contact_start'] = pd.to_datetime(df['contact_start'])
df['contact_end'] = pd.to_datetime(df['contact_end'])

# Optional filtering
# Keep only meaningful contacts
# df = df[df['duration_seconds'] >= 30]

# Optional RSSI quality filter
# df = df[df['rssi_mean'] > -80]

# Sort
df = df.sort_values('contact_start')

# ----------------------------
# BUILD NODE ROLE MAP
# ----------------------------

role_map = {}

for _, row in df.iterrows():

    src_role = 'patient' if row['id_src_ispat'] else 'staff'
    tgt_role = 'patient' if row['id_tgt_ispat'] else 'staff'

    role_map[row['id_src']] = src_role
    role_map[row['id_tgt']] = tgt_role

# All nodes
nodes = sorted(
    set(df['id_src']).union(set(df['id_tgt']))
)

# ----------------------------
# STABLE LAYOUT
# ----------------------------

G_all = nx.Graph()
G_all.add_nodes_from(nodes)

all_edges = df[['id_src', 'id_tgt']].drop_duplicates()

G_all.add_edges_from(
    all_edges.itertuples(index=False, name=None)
)

# Initial stable positions
pos = nx.spring_layout(G_all, seed=42, k=1.5)
pos_holder = {'pos': pos}
# ----------------------------
# TIME FRAMES
# ----------------------------

# Round to minute for animation frames
df['frame_time'] = df['contact_start'].dt.floor('1min')

frames = sorted(df['frame_time'].unique())
frame_times = pd.date_range(
    df['contact_start'].min().floor('min'),
    df['contact_end'].max().ceil('min'),
    freq='30s'
)
# ----------------------------
# FIGURE
# ----------------------------

fig, ax = plt.subplots(figsize=(12, 10))

# ----------------------------
# ANIMATION FUNCTION
# ----------------------------

def update(frame_time):
    global pos

    ax.clear()

    current = df[
        (df['contact_start'] <= frame_time) &
        (df['contact_end'] >= frame_time)
    ]

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for _, row in current.iterrows():

        duration = row['duration_seconds']

        # 300 = 5 minutes
        # Increase this if the movement is too strong
        # Decrease it if movement is too weak
        strength = max(0.01, duration / 300)

        G.add_edge(
            row['id_src'],
            row['id_tgt'],
            weight=strength,
            duration=duration
        )
        if G.number_of_edges() > 0:
            new_pos = nx.spring_layout(
                G,
                pos=pos_holder['pos'],
                weight='weight',
                iterations=5,
                k=1.2,
                seed=42
            )
        
            alpha = 0.25
        
            pos_holder['pos'] = {
                n: (
                    (1 - alpha) * pos_holder['pos'][n][0] + alpha * new_pos[n][0],
                    (1 - alpha) * pos_holder['pos'][n][1] + alpha * new_pos[n][1]
                )
                for n in pos_holder['pos']
            }

    node_colors = [
        'tab:blue' if role_map.get(n) == 'patient'
        else 'tab:orange'
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.0,
        alpha=0.35,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        ax=ax
    )

    ax.set_title(
        f'Contact Network\n'
        f'{pd.Timestamp(frame_time)}\n'
        f'Active contacts: {len(current)}'
    )

    ax.axis('off')
# ----------------------------
# CREATE VIDEO
# ----------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=frame_times,
    interval=100
)

writer = FFMpegWriter(fps=10)
# or PillowWriter(fps=10)

ani.save(
    'hospital_contact_network.mp4',
    writer=writer
)

plt.close()

print("Saved: hospital_contact_network.mp4")