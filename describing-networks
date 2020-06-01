# Configure plotting in Jupyter
from matplotlib import pyplot as plt
# matplotlib inline
plt.rcParams.update({
    'figure.figsize': (7.5, 7.5)})
# Seed random number generator
from numpy import random as nprand
seed = hash("Network Science in Python") % 2**32
nprand.seed(seed)

import networkx as nx

from pathlib import Path
data_dir = Path('.') / 'data'

# Load karate club network
G_karate = nx.karate_club_graph()
mr_hi = 0
john_a = 33

# Load internet point of presence network
G_internet = nx.read_graphml(data_dir / 'UAITZ' / 'Geant2012.graphml')

# Load Germany electrical grid
with open(data_dir / 'mureddu2016' / '0.2' / 'branches.csv', 'rb') as f:
    # Skip header
    next(f)
    # Read edgelist format
    G_electric = nx.read_edgelist(
        f,
        delimiter="\t",
        create_using=nx.Graph,
        data=[('X', float), ('Pmax', float)])
    
plt.figure(figsize=(7.5, 2.75))
plt.subplot(1, 3, 1);
plt.title("Karate")
nx.draw_networkx(G_karate, node_size=0, with_labels=False)
plt.subplot(1, 3, 2)
plt.title("Electric")
nx.draw_networkx(G_electric, node_size=0, with_labels=False)
plt.subplot(1, 3, 3)
plt.title("Internet")
nx.draw_networkx(G_internet, node_size=0, with_labels=False)
plt.tight_layout()

list(nx.all_shortest_paths(G_karate, mr_hi, john_a))

nx.shortest_path(G_karate, mr_hi, john_a)

nx.shortest_path_length(G_karate, mr_hi, john_a)

# Calculate dictionary of all shortest paths
length_source_target = dict(nx.shortest_path_length(G_karate))
length_source_target[0][33]

def path_length_histogram(G, title=None):
    # Find path lengths
    length_source_target = dict(nx.shortest_path_length(G))
    # Convert dict of dicts to flat list
    all_shortest = sum(
        [list(length_target.values()) for length_target in length_source_target.values()],
        [])
    # Calculate integer bins
    high = max(all_shortest)
    bins = [-0.5 + i for i in range(high + 2)]
    # Plot histogram
    plt.hist(all_shortest, bins=bins, rwidth=0.8)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    
plt.figure(figsize=(7.5, 2.5))
plt.subplot(1, 3, 1)
path_length_histogram(G_karate, title="Karate")
plt.subplot(1, 3, 2)
path_length_histogram(G_electric, title="Electric")
plt.subplot(1, 3, 3)
path_length_histogram(G_internet, title="Internet")
plt.tight_layout()

nx.average_shortest_path_length(G_karate)

nx.average_shortest_path_length(G_electric)

nx.average_shortest_path_length(G_internet)

nx.diameter(G_karate)

nx.diameter(G_electric)

nx.diameter(G_internet)

nx.density(G_karate)

nx.density(G_electric)

nx.density(G_internet)

import networkx.algorithms.connectivity as nxcon

nxcon.minimum_st_node_cut(G_karate, mr_hi, john_a)

nxcon.minimum_st_edge_cut(G_karate, mr_hi, john_a)

nx.node_connectivity(G_karate, mr_hi, john_a)

nx.edge_connectivity(G_karate, mr_hi, john_a)

nxcon.minimum_node_cut(G_karate)

nxcon.minimum_edge_cut(G_karate)

nx.node_connectivity(G_karate)

nx.node_connectivity(G_electric)

nx.node_connectivity(G_internet)

nx.average_node_connectivity(G_karate)

nx.average_node_connectivity(G_electric)

nx.average_node_connectivity(G_internet)

# Function to plot a single histogram
def centrality_histogram(x, title=None):
    plt.hist(x, density=True)
    plt.title(title)
    plt.xlabel("Centrality")
    plt.ylabel("Density")

# Create a figure
plt.figure(figsize=(7.5, 2.5))
# Calculate centralities for each example and plot
plt.subplot(1, 3, 1)
centrality_histogram(
    nx.eigenvector_centrality(G_karate).values(), title="Karate")
plt.subplot(1, 3, 2)
centrality_histogram(
    nx.eigenvector_centrality(G_electric, max_iter=1000).values(), title="Electric")
plt.subplot(1, 3, 3)
centrality_histogram(
    nx.eigenvector_centrality(G_internet).values(), title="Internet")

# Adjust the layout
plt.tight_layout()

import math
def entropy(x):
    # Normalize
    total = sum(x)
    x = [xi / total for xi in x]
    H = sum([-xi * math.log2(xi) for xi in x])
    return H
    
entropy(nx.eigenvector_centrality(G_karate).values())

entropy(nx.eigenvector_centrality(G_electric, max_iter=1000).values())

entropy(nx.eigenvector_centrality(G_internet).values())

def gini(x):
    x = [xi for xi in x]
    n = len(x)
    gini_num = sum([sum([abs(x_i - x_j) for x_j in x]) for x_i in x])
    gini_den = 2.0 * n * sum(x)
    return gini_num / gini_den
    
gini(nx.eigenvector_centrality(G_karate).values())

gini(nx.eigenvector_centrality(G_electric, max_iter=1000).values())

gini(nx.eigenvector_centrality(G_internet).values())

nx
