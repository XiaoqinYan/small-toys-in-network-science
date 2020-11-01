# Configure plotting in Jupyter
from matplotlib import pyplot as plt
# matplotlib inline
plt.rcParams.update({
    'figure.figsize': (7.5, 7.5),
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False})
# Seed random number generator
from numpy import random as nprand
seed = hash("Network Science in Python") % 2**32
nprand.seed(seed)

import networkx as nx

# Import bipartite module
from networkx.algorithms import bipartite
from networkx import NetworkXError
# Load Zachary karate network
G = nx.karate_club_graph()
try:
    # Find and print node sets
    left, right = bipartite.sets(G)
    print("Left nodes\n", left)
    print("\nRight nodes\n", right)
except NetworkXError as e:
    # Not an affiliation network
    print(e)
    
B = nx.Graph()
B.add_edges_from([(v, (v, w)) for v, w in G.edges])
B.add_edges_from([(w, (v, w)) for v, w in G.edges])
try:
    # Find and print node sets
    left, right = bipartite.sets(B)
    print("Left nodes\n", left)
    print("\nRight nodes\n", right)
except NetworkXError as e:
    # Not an affiliation network
    print(e)

bipartite.is_bipartite(B) 

# Create data directory path
from pathlib import Path
data_dir = Path('.') / 'data'
B = nx.Graph()
with open(data_dir / 'bartomeus2008' / 'Bartomeus_Ntw_nceas.txt') as f:
    # Skip header row
    next(f)
    for row in f:
        # Break row into cells
        cells = row.strip().split('\t')
        # Get plant species and pollinator species
        plant = cells[4].replace('_', '\n')
        pollinator = cells[8].replace('_', '\n')
        B.add_edge(pollinator, plant)
        # Set node types
        B.nodes[pollinator]["bipartite"] = 0
        B.nodes[plant]["bipartite"] = 1
# Only consider connected species
B = B.subgraph(list(nx.connected_components(B))[0])

# Get node sets
pollinators = [v for v in B.nodes if B.nodes[v]["bipartite"] == 0]
plants = [v for v in B.nodes if B.nodes[v]["bipartite"] == 1]

# Create figure
plt.figure(figsize=(30,30))
# Calculate layout
pos = nx.spring_layout(B, k=0.9)
# Draw using different shapes and colors for plant/pollinators
nx.draw_networkx_edges(B, pos, width=3, alpha=0.2)
nx.draw_networkx_nodes(B, pos, nodelist=plants, node_color="#bfbf7f", node_shape="h", node_size=3000)
nx.draw_networkx_nodes(B, pos, nodelist=pollinators, node_color="#9f9fff", node_size=3000)
nx.draw_networkx_labels(B, pos)
plt.savefig('output-4.1.png', dpi=150)

# Create co-affiliation network
G = bipartite.projected_graph(B, plants) # Returns the projection of B onto one of its node sets 
# Create figure
plt.figure(figsize=(24,24))
# Calculate layout
pos = nx.spring_layout(G, k=0.5)
# Draw edges, nodes, and labels
nx.draw_networkx_edges(G, pos, width=3, alpha=0.2)
nx.draw_networkx_nodes(G, pos, node_color="#bfbf7f", node_shape="h", node_size=10000)
nx.draw_networkx_labels(G, pos)
plt.savefig('output-4.2.png', dpi=150)

# Create co-affiliation network
G = bipartite.projected_graph(B, pollinators)
# Create figure
plt.figure(figsize=(30,30))
# Calculate layout
pos = nx.spring_layout(G, k=0.5)
# Draw edges, nodes, and labels
nx.draw_networkx_edges(G, pos, width=3, alpha=0.2)
nx.draw_networkx_nodes(G, pos, node_color="#9f9fff", node_size=6000)
nx.draw_networkx_labels(G, pos)
plt.savefig('output-4.3.png', dpi=150)

G = bipartite.weighted_projected_graph(B, plants) 
list(G.edges(data=True))[0]

# Create co-affiliation network
G = bipartite.overlap_weighted_projected_graph(B, pollinators) 
# Get weights
weight = [G.edges[e]['weight'] for e in G.edges]
# Create figure
plt.figure(figsize=(30,30))
# Calculate layout
pos = nx.spring_layout(G, weight='weight', k=0.5)
# Draw edges, nodes, and labels
nx.draw_networkx_edges(G, pos, edge_color=weight, edge_cmap=plt.cm.Blues, width=6, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_color="#9f9fff", node_size=6000)
nx.draw_networkx_labels(G, pos)
plt.savefig('output-4.4.png', dpi=150)

    
