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

# Create empty affiliation network and list of people
B = nx.Graph()
people = set()
# Load data file into network
from pathlib import Path
data_dir = Path('.') / 'data'
with open(data_dir / 'crossley2012' / '50_ALL_2M.csv') as f:
    # Parse header
    events = next(f).strip().split(",")[1:]
    # Parse rows
    for row in f:
        parts = row.strip().split(",")
        person = parts[0]
        people.add(person)
        for j, value in enumerate(parts[1:]):
            if value != "0":
                B.add_edge(person, events[j], weight=int(value))
# Project into person-person co-affilation network
from networkx import bipartite
G = bipartite.projected_graph(B, people)

betweenness = nx.betweenness_centrality(G, normalized=False)
sorted(betweenness.items(), key=lambda x:x[1], reverse=True)[0:10]

eigenvector = nx.eigenvector_centrality(G)
sorted(eigenvector.items(), key=lambda x:x[1], reverse=True)[0:10]

closeness = nx.closeness_centrality(G)
sorted(closeness.items(), key=lambda x:x[1], reverse=True)[0:10]

triangles = nx.triangles(G)
sorted(triangles.items(), key=lambda x:x[1], reverse=True)[0:10]

clustering = nx.clustering(G)
[(x, clustering[x]) for x in sorted(people, key=lambda x:eigenvector[x], reverse=True)[0:10]]

