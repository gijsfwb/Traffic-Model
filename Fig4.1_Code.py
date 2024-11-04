import geopandas as gpd
import networkx as nx
import contextily as ctx
from scipy import stats

#Code for Fig 4.1 (Map)
#DOWNLOAD '/content/110m_cultural.zip' package from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
netherlands = gpd.read_file('/content/110m_cultural.zip')
netherlands = netherlands[netherlands['ADMIN'] == "Netherlands"]

G = nx.DiGraph()
#Coordinates of locations
locations = {
    "Amsterdam": (4.9041, 52.3676),
    "Groningen": (6.5665, 53.2194),
    "Stevinsluizen": (5.042256, 52.934446),  # Northern part of Afsluitdijk
    "Lorentzsluizen": (5.331486, 53.073577), # Southern part of Afsluitdijk
    "Emmeloord": (5.7485, 52.7108),
    "Lelystad": (5.4714, 52.5185),
    "Almere": (5.2236, 52.3508),
    "Utrecht": (5.1214, 52.0907),
    "Den Haag": (4.3007, 52.0705)
}

for place, coord in locations.items():
    G.add_node(place, pos=coord)

edges = [
    ("Groningen", "Utrecht"),
    ("Groningen", "Emmeloord"),
    ("Groningen", "Lorentzsluizen"),
    ("Emmeloord", "Lelystad"),
    ("Lelystad", "Almere"),
    ("Amsterdam", "Utrecht"),
    ("Almere", "Amsterdam"),
    ("Almere", "Utrecht"),
    ("Lorentzsluizen", "Stevinsluizen"),
    ("Utrecht", "Den Haag"),
    ("Amsterdam", "Den Haag"),
    ("Stevinsluizen", "Amsterdam")
]

G.add_edges_from(edges)

special_edges = [("Lorentzsluizen", "Stevinsluizen"), ("Emmeloord", "Lelystad")]
regular_edges = list(set(edges) - set(special_edges))

fig, ax = plt.subplots(figsize=(10, 10))
pos = nx.get_node_attributes(G, 'pos')  # Get node positions

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color='red', alpha=0.6)
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=regular_edges, arrows=True, edge_color='black', alpha=0.5)
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=special_edges, arrows=True, edge_color='orange', alpha=0.8, width=2)

nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='black')

ctx.add_basemap(ax, crs=netherlands.crs.to_string())

plt.title('Network used')
plt.axis('off')
plt.show()