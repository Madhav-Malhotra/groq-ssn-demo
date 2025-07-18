import ast
import networkx as nx
import matplotlib.pyplot as plt


def adjacency_to_graph(adjacency_list: str) -> nx.Graph:
    """
    Convert an adjacency list string to a NetworkX directed graph.
    """
    G = nx.Graph()
    # WARN: avodi eval() for security
    adjacency_list = ast.literal_eval(adjacency_list)
    G.add_edges_from(adjacency_list)
    return G


def draw_graph(G: nx.Graph, outfile: str = "graph.png"):
    """
    Draw the directed graph using matplotlib. Claude helped me with this.
    """
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=1000, linewidths=1, edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.1")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
