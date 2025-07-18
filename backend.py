import ast
import uuid
import networkx as nx
import matplotlib.pyplot as plt


def adjacency_to_graph(adjacency_list: str) -> nx.Graph:
    """
    Convert the streamlit input into a proper graph for rendering
    """
    G = nx.Graph()
    # WARN: avodi eval() for security
    adjacency_list = ast.literal_eval(adjacency_list)
    G.add_edges_from(adjacency_list)
    return G


def draw_graph(G: nx.Graph, outfile: str = "graph.png"):
    """
    Draw the graph using matplotlib. Claude helped me with this.
    """
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=1000, linewidths=1, edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.1")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def create_schedule(
    adjacency_list: str, data_transfers: str, min_latency: bool = True
) -> dict[int, dict[tuple[str, str], str]]:
    """
    Creates a schedule to either minimise latecny or maximise throughput.

    Returns a dictionary where each key is a timestep with values being the
    edges (links) active at that timestep. Each edge is a dict mapping an edge
    (tuple of nodes) to a transfer id.
    """
    adjacency = ast.literal_eval(adjacency_list)
    transfers = ast.literal_eval(data_transfers)

    # Get the data transfer paths based on the objective
    if min_latency:
        paths = [shortest_path(adjacency, t) for t in transfers]
    else:
        paths = uncontested_paths(adjacency, transfers)

    # Allocate paths across times
    ids = [str(uuid.uuid4()) for _ in paths]
    schedule = {}

    for i, path in enumerate(paths):
        id = ids[i]
        timestep = 0

        # Schedule each edge in the path, proagating one timestep at a time
        for j in range(1, len(path)):
            edge = (path[j - 1], path[j])
            if timestep not in schedule:
                schedule[timestep] = {}

            # Pipeline operation to next timestep if edge in use at present
            while edge in schedule[timestep]:
                timestep += 1
            schedule[timestep][edge] = id

    return schedule


def shortest_path(
    adjacency_list: list[tuple[str, str]], transfer: tuple[str, str]
) -> list[str]:
    """
    Find shortest path from node A to node B as a list of nodes
    """
    raise NotImplementedError(
        "This function should implement a shortest path algorithm, e.g., Dijkstra's or A*."
    )


def uncontested_paths(
    adjacency_list: list[tuple[str, str]], data_transfers: list[tuple[str, str]]
) -> list[list[str]]:
    """
    Find non-overlaping paths for all data transfers
    """

    paths = []
    for transfer in data_transfers:
        path = shortest_path(adjacency_list, transfer)

        # If path found, remove its edges from adjacency list to avoid conflicts
        if path:
            paths.append(path)

            path_edges = {}
            for i in range(1, len(path)):
                path_edges[str((path[i - 1], path[i]))] = True
                path_edges[str((path[i], path[i - 1]))] = True

            adjacency_list = [
                edge for edge in adjacency_list if str(edge) not in path_edges
            ]

    return paths
