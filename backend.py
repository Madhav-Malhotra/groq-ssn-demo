import ast
import uuid
import networkx as nx
import matplotlib.pyplot as plt


class VisualisationUtils:
    def adjacency_to_graph(self, adjacency_list: str) -> nx.Graph:
        """
        Convert the streamlit input into a proper graph for rendering
        """
        G = nx.Graph()
        # WARN: avodi eval() for security
        adjacency_list = ast.literal_eval(adjacency_list)
        G.add_edges_from(adjacency_list)
        return G

    def draw_graph(self, G: nx.Graph, outfile: str = "graph.png"):
        """
        Draw the graph using matplotlib. Claude helped me with this.
        """
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="lightblue",
            node_size=1000,
            linewidths=1,
            edgecolors="black",
        )
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.1")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()


class SchedulingUtils:
    def __init__(self) -> None:
        self._adjacency = None
        self._transfers = None
        self._graph = None

    def create_graph(
        self, adjacency_list: list[tuple[str, str]]
    ) -> dict[str, list[str]]:
        """
        Create hashtable graph from adjacency list (simplifies Dijkstra)
        """
        graph = {}
        for edge in adjacency_list:
            if edge[0] not in graph:
                graph[edge[0]] = []
            if edge[1] not in graph:
                graph[edge[1]] = []
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        return graph

    def shortest_path(
        self, graph: dict[str, list[str]], transfer: tuple[str, str]
    ) -> list[str]:
        """
        Find shortest path from node A to node B as a list of nodes
        """
        raise NotImplementedError(
            "This method should be implemented using Dijkstra's algorithm or similar."
        )

    def uncontested_paths(
        self,
        graph: dict[str, list[str]],
        data_transfers: list[tuple[str, str]],
    ) -> list[list[str]]:
        """
        Find non-overlaping paths for all data transfers
        """
        # Deep clone graph for modifications
        graph_local = graph.copy()

        paths = []
        for transfer in data_transfers:
            path = self.shortest_path(graph_local, transfer)

            # If path found, remove its edges from graph to avoid conflicts
            if path:
                paths.append(path)

                for i in range(1, len(path)):
                    try:
                        graph_local[path[i - 1]].remove(path[i])
                        graph_local[path[i]].remove(path[i - 1])
                    # Could be that path doesn't exist ing raph
                    except Exception as e:
                        pass

        return paths

    def create_schedule(
        self, adjacency_list: str, data_transfers: str, min_latency: bool = True
    ) -> dict[int, dict[tuple[str, str], str]]:
        """
        Creates a schedule to either minimise latecny or maximise throughput.

        Returns a dictionary where each key is a timestep with values being the
        edges (links) active at that timestep. Each edge is a dict mapping an edge
        (tuple of nodes) to a transfer id.
        """
        self._adjacency = ast.literal_eval(adjacency_list)
        self._transfers = ast.literal_eval(data_transfers)
        self._graph = self.create_graph(self._adjacency)

        # Get the data transfer paths based on the objective
        if min_latency:
            paths = [self.shortest_path(self._graph, t) for t in self._transfers]
        else:
            paths = self.uncontested_paths(self._graph, self._transfers)

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
