import ast
import math
import uuid
import heapq
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt


class VisualisationUtils:
    def __init__(self) -> None:
        self._seed = 1

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
        pos = nx.spring_layout(G, seed=self._seed)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="lightblue",
            node_size=1000,
            linewidths=1,
            edgecolors="black",
        )
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, width=2)
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()

    def schedule_to_gif(
        self,
        G: nx.Graph,
        schedule: dict[int, dict[tuple[str, str], str]],
        ids: dict[tuple[str, str], str],
        outfile: str = "schedule.gif",
    ):
        """
        Convert the schedule to a GIF for visualisation.
        Create individual frames for each timestep and combine with PIL.
        Claude helped me write this function.
        """
        # Create random colours for each UUID
        colours = {id: f"#{id[:6]}" for id in ids.values()}

        # Compute layout once
        pos = nx.spring_layout(G, seed=self._seed)

        # Map transfer id to (from, to) nodes for legend
        transfer_map = {id: edge for edge, id in ids.items()}

        # Prepare frames
        images = []
        for timestep in sorted(schedule.keys()):
            plt.figure(figsize=(6, 4))
            # Draw nodes and labels
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color="lightblue",
                node_size=1000,
                linewidths=1,
                edgecolors="black",
            )
            nx.draw_networkx_labels(G, pos)

            # Prepare edge colours: default black, highlight scheduled edges
            edge_colours = []
            for edge in G.edges():
                # Edge may be in either direction
                edge_key = (edge[0], edge[1])
                edge_key_rev = (edge[1], edge[0])
                if edge_key in schedule[timestep]:
                    edge_colours.append(colours[schedule[timestep][edge_key]])
                elif edge_key_rev in schedule[timestep]:
                    edge_colours.append(colours[schedule[timestep][edge_key_rev]])
                else:
                    edge_colours.append("black")

            nx.draw_networkx_edges(G, pos, edge_color=edge_colours, width=2)

            # Add cycle counter to top left
            plt.text(
                0.01,
                0.98,
                f"Cycle {timestep}",
                transform=plt.gca().transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="black",
            )

            # Add legend for active transfers in this cycle
            active_ids = set(schedule[timestep].values())
            legend_y = 0.91
            for id in sorted(active_ids):
                colour = colours[id]
                edge = transfer_map[id]
                label = f"{edge}"
                # Draw coloured square
                plt.gca().add_patch(
                    plt.Rectangle(
                        (0.01, legend_y),
                        0.025,
                        0.025,
                        transform=plt.gca().transAxes,
                        color=colour,
                        clip_on=False,
                    )
                )
                # Draw label next to square
                plt.text(
                    0.04,
                    legend_y + 0.012,
                    label,
                    transform=plt.gca().transAxes,
                    fontsize=5,
                    va="center",
                    ha="left",
                    color="black",
                )
                legend_y -= 0.04  # Move down for next legend entry

            # Save frame
            frame_path = f"{outfile}_{timestep}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches="tight")
            plt.close()
            images.append(Image.open(frame_path))

        # Save GIF
        images[0].save(
            f"{outfile}.gif",
            save_all=True,
            append_images=images[1:],
            duration=2000,
            loop=0,
        )


class SchedulingUtils:
    def __init__(self) -> None:
        self._adjacency = None
        self._transfers = None
        self._graph = None

    def create_graph(
        self, adjacency_list: list[tuple[str, str]]
    ) -> dict[str, set[str]]:
        """
        Create hashtable graph from adjacency list (simplifies Dijkstra)

        Args:
            adjacency_list: List of tuples representing edges in the graph

        Returns:
            Dict where keys are node labesl and values are adjacent nodes
        """
        graph = {}
        for edge in adjacency_list:
            if edge[0] not in graph:
                graph[edge[0]] = set()
            if edge[1] not in graph:
                graph[edge[1]] = set()
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])
        return graph

    def shortest_path(
        self,
        graph: dict[str, set[str]],
        transfer: tuple[str, str],
        costs: dict[tuple[str, str], int] = {},
    ) -> list[str]:
        """
        Find shortest path from node A to node B as a list of nodes

        Args:
            graph: see self.create_graph()
            transfer: Tuple (from_node, to_node) representing transfer to schedule
            costs: Optional cost per edge. If not set, each edge has cost 1.

        Returns:
            List of nodes in the shortest path from A to B, or empty list if no
            possible path exists.
        """
        # Validation
        from_node, to_node = transfer
        if from_node not in graph or to_node not in graph:
            return []

        # Setup
        pq = []
        # pq format: (numEdges, path)
        heapq.heappush(pq, (0, [from_node]))
        visited = set()

        # Dijkstra's algorithm
        while pq:
            cost, path = heapq.heappop(pq)
            curr = path[-1]

            if curr == to_node:
                return path
            if curr not in visited:
                visited.add(curr)
                for adjacent in graph[curr]:
                    if adjacent not in visited:
                        d_cost = costs.get((curr, adjacent), 1)
                        heapq.heappush(pq, (cost + d_cost, path + [adjacent]))

        return []

    def uncontested_paths(
        self,
        graph: dict[str, set[str]],
        data_transfers: list[tuple[str, str]],
        MAX_RETRIES: int = 3,
    ) -> list[list[str]]:
        """
        Find non-overlapping paths for as many data transfers as possible.

        Args:
            graph: see self.create_graph()
            data_transfers: List of tuples holding data transfers to schedule
            MAX_RETRIES: max iterations to increase path cost to reduce contenttion
        """
        paths = []
        costs = {}

        for R in range(1, MAX_RETRIES + 1):
            # Initial run to find paths
            paths = [self.shortest_path(graph, t, costs) for t in data_transfers]

            # Update costs to penalise edges in use
            costs = {}
            longest_path = max([len(p) for p in paths])

            # Check for conflicted paths in the same timestep
            for i in range(1, longest_path):
                counts = {}
                for p in paths:
                    try:
                        edge = (p[i], p[i - 1])
                        edge_rev = (p[i - 1], p[i])
                        counts[edge] = counts.get(edge, 0) + R
                        counts[edge_rev] = counts.get(edge_rev, 0) + R
                    except IndexError:
                        continue

                for k, v in counts.items():
                    if v > R:
                        costs[k] = v
        return paths

    def create_schedule(
        self, adjacency_list: str, data_transfers: str, min_latency: bool = True
    ) -> tuple[dict[int, dict[tuple[str, str], str]], dict[tuple[str, str], str]]:
        """
        Creates a schedule to either minimise latecny or maximise throughput.

        Returns:
        (1) dictionary where each key is a timestep with values being the
        edges (links) active at that timestep. Each edge is a dict mapping an edge
        (tuple of nodes) to a transfer id.
        (2) dictionary mapping transfer ids to (from, to) nodes for legend
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
        ids = {}
        schedule = {}

        for path in paths:
            ids[(path[0], path[-1])] = str(uuid.uuid4())
            timestep = 0

            # Schedule each edge in the path, proagating one timestep at a time
            for j in range(1, len(path)):
                edge = (path[j - 1], path[j])

                # Pipeline operation to next timestep if edge in use at present
                while edge in schedule.get(timestep, {}):
                    timestep += 1
                if timestep not in schedule:
                    schedule[timestep] = {}

                schedule[timestep][edge] = ids[(path[0], path[-1])]
                timestep += 1

        return schedule, ids
