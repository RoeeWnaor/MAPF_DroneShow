import numpy as np
from heapq import heappop, heappush

# Definition of possible movements (6-directions: x, -x, y, -y, z, -z)
MOVEMENTS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]


def heuristic(current, goal):
    """Calculates the Manhattan distance (H-cost) between current cell and goal cell."""
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1]) + abs(current[2] - goal[2])


def a_star_search(start_node, goal_node, dims, occupied_cells=None, path_constraint=None, max_time_steps=10000):
    """
    Implements the A* search algorithm (Robust and Simplified).
    """
    if occupied_cells is None:
        occupied_cells = set()
    if path_constraint is None:
        path_constraint = set()

    open_list = [(heuristic(start_node, goal_node), 0, start_node)]

    came_from = {start_node: None}
    g_score = {start_node: 0}

    while open_list:
        f_cost, current_g, current_node = heappop(open_list)

        # Optimization: Skip if a shorter path to this node was already found
        if current_g > g_score.get(current_node, float('inf')):
            continue

        # 1. Check for goal achievement
        if current_node == goal_node:
            # Reconstruct and return the path
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = came_from.get(node)

            path.reverse()
            return path

        # Explore neighbors
        for move in MOVEMENTS:
            neighbor_node = (
                current_node[0] + move[0],
                current_node[1] + move[1],
                current_node[2] + move[2]
            )

            new_g = current_g + 1  # Each move costs 1

            # 2. Check bounds and constraints
            if not (0 <= neighbor_node[0] < dims[0] and
                    0 <= neighbor_node[1] < dims[1] and
                    0 <= neighbor_node[2] < dims[2]):
                continue

            # 3. Check for better path
            if new_g < g_score.get(neighbor_node, float('inf')):
                g_score[neighbor_node] = new_g
                came_from[neighbor_node] = current_node

                h_cost = heuristic(neighbor_node, goal_node)
                f_cost = new_g + h_cost

                heappush(open_list, (f_cost, new_g, neighbor_node))

    return None  # No path found