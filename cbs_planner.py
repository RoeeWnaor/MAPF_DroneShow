import pandas as pd
import numpy as np
import time
import heapq  # Required for the CBS priority queue
from a_star_searcher import a_star_search  # Import A* search engine

# --- 1. GLOBAL CONFIGURATION (Shared Logic) ---
INPUT_FILE = 'cleaned_drone.csv'
OUTPUT_FILE = 'cbs_trajectory.csv'
FLIGHT_TIME_TARGET = 30
DT = 0.5

# --- 2. GRID CONFIGURATION ---
GRID_RESOLUTION = 3
DRONE_SAFETY_RADIUS = 1
MAX_STEPS_ALLOWED = 10000

# --- DATA LOADING ---
try:
    df_cleaned = pd.read_csv(INPUT_FILE)
    P_start_continuous = df_cleaned[['X_start', 'Y_start', 'Z_start']].values
    P_end_continuous = df_cleaned[['X_end', 'Y_end', 'Z_end']].values
except FileNotFoundError:
    print(f"ERROR: Input file '{INPUT_FILE}' not found. Please ensure data_processor.py was run.")
    exit()

# --- GRID DIMENSION CALCULATION (Shared Logic) ---
X_min, X_max = P_start_continuous[:, 0].min(), P_end_continuous[:, 0].max()
Y_min, Y_max = P_start_continuous[:, 1].min(), P_end_continuous[:, 1].max()
Z_min, Z_max = P_start_continuous[:, 2].min(), P_end_continuous[:, 2].max()

MARGIN = GRID_RESOLUTION * 4
X_min -= MARGIN
X_max += MARGIN
Y_min -= MARGIN
Y_max += MARGIN
Z_min = 0
Z_max += MARGIN

GRID_WIDTH = int(np.ceil((X_max - X_min) / GRID_RESOLUTION))
GRID_DEPTH = int(np.ceil((Y_max - Y_min) / GRID_RESOLUTION))
GRID_HEIGHT = int(np.ceil((Z_max - Z_min) / GRID_RESOLUTION))
GRID_DIMS = (GRID_WIDTH, GRID_DEPTH, GRID_HEIGHT)


# --- CORE CONVERSION FUNCTIONS (Shared Logic) ---
def continuous_to_grid(coords):
    epsilon = 1e-6
    grid_x = int(np.floor((coords[0] - X_min + epsilon) / GRID_RESOLUTION))
    grid_y = int(np.floor((coords[1] - Y_min + epsilon) / GRID_RESOLUTION))
    grid_z = int(np.floor((coords[2] - Z_min + epsilon) / GRID_RESOLUTION))
    return (grid_x, grid_y, grid_z)


def grid_to_continuous(grid_coords):
    x = grid_coords[0] * GRID_RESOLUTION + X_min
    y = grid_coords[1] * GRID_RESOLUTION + Y_min
    z = grid_coords[2] * GRID_RESOLUTION + Z_min
    return (x, y, z)


def setup_mapf_tasks():
    mapf_tasks = []
    for i in range(len(df_cleaned)):
        start_grid = continuous_to_grid(P_start_continuous[i])
        end_grid = continuous_to_grid(P_end_continuous[i])

        mapf_tasks.append({
            'drone_id': df_cleaned.iloc[i]['Drone_ID'],
            'start': start_grid,
            'end': end_grid
        })

    print(f"Loaded {len(mapf_tasks)} MAPF tasks.")
    print(f"Grid Dimensions (WxDxH): {GRID_DIMS}")

    return mapf_tasks


MAPF_TASKS = setup_mapf_tasks()


# --- 3. ALGORITHMS IMPLEMENTATION (CBS) ---

def find_first_conflict(current_paths):
    schedule = {}
    max_len = max(len(path) for path in current_paths.values())

    for t in range(max_len):
        for drone_id, path in current_paths.items():
            if t < len(path):
                node = path[t]
                key = (node[0], node[1], node[2], t)
                if key in schedule:
                    return (drone_id, schedule[key], *node, t)
                schedule[key] = drone_id
    return None


def run_cbs_planner(tasks, dims):
    """
    Implements the Conflict-Based Search (CBS) algorithm.
    """
    start_time = time.time()

    # CRITICAL FIX 1: Tie-breaker Index
    tie_breaker = 0

    # 1. Initialize the root node (High-Level Search)
    root_paths = {}

    for task in tasks:
        # Pass max_time_steps to A*
        path = a_star_search(
            start_node=task['start'],
            goal_node=task['end'],
            dims=dims,
            occupied_cells=set(),
            path_constraint=set(),
            max_time_steps=MAX_STEPS_ALLOWED
        )
        if path is None:
            print(f"CBS FAILED: Agent {task['drone_id']} has no initial path.")
            return None, 0

        root_paths[task['drone_id']] = path

    initial_cost = sum(len(path) for path in root_paths.values())

    # CRITICAL FIX 2: Added tie_breaker index (0) to the tuple
    open_list = [(initial_cost, tie_breaker, root_paths, {})]

    solution_paths = None

    print("Running Conflict-Based Search (CBS)...")

    # 2. Main High-Level Search Loop
    while open_list:
        # CRITICAL FIX 3: Unpack the tie-breaker. The second element is ignored (_)
        current_cost, _, current_paths, current_constraints = heapq.heappop(open_list)

        conflict = find_first_conflict(current_paths)

        if conflict is None:
            solution_paths = current_paths
            break

        drone_id1, drone_id2, x, y, z, t = conflict

        # Branch 1 & 2: Constrain each agent involved in the conflict
        for agent_id in [drone_id1, drone_id2]:
            new_constraints = current_constraints.copy()

            if agent_id not in new_constraints:
                new_constraints[agent_id] = set()

            new_constraints[agent_id].add((x, y, z, t))

            constrained_paths = current_paths.copy()

            task = next(t for t in tasks if t['drone_id'] == agent_id)

            # Run A* with the new constraint (CRITICAL FIX: Pass max_time_steps)
            new_path = a_star_search(
                start_node=task['start'],
                goal_node=task['end'],
                dims=dims,
                occupied_cells=set(),
                path_constraint=new_constraints[agent_id],
                max_time_steps=MAX_STEPS_ALLOWED
            )

            if new_path is not None:
                constrained_paths[agent_id] = new_path
                new_cost = sum(len(path) for path in constrained_paths.values())

                # CRITICAL FIX 4: Increment and use the tie-breaker
                tie_breaker += 1
                heapq.heappush(open_list, (new_cost, tie_breaker, constrained_paths, new_constraints))

    end_time = time.time()
    running_time = end_time - start_time

    if solution_paths:
        print("CBS execution successful.")
    else:
        print("CBS FAILED to find a collision-free path.")

    return solution_paths, running_time


# --- 4. TRAJECTORY GENERATION AND MEASUREMENTS (Shared Logic) ---

def generate_final_trajectory(paths, dt):
    """Converts the discrete grid paths into a final time-stamped continuous trajectory."""

    final_data = []
    max_velocity = 0
    max_path_length = max(len(path) for path in paths.values())
    actual_makespan = max_path_length * 1.0

    for drone_id, path_grid in paths.items():

        for t_step in range(len(path_grid) - 1):
            start_coords = grid_to_continuous(path_grid[t_step])
            end_coords = grid_to_continuous(path_grid[t_step + 1])

            segment_duration = 1.0
            distance = np.linalg.norm(np.array(end_coords) - np.array(start_coords))
            velocity = distance / segment_duration
            max_velocity = max(max_velocity, velocity)
            num_samples = int(segment_duration / dt)

            for i in range(num_samples):
                t_segment = i * dt
                t_global = t_step + t_segment

                current_x = start_coords[0] + (t_segment / segment_duration) * (end_coords[0] - start_coords[0])
                current_y = start_coords[1] + (t_segment / segment_duration) * (end_coords[1] - start_coords[1])
                current_z = start_coords[2] + (t_segment / segment_duration) * (end_coords[2] - start_coords[2])

                final_data.append({
                    'Drone_ID': drone_id, 'Time': round(t_global, 1),
                    'X': round(current_x, 2), 'Y': round(current_y, 2), 'Z': round(current_z, 2),
                    'Algorithm': 'CBS'
                })

        if len(path_grid) * 1.0 < actual_makespan:
            final_coord = grid_to_continuous(path_grid[-1])
            stay_start_time = len(path_grid)

            for t_extra in np.arange(stay_start_time, actual_makespan + dt, dt):
                final_data.append({
                    'Drone_ID': drone_id, 'Time': round(t_extra, 1),
                    'X': round(final_coord[0], 2), 'Y': round(final_coord[1], 2), 'Z': round(final_coord[2], 2),
                    'Algorithm': 'CBS'
                })

    print(f"\n--- CBS Measurements ---")
    print(f"Calculated Max Velocity: {round(max_velocity, 2)} m/s")
    print(f"Actual Makespan (Time to finish): {round(actual_makespan, 1)} seconds")

    return pd.DataFrame(final_data)


# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Trajectory Planning (CBS) ---")

    paths_dict, run_time = run_cbs_planner(MAPF_TASKS, GRID_DIMS)

    if paths_dict is None:
        print("Trajectory planning failed. Cannot proceed.")
    else:
        final_trajectory_df = generate_final_trajectory(paths_dict, DT)
        final_trajectory_df.to_csv(OUTPUT_FILE, index=False)

        print(f"\n✅ CBS RESULTS:")
        print(f"   Execution Time: {round(run_time, 4)} seconds")
        print(f"   Trajectory data saved to {OUTPUT_FILE}")