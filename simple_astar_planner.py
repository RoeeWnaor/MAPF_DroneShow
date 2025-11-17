import pandas as pd
import numpy as np
from collections import deque
import time
from a_star_searcher import a_star_search

# --- 1. GLOBAL CONFIGURATION ---
INPUT_FILE = 'cleaned_drone.csv'
OUTPUT_FILE = 'simple_astar_trajectory.csv'
DT = 0.5
MAX_STEPS_ALLOWED = 10000  # Passed to A* for safety

# --- 2. GRID CONFIGURATION ---
GRID_RESOLUTION = 3
DRONE_SAFETY_RADIUS = 1

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

# --- CRITICAL FIX 1: Ensure boundaries are expanded safely ---
# We use a larger margin and the highest possible Z
MARGIN = GRID_RESOLUTION * 4
X_min -= MARGIN
X_max += MARGIN
Y_min -= MARGIN
Y_max += MARGIN
Z_min = 0
Z_max += MARGIN

# Calculate Grid Dimensions (number of cells)
# We ensure the calculated size is safely larger than the required space.
GRID_WIDTH = int(np.ceil((X_max - X_min) / GRID_RESOLUTION))
GRID_DEPTH = int(np.ceil((Y_max - Y_min) / GRID_RESOLUTION))
GRID_HEIGHT = int(np.ceil((Z_max - Z_min) / GRID_RESOLUTION))
GRID_DIMS = (GRID_WIDTH, GRID_DEPTH, GRID_HEIGHT)


# --- CORE CONVERSION FUNCTIONS (Shared Logic) ---
def continuous_to_grid(coords):
    """
    Converts continuous (meter) coordinates to discrete grid indices (integers).
    CRITICAL FIX 2: Added small safety epsilon (1e-6) to prevent rounding errors at the boundary.
    """
    epsilon = 1e-6
    grid_x = int(np.floor((coords[0] - X_min + epsilon) / GRID_RESOLUTION))
    grid_y = int(np.floor((coords[1] - Y_min + epsilon) / GRID_RESOLUTION))
    grid_z = int(np.floor((coords[2] - Z_min + epsilon) / GRID_RESOLUTION))
    return (grid_x, grid_y, grid_z)


def grid_to_continuous(grid_coords):
    """Converts discrete grid indices back to continuous meter coordinates (cell center)."""
    x = grid_coords[0] * GRID_RESOLUTION + X_min
    y = grid_coords[1] * GRID_RESOLUTION + Y_min
    z = grid_coords[2] * GRID_RESOLUTION + Z_min
    return (x, y, z)


def setup_mapf_tasks():
    """Defines the MAPF tasks (start/end points) in grid coordinates."""

    mapf_tasks = []
    for i in range(len(df_cleaned)):
        start_grid = continuous_to_grid(P_start_continuous[i])
        end_grid = continuous_to_grid(P_end_continuous[i])

        # Validate that the calculated grid points are within the bounds
        if not (0 <= start_grid[0] < GRID_DIMS[0] and
                0 <= start_grid[1] < GRID_DIMS[1] and
                0 <= start_grid[2] < GRID_DIMS[2] and
                0 <= end_grid[0] < GRID_DIMS[0] and
                0 <= end_grid[1] < GRID_DIMS[1] and
                0 <= end_grid[2] < GRID_DIMS[2]):
            # This should NEVER happen now
            print(f"FATAL ERROR: Start/End point of Drone {df_cleaned.iloc[i]['Drone_ID']} is outside grid bounds!")
            print(f"Grid Size: {GRID_DIMS}. Failed Start: {start_grid}. Failed End: {end_grid}")
            exit()

        mapf_tasks.append({
            'drone_id': df_cleaned.iloc[i]['Drone_ID'],
            'start': start_grid,
            'end': end_grid
        })

    print(f"Loaded {len(mapf_tasks)} MAPF tasks.")
    print(f"Grid Dimensions (WxDxH): {GRID_DIMS}")

    return mapf_tasks


# Define the MAPF tasks for immediate use
MAPF_TASKS = setup_mapf_tasks()


# --- 3. ALGORITHMS IMPLEMENTATION (Simple A*) ---

def run_simple_astar_planner(tasks, dims):
    """
    Implements the Simple A* Planner (Decoupled).
    Finds the shortest path for each agent individually, IGNORING all collisions.
    """
    start_time = time.time()

    paths = {}

    print(f"Running Simple A* Planner (Decoupled) for {len(tasks)} drones...")

    for task in tasks:

        # Run A* search without any constraints
        path = a_star_search(
            start_node=task['start'],
            goal_node=task['end'],
            dims=dims,
            occupied_cells=set(),  # No dynamic obstacles
            path_constraint=set(),  # No constraints
            max_time_steps=MAX_STEPS_ALLOWED  # Use the increased limit
        )

        if path is None:
            # If path still fails, it means the search horizon was too small OR the nodes were invalid.
            print(f"Simple A* FAILED to find a path for Drone {task['drone_id']}!")
            return None, 0

        paths[task['drone_id']] = path

    end_time = time.time()
    running_time = end_time - start_time

    print("Simple A* execution successful.")
    return paths, running_time


# --- 4. TRAJECTORY GENERATION AND MEASUREMENTS ---

def generate_final_trajectory(paths, dt):
    """Converts the discrete grid paths into a final time-stamped continuous trajectory."""

    final_data = []

    # --- MEASUREMENTS ---
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

                # Linear Interpolation
                current_x = start_coords[0] + (t_segment / segment_duration) * (end_coords[0] - start_coords[0])
                current_y = start_coords[1] + (t_segment / segment_duration) * (end_coords[1] - start_coords[1])
                current_z = start_coords[2] + (t_segment / segment_duration) * (end_coords[2] - start_coords[2])

                final_data.append({
                    'Drone_ID': drone_id, 'Time': round(t_global, 1),
                    'X': round(current_x, 2), 'Y': round(current_y, 2), 'Z': round(current_z, 2),
                    'Algorithm': 'Simple_A_Star'
                })

        # Ensure the final point is added at the final Makespan time
        if len(path_grid) * 1.0 < actual_makespan:
            final_coord = grid_to_continuous(path_grid[-1])
            stay_start_time = len(path_grid)

            for t_extra in np.arange(stay_start_time, actual_makespan + dt, dt):
                final_data.append({
                    'Drone_ID': drone_id, 'Time': round(t_extra, 1),
                    'X': round(final_coord[0], 2), 'Y': round(final_coord[1], 2), 'Z': round(final_coord[2], 2),
                    'Algorithm': 'Simple_A_Star'
                })

    print(f"\n--- Simple A* Measurements ---")
    print(f"Calculated Max Velocity: {round(max_velocity, 2)} m/s")
    print(f"Actual Makespan (Time to finish): {round(actual_makespan, 1)} seconds")

    return pd.DataFrame(final_data)


# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Trajectory Planning (Simple A*) ---")

    paths_dict, run_time = run_simple_astar_planner(MAPF_TASKS, GRID_DIMS)

    if paths_dict is None:
        print("Trajectory planning failed. Cannot proceed.")
    else:
        final_trajectory_df = generate_final_trajectory(paths_dict, DT)
        final_trajectory_df.to_csv(OUTPUT_FILE, index=False)

        print(f"\n✅ SIMPLE A* RESULTS:")
        print(f"   Execution Time: {round(run_time, 4)} seconds")
        print(f"   Trajectory data saved to {OUTPUT_FILE}")