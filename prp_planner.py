import pandas as pd
import numpy as np
from collections import deque
import time
from a_star_searcher import a_star_search  # Import A* search engine

# --- 1. GLOBAL CONFIGURATION ---
INPUT_FILE = 'cleaned_drone.csv'
OUTPUT_FILE = 'prp_trajectory.csv'
FLIGHT_TIME_TARGET = 30  # Makespan target in seconds (Reference only)
DT = 0.5  # Time step for the final trajectory in seconds

# --- 2. GRID CONFIGURATION ---
GRID_RESOLUTION = 3  # Cell size in meters (X, Y, Z)
DRONE_SAFETY_RADIUS = 1  # Drone safety radius in meters

# --- DATA LOADING ---
try:
    df_cleaned = pd.read_csv(INPUT_FILE)
    P_start_continuous = df_cleaned[['X_start', 'Y_start', 'Z_start']].values
    P_end_continuous = df_cleaned[['X_end', 'Y_end', 'Z_end']].values
except FileNotFoundError:
    print(f"ERROR: Input file '{INPUT_FILE}' not found. Please ensure data_processor.py was run.")
    exit()

# --- GRID DIMENSION CALCULATION ---
X_min, X_max = P_start_continuous[:, 0].min(), P_end_continuous[:, 0].max()
Y_min, Y_max = P_start_continuous[:, 1].min(), P_end_continuous[:, 1].max()
Z_min, Z_max = P_start_continuous[:, 2].min(), P_end_continuous[:, 2].max()

MARGIN = GRID_RESOLUTION * 2
X_min -= MARGIN
X_max += MARGIN
Y_min -= MARGIN
Y_max += MARGIN
Z_min = 0
Z_max += MARGIN

GRID_WIDTH = int(np.ceil((X_max - X_min) / GRID_RESOLUTION)) + 1
GRID_DEPTH = int(np.ceil((Y_max - Y_min) / GRID_RESOLUTION)) + 1
GRID_HEIGHT = int(np.ceil((Z_max - Z_min) / GRID_RESOLUTION)) + 1
GRID_DIMS = (GRID_WIDTH, GRID_DEPTH, GRID_HEIGHT)


# --- CORE CONVERSION FUNCTIONS ---

def continuous_to_grid(coords):
    """Converts continuous (meter) coordinates to discrete grid indices (integers)."""
    grid_x = np.round((coords[0] - X_min) / GRID_RESOLUTION).astype(int)
    grid_y = np.round((coords[1] - Y_min) / GRID_RESOLUTION).astype(int)
    grid_z = np.round((coords[2] - Z_min) / GRID_RESOLUTION).astype(int)
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


# --- 3. ALGORITHMS IMPLEMENTATION (PRP) ---

def run_prp_planner(tasks, dims):
    """
    Implements the Priority-Based Planning (PRP) algorithm.

    1. Sorts drones by ID (priority).
    2. Sequentially runs A* for each drone.
    3. Treats paths of higher-priority drones as dynamic obstacles (occupied_cells)
       for lower-priority drones.
    """
    start_time = time.time()

    # 1. Define Priority Order (Simplest: by Drone ID)
    sorted_tasks = sorted(tasks, key=lambda x: x['drone_id'])

    # schedule holds the space-time coordinates of planned paths (as dynamic obstacles)
    schedule = set()
    paths = {}

    print(f"Running Priority-Based Planner (PRP) for {len(tasks)} drones...")

    for task in sorted_tasks:
        drone_id = task['drone_id']

        # Run A* search, considering the existing schedule as occupied cells
        path = a_star_search(
            start_node=task['start'],
            goal_node=task['end'],
            dims=dims,
            # PRP uses occupied_cells (space-time conflicts)
            occupied_cells=schedule,
            path_constraint=None
        )

        if path is None:
            # If no path is found, the algorithm failed for this drone
            print(f"PRP FAILED to find a path for Drone {drone_id}!")
            return None, 0

        # Add the newly found path to the overall schedule for subsequent drones
        for t, node in enumerate(path):
            # Add the current node and time to the schedule set
            schedule.add((node[0], node[1], node[2], t))

        paths[drone_id] = path

    end_time = time.time()
    running_time = end_time - start_time

    print("PRP execution successful.")
    return paths, running_time


# --- 4. TRAJECTORY GENERATION AND MEASUREMENTS ---

def generate_final_trajectory(paths, dt):
    """Converts the discrete grid paths into a final time-stamped continuous trajectory."""

    final_data = []

    # --- MEASUREMENTS ---
    max_velocity = 0

    # Calculate Makespan based on the longest path length found
    max_path_length = max(len(path) for path in paths.values())
    actual_makespan = max_path_length * 1.0  # 1 step = 1 second

    for drone_id, path_grid in paths.items():

        # Simple linear interpolation between each grid point for smoothness:
        for t_step in range(len(path_grid) - 1):
            start_coords = grid_to_continuous(path_grid[t_step])
            end_coords = grid_to_continuous(path_grid[t_step + 1])

            segment_duration = 1.0

            distance = np.linalg.norm(np.array(end_coords) - np.array(start_coords))
            velocity = distance / segment_duration
            max_velocity = max(max_velocity, velocity)  # Update Vmax

            num_samples = int(segment_duration / dt)

            for i in range(num_samples):
                t_segment = i * dt
                t_global = t_step + t_segment  # Total time elapsed

                # Linear Interpolation within the segment
                current_x = start_coords[0] + (t_segment / segment_duration) * (end_coords[0] - start_coords[0])
                current_y = start_coords[1] + (t_segment / segment_duration) * (end_coords[1] - start_coords[1])
                current_z = start_coords[2] + (t_segment / segment_duration) * (end_coords[2] - start_coords[2])

                final_data.append({
                    'Drone_ID': drone_id,
                    'Time': round(t_global, 1),
                    'X': round(current_x, 2),
                    'Y': round(current_y, 2),
                    'Z': round(current_z, 2),
                    'Algorithm': 'PRP'
                })

        # Ensure the final point is added at the final Makespan time (if path is shorter)
        if len(path_grid) * 1.0 < actual_makespan:
            final_coord = grid_to_continuous(path_grid[-1])
            stay_start_time = len(path_grid)

            for t_extra in np.arange(stay_start_time, actual_makespan + dt, dt):
                final_data.append({
                    'Drone_ID': drone_id,
                    'Time': round(t_extra, 1),
                    'X': round(final_coord[0], 2),
                    'Y': round(final_coord[1], 2),
                    'Z': round(final_coord[2], 2),
                    'Algorithm': 'PRP'
                })

    # Measurement output (printed to console)
    print(f"\n--- PRP Measurements ---")
    print(f"Calculated Max Velocity: {round(max_velocity, 2)} m/s")
    print(f"Actual Makespan (Time to finish): {round(actual_makespan, 1)} seconds")

    return pd.DataFrame(final_data)


# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Trajectory Planning (PRP) ---")

    # 1. Run the core MAPF algorithm (PRP)
    paths_dict, run_time = run_prp_planner(MAPF_TASKS, GRID_DIMS)

    if paths_dict is None:
        print("Trajectory planning failed. Cannot proceed.")
    else:
        # 2. Convert discrete paths to smooth, time-stamped trajectory
        final_trajectory_df = generate_final_trajectory(paths_dict, DT)

        # 3. Save the output
        final_trajectory_df.to_csv(OUTPUT_FILE, index=False)

        print(f"\n✅ PRP RESULTS:")
        print(f"   Execution Time: {round(run_time, 4)} seconds")
        print(f"   Trajectory data saved to {OUTPUT_FILE}")