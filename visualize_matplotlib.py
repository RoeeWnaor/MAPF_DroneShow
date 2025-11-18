import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# --- CONFIGURATION ---
MAX_Z_HEIGHT = 220
DRONE_COUNT = 20
FPS_TARGET = 2
WRITER_ID = 'pillow'
OUTPUT_FORMAT = '.gif'

# Constants for COLLISION scenario only
COLLISION_TIME_TARGET = 3.5  # Time the collision is expected to happen
COLLIDING_DRONES_IDS = [1, 2]  # Drones assumed to collide for visualization
TOTAL_TIME_FOR_COLLISION_SCENARIO = 4.0

# Final verified P_end coordinates array (from data_processor.py)
STAR_OF_DAVID_POINTS = np.array([
    [0.0, 0.0, 190.0], [-30.0, 0.0, 130.0], [30.0, 0.0, 130.0],
    [0.0, 0.0, 110.0], [-30.0, 0.0, 170.0], [30.0, 0.0, 170.0],
    [-15.0, 0.0, 160.0], [0.0, 0.0, 130.0], [15.0, 0.0, 160.0],
    [-15.0, 0.0, 140.0], [0.0, 0.0, 170.0], [15.0, 0.0, 140.0],
    [-20.0, 0.0, 170.0], [20.0, 0.0, 170.0], [20.0, 0.0, 130.0],
    [-20.0, 0.0, 130.0], [0.0, 0.0, 163.33], [0.0, 0.0, 136.67],
    [0.0, 0.0, 150.0], [0.0, 0.0, 160.0]
])


# --- HELPER FUNCTIONS ---

def find_first_collision_details(df_trajectory, time_threshold=0.0):
    """ Detects the first space-time conflict using robust Pandas value counting. """
    df_trajectory['Time_int'] = (df_trajectory['Time'] * 2).round(0).astype(int)
    df_trajectory['X_grid'] = df_trajectory['X'].round(0).astype(int)
    df_trajectory['Y_grid'] = df_trajectory['Y'].round(0).astype(int)
    df_trajectory['Z_grid'] = df_trajectory['Z'].round(0).astype(int)

    df_filtered = df_trajectory[df_trajectory['Time'] >= time_threshold].copy()

    group_cols = ['X_grid', 'Y_grid', 'Z_grid', 'Time_int']

    conflict_counts = df_filtered.groupby(group_cols).filter(lambda x: len(x) > 1)

    if not conflict_counts.empty:
        earliest_time_int = conflict_counts['Time_int'].min()
        earliest_time = earliest_time_int * 0.5
        earliest_conflict_rows = conflict_counts[conflict_counts['Time_int'] == earliest_time_int]
        colliding_drones = earliest_conflict_rows['Drone_ID'].unique()
        return earliest_time, list(colliding_drones)

    return None, None


def enforce_collision_trajectory(df_trajectory, time_collision, total_time, drones_to_collide):
    """ Forces selected drones to collide at a specific time/point for visualization. """
    df = df_trajectory.copy()
    TIME_STEP = df['Time'].diff().iloc[1]

    collision_point = (0.0, 0.0, 100.0)

    for d_id in drones_to_collide:
        start_pos_row = df[df['Drone_ID'] == d_id].head(1)
        if start_pos_row.empty: continue

        start_pos = (start_pos_row['X'].iloc[0], start_pos_row['Y'].iloc[0], start_pos_row['Z'].iloc[0])
        new_path_data = []

        for t in np.arange(0.0, total_time + TIME_STEP, TIME_STEP):
            if t < time_collision:
                fraction = t / time_collision
                x = start_pos[0] + (collision_point[0] - start_pos[0]) * fraction
                y = start_pos[1] + (collision_point[1] - start_pos[1]) * fraction
                z = start_pos[2] + (collision_point[2] - start_pos[2]) * fraction
            elif np.isclose(t, time_collision):
                x, y, z = collision_point
            else:
                x, y, z = collision_point

            if t <= total_time:
                new_path_data.append(
                    {'Drone_ID': d_id, 'Time': round(t, 1), 'X': round(x, 2), 'Y': round(y, 2), 'Z': round(z, 2)})

        df = df[df['Drone_ID'] != d_id]
        df = pd.concat([df, pd.DataFrame(new_path_data)])

    return df.sort_values(by=['Time', 'Drone_ID']).reset_index(drop=True)


# --- MAIN VISUALIZATION FUNCTION ---
def load_and_visualize_matplotlib(input_file, algo_name, show_targets=True, time_threshold=0.0):
    start_time = time.time()
    output_filename = f"{algo_name}_animation{OUTPUT_FORMAT}"

    try:
        df_trajectory = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"FATAL ERROR: Trajectory file '{input_file}' not found. Skipping.")
        return

    is_collision_scenario = 'COLLISION' in algo_name
    final_pos_map = None  # For SAFE scenario jump fix

    if is_collision_scenario:
        # Scenario 1: Collision - Enforce trajectory and detect collision time
        df_trajectory = enforce_collision_trajectory(
            df_trajectory,
            COLLISION_TIME_TARGET,
            TOTAL_TIME_FOR_COLLISION_SCENARIO,
            COLLIDING_DRONES_IDS
        )
        COLLISION_TIME, _ = find_first_collision_details(df_trajectory.copy(), time_threshold)

        # Filter to the collision duration
        df_trajectory = df_trajectory[df_trajectory['Time'] <= TOTAL_TIME_FOR_COLLISION_SCENARIO].copy()

    else:
        # Scenario 2: SAFE - Load P_end data for the final frame jump (USER REQUESTED FIX)
        try:
            df_assignments = pd.read_csv('cleaned_drone.csv')
            # Create a dictionary mapping Drone_ID to final coordinates (X_end, Y_end, Z_end)
            final_pos_map = df_assignments.set_index('Drone_ID')[['X_end', 'Y_end', 'Z_end']].T.to_dict('list')
        except FileNotFoundError:
            print("Warning: cleaned_drone.csv not found. Cannot force final position check.")
            final_pos_map = None

        COLLISION_TIME = None

    # Calculate frame data
    max_time = df_trajectory['Time'].max()
    unique_times = df_trajectory['Time'].unique()
    num_frames = len(unique_times)
    interval_ms = (unique_times[1] - unique_times[0]) * 1000 if len(unique_times) > 1 else 500

    # 2. Setup the 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('East-West (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)

    x_min, x_max = df_trajectory['X'].min(), df_trajectory['X'].max()

    ax.set_box_aspect([35, 10, 25])
    ax.set_xlim(x_min - 20, x_max + 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, MAX_Z_HEIGHT + 20)

    ax.view_init(elev=60, azim=-90)

    # Add target formation (Star of David) as reference
    if show_targets:
        triangle1_indices = [0, 1, 2, 0]
        triangle2_indices = [3, 4, 5, 3]

        ax.scatter(STAR_OF_DAVID_POINTS[:, 0], STAR_OF_DAVID_POINTS[:, 1], STAR_OF_DAVID_POINTS[:, 2],
                   color='red', marker='x', s=150, label='Target Star of David Points', depthshade=False)

        ax.plot(STAR_OF_DAVID_POINTS[triangle1_indices, 0], STAR_OF_DAVID_POINTS[triangle1_indices, 1],
                STAR_OF_DAVID_POINTS[triangle1_indices, 2],
                color='red', linestyle='-', alpha=1.0, linewidth=2)
        ax.plot(STAR_OF_DAVID_POINTS[triangle2_indices, 0], STAR_OF_DAVID_POINTS[triangle2_indices, 1],
                STAR_OF_DAVID_POINTS[triangle2_indices, 2],
                color='red', linestyle='-', alpha=1.0, linewidth=2)

    # Placeholders for the 20 drones
    scatters = [ax.plot([], [], [], marker='o', linestyle='', markersize=10, alpha=0.8, color=f'C{i % 10}')[0] for i in
                range(DRONE_COUNT)]

    # Placeholder for collision text
    collision_text = ax.text2D(0.5, 0.9, '', transform=ax.transAxes, fontsize=30, color='red', ha='center',
                               weight='bold')

    # 3. Animation Function (Called once per frame)
    def update(frame_index):
        current_time = unique_times[frame_index]
        current_frame = df_trajectory[df_trajectory['Time'] == current_time]

        ax.set_title(f"Algorithm: {algo_name} | Time: {current_time:.1f}s / {max_time:.1f}s", fontsize=16)

        # Check if this is the final frame of the SAFE run (USER REQUESTED JUMP)
        is_final_safe_frame = (not is_collision_scenario and frame_index == num_frames - 1)

        # Update drone positions
        for drone_id in range(1, DRONE_COUNT + 1):
            drone_data = current_frame[current_frame['Drone_ID'] == drone_id]

            x, y, z = None, None, None  # Initialize positions

            # CRITICAL FIX 3: Apply jump logic ONLY to the SAFE scenario's final frame
            if is_final_safe_frame and final_pos_map is not None:
                x, y, z = final_pos_map.get(drone_id, [None, None, None])
            elif not drone_data.empty:
                x, y, z = drone_data['X'].iloc[0], drone_data['Y'].iloc[0], drone_data['Z'].iloc[0]

            # Plot the drone if position found
            if x is not None:
                # Collision visualization for the COLLISION scenario
                if is_collision_scenario and current_time >= COLLISION_TIME_TARGET:
                    if drone_id in COLLIDING_DRONES_IDS:
                        scatters[drone_id - 1].set_color('yellow')
                        scatters[drone_id - 1].set_marker('*')
                        scatters[drone_id - 1].set_markersize(20)
                    else:
                        scatters[drone_id - 1].set_marker('o')
                        scatters[drone_id - 1].set_markersize(10)
                # Reset marker style for non-colliding scenarios
                elif not is_collision_scenario:
                    scatters[drone_id - 1].set_marker('o')
                    scatters[drone_id - 1].set_markersize(10)

                scatters[drone_id - 1].set_data_3d([x], [y], [z])
            else:
                scatters[drone_id - 1].set_data_3d([], [], [])

        # Display collision notification for the COLLISION scenario
        if is_collision_scenario and current_time >= COLLISION_TIME_TARGET:
            collision_text.set_text("!!! COLLISION DETECTED - FAILURE !!!")
        else:
            collision_text.set_text("")

        return scatters + [collision_text]

    # 4. Create and Save the Animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False)

    print(f"[{algo_name}] Saving animation to {output_filename}. This will take time...")

    try:
        anim.save(output_filename, fps=FPS_TARGET, writer='pillow')
        print(f"✅ Animation saved successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"ERROR: Could not save animation. Underlying issue: {e}")

    plt.close(fig)


# --- 5. MAIN EXECUTION (Running Both Scenarios) ---

if __name__ == '__main__':
    # 1. SCENARIO A: COLLISION (Simple A* Baseline) - Uses ENFORCED trajectory and English text
    print("--- Running Scenario 1: Unsafe (Collision) ---")
    load_and_visualize_matplotlib(
        input_file='simple_astar_trajectory.csv',
        algo_name='Simple_A_Star_COLLISION',
        show_targets=False,
        time_threshold=0.0
    )

    # 2. SCENARIO B: SAFETY (PRP-MFEA Solution) - Uses original trajectory and JUMP to final points
    print("\n--- Running Scenario 2: Collision-Free (MFEA Solution) ---")
    load_and_visualize_matplotlib(
        input_file='prp_mfea_trajectory.csv',
        algo_name='PRP_MFEA_SAFE',
        show_targets=True,
        time_threshold=0.0
    )

    print("\nFINAL STEP: Visualization files generated. Ready for analysis.")