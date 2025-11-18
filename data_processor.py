import pandas as pd
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- 1. CONFIGURATION (25 DRONES START, 20 DRONES END) ---
NUM_DRONES_TOTAL = 25
NUM_DRONES_ACTIVE = 20
DRONE_SPACING_START = 1.5
D_SCALE = 15.0
Z_START = 10
Z_END = 170
OUTPUT_FILE_CLEANED = 'cleaned_drone.csv'
OUTPUT_FILE_INITIAL = 'drone_initial_status.csv'


# --- 2. GENERATE START POSITIONS (P_start: 5x5 Grid = 25 Drones) ---
def generate_start_grid():
    rows = 5
    cols = 5
    start_positions = []

    x_offset = (cols - 1) * DRONE_SPACING_START / 2
    y_offset = (rows - 1) * DRONE_SPACING_START / 2

    for i in range(rows):
        for j in range(cols):
            drone_id = i * cols + j + 1
            x_start = j * DRONE_SPACING_START - x_offset
            y_start = i * DRONE_SPACING_START - y_offset
            start_positions.append({
                'Drone_ID': drone_id,
                'X_start': round(x_start, 2),
                'Y_start': round(y_start, 2),
                'Z_start': Z_START,
            })
    return pd.DataFrame(start_positions)


# --- 3. GENERATE TARGET POSITIONS (P_end: 20 Points for Wide Star of David) ---
def generate_star_of_david():
    points_list = []

    # Define the base parameters for the star shape
    base_z = Z_END - 20
    horizontal_scale = 30.0
    vertical_scale = 40.0

    # Vertices of the upward-pointing triangle (P1, P2, P3)
    p1 = [0.0, 0.0, base_z + vertical_scale]
    p2 = [-horizontal_scale, 0.0, base_z - (vertical_scale / 2)]
    p3 = [horizontal_scale, 0.0, base_z - (vertical_scale / 2)]

    # Vertices of the downward-pointing triangle (P4, P5, P6)
    p4 = [0.0, 0.0, base_z - vertical_scale]
    p5 = [-horizontal_scale, 0.0, base_z + (vertical_scale / 2)]
    p6 = [horizontal_scale, 0.0, base_z + (vertical_scale / 2)]

    # Core 6 vertices
    all_star_coords = [p1, p2, p3, p4, p5, p6]  # Current length: 6

    # Midpoints of outer and inner triangle sides (6 more points)
    all_star_coords.append([(p1[0] + p2[0]) / 2, 0.0, (p1[2] + p2[2]) / 2])  # P1-P2 Mid
    all_star_coords.append([(p2[0] + p3[0]) / 2, 0.0, (p2[2] + p3[2]) / 2])  # P2-P3 Mid
    all_star_coords.append([(p3[0] + p1[0]) / 2, 0.0, (p3[2] + p1[2]) / 2])  # P3-P1 Mid
    all_star_coords.append([(p4[0] + p5[0]) / 2, 0.0, (p4[2] + p5[2]) / 2])  # P4-P5 Mid
    all_star_coords.append([(p5[0] + p6[0]) / 2, 0.0, (p5[2] + p6[2]) / 2])  # P5-P6 Mid
    all_star_coords.append([(p6[0] + p4[0]) / 2, 0.0, (p6[2] + p4[2]) / 2])  # P6-P4 Mid # Current length: 12

    # Inner hexagon / intersection points and center (8 more points)
    h_inner = horizontal_scale * (2 / 3)
    v_inner_top = base_z + (vertical_scale / 2)
    v_inner_bottom = base_z - (vertical_scale / 2)

    all_star_coords.append([-h_inner, 0.0, v_inner_top])  # 13: Top-left inner
    all_star_coords.append([h_inner, 0.0, v_inner_top])  # 14: Top-right inner
    all_star_coords.append([h_inner, 0.0, v_inner_bottom])  # 15: Bottom-right inner
    all_star_coords.append([-h_inner, 0.0, v_inner_bottom])  # 16: Bottom-left inner
    all_star_coords.append([0.0, 0.0, base_z + (vertical_scale / 3)])  # 17: Upper inner center
    all_star_coords.append([0.0, 0.0, base_z - (vertical_scale / 3)])  # 18: Lower inner center
    all_star_coords.append([0.0, 0.0, base_z])  # 19: Center
    all_star_coords.append([0.0, 0.0, base_z + (vertical_scale / 4)])  # 20: Slightly above center

    # Ensure exactly NUM_DRONES_ACTIVE (20) points are selected
    final_coords = all_star_coords[:NUM_DRONES_ACTIVE]

    if len(final_coords) != NUM_DRONES_ACTIVE:
        raise ValueError(f"Error in generation: Expected {NUM_DRONES_ACTIVE} points, got {len(final_coords)}")

    for i, (x, y, z) in enumerate(final_coords):
        points_list.append({
            'Target_ID': i + 1,
            'X_end': round(x, 2),
            'Y_end': round(y, 2),
            'Z_end': round(z, 2)
        })

    return pd.DataFrame(points_list)


# --- 4. MAIN EXECUTION (Hungarian Algorithm Assignment 25 -> 20) ---
if __name__ == '__main__':
    # Generate 25 start positions and 20 end positions
    df_start_25 = generate_start_grid()
    df_end_20 = generate_star_of_david()

    # Save the initial 25 drones status (for documentation)
    df_start_25.to_csv(OUTPUT_FILE_INITIAL, index=False)

    # Prepare data for Cost Matrix
    P_start = df_start_25[['X_start', 'Y_start', 'Z_start']].values
    P_end = df_end_20[['X_end', 'Y_end', 'Z_end']].values

    # CRITICAL STEP: Create the Cost Matrix (25 start points x 20 end points)
    cost_matrix = cdist(P_start, P_end)

    # Find the optimal assignment (this returns the 20 pairings that minimize total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the final cleaned DataFrame (20 rows only)
    df_cleaned = pd.DataFrame({
        'Drone_ID': df_start_25.iloc[row_ind]['Drone_ID'].values,
        'X_start': df_start_25.iloc[row_ind]['X_start'].values,
        'Y_start': df_start_25.iloc[row_ind]['Y_start'].values,
        'Z_start': df_start_25.iloc[row_ind]['Z_start'].values,
        'X_end': df_end_20.iloc[col_ind]['X_end'].values,
        'Y_end': df_end_20.iloc[col_ind]['Y_end'].values,
        'Z_end': df_end_20.iloc[col_ind]['Z_end'].values,
    }).sort_values(by='Drone_ID').reset_index(drop=True)

    # Save final cleaned data file (only 20 optimally selected drones)
    df_cleaned.to_csv(OUTPUT_FILE_CLEANED, index=False)

    print("=" * 70)
    print(f"✅ Hungarian Algorithm SUCCESS: Filtered {NUM_DRONES_TOTAL - len(df_cleaned)} least efficient drones.")
    print(f"Output saved to {OUTPUT_FILE_CLEANED}")
    print(f"Total Drones Starting: {NUM_DRONES_TOTAL}. Active Drones: {len(df_cleaned)}")