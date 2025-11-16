import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- 1. CONFIGURATION ---
NUM_DRONES_ACTIVE = 100
DRONE_SPACING = 3
Z_END = 150
INPUT_FILE = 'drone_initial_status.csv'
OUTPUT_FILE = 'cleaned_drone.csv'

# Allocation of points to letters/gaps (Total 100)
ALLOCATION = {
    'AYIN': 8,
    'MEM': 7,
    'GAP_1': 5,
    'YOD': 8,
    'SHIN': 12,
    'RESH': 10,
    'ALEPH': 15,
    'LAMED': 10,
    'GAP_2': 5,
    'CHET': 10,
    'YOD_2': 10
}


# --- 2. TARGET GENERATION FUNCTION ---
def generate_letter_points(char_name, num_points, current_x_offset):
    """
    Generates P_end coordinates for a given Hebrew letter.
    """
    points = []

    # Simple block font definitions
    V_HEIGHT = 4 * DRONE_SPACING

    if char_name == 'AYIN':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(DRONE_SPACING, V_HEIGHT / 2 - DRONE_SPACING), (1.5 * DRONE_SPACING, 0),
                       (DRONE_SPACING, -V_HEIGHT / 2 + DRONE_SPACING)])
    elif char_name == 'MEM':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(DRONE_SPACING, V_HEIGHT / 2), (DRONE_SPACING * 2, V_HEIGHT / 2)])
    elif char_name.startswith('YOD'):
        num = num_points
        V_HEIGHT_YOD = (num - 1) * DRONE_SPACING
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT_YOD / 2, V_HEIGHT_YOD / 2, num)])
    elif char_name == 'SHIN':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 4)])
        points.extend([(DRONE_SPACING * 2, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 4)])
        points.extend([(DRONE_SPACING, V_HEIGHT / 2), (DRONE_SPACING * 3, V_HEIGHT / 2)])
        points.extend(
            [(DRONE_SPACING / 2, V_HEIGHT / 2 + DRONE_SPACING), (DRONE_SPACING * 2.5, V_HEIGHT / 2 + DRONE_SPACING)])
    elif char_name == 'RESH':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(x, V_HEIGHT / 2) for x in np.linspace(0, DRONE_SPACING * 3, 5)])
    elif char_name == 'ALEPH':
        points.extend([(DRONE_SPACING, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(x, x / 2) for x in np.linspace(0, DRONE_SPACING * 3, 5)])
        points.extend([(x, -x / 2) for x in np.linspace(0, DRONE_SPACING * 3, 5)])
    elif char_name == 'LAMED':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(x, V_HEIGHT / 2 + DRONE_SPACING) for x in np.linspace(0, DRONE_SPACING * 4, 5)])
    elif char_name == 'CHET':
        points.extend([(0, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
        points.extend([(DRONE_SPACING * 3, z) for z in np.linspace(-V_HEIGHT / 2, V_HEIGHT / 2, 5)])
    elif char_name.startswith('GAP'):
        # Gaps are placed far away but must exist to consume a drone ID
        points.extend([(current_x_offset + i * DRONE_SPACING * 2, Z_END * 2) for i in range(num_points)])

    # Standardize point count
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        points.extend([(points[0][0], points[0][1]) for _ in range(num_points - len(points))])

    final_points = []
    for px, pz in points:
        final_points.append({
            'X_end_raw': px + current_x_offset,
            'Y_end': 0,  # Flat 2D shape
            'Z_end': pz + Z_END,
            'Shape_Part': char_name
        })

    # Calculate step size for next letter
    if final_points:
        max_x = max(p['X_end_raw'] for p in final_points)
        x_step = max_x - current_x_offset
        return final_points, current_x_offset + x_step + 2 * DRONE_SPACING
    else:
        return final_points, current_x_offset + 3 * DRONE_SPACING


def create_target_dataframe():
    all_end_positions = []
    current_x = 0

    for char_name, count in ALLOCATION.items():
        points, next_x = generate_letter_points(char_name, count, current_x)
        all_end_positions.extend(points)
        current_x = next_x  # Update the starting X for the next letter

    df_end = pd.DataFrame(all_end_positions)

    # Center the entire shape around X=0
    total_width = df_end['X_end_raw'].max() - df_end['X_end_raw'].min()
    center_offset = df_end['X_end_raw'].min() + total_width / 2

    df_end['X_end'] = (df_end['X_end_raw'] - center_offset).round(2)
    df_end['Z_end'] = df_end['Z_end'].round(2)

    df_end['Target_ID'] = range(1, NUM_DRONES_ACTIVE + 1)
    return df_end[['Target_ID', 'X_end', 'Y_end', 'Z_end']]


# --- 3. MAIN PROCESSING LOGIC ---
if __name__ == '__main__':
    try:
        # A. Read initial drone status (120 Drones)
        df_start_120 = pd.read_csv(INPUT_FILE)

        # B. Create the target positions (100 Targets)
        df_end_100 = create_target_dataframe()

        # C. Prepare data for assignment
        P_start = df_start_120[['X_start', 'Y_start', 'Z_start']].values
        P_end = df_end_100[['X_end', 'Y_end', 'Z_end']].values

        # D. Create Cost Matrix (120 Start Points x 100 End Points)
        # Cost is the Euclidean distance
        cost_matrix = cdist(P_start, P_end)

        # E. Optimal Assignment (Hungarian Algorithm)
        # Finds the 100 optimal pairings (start drone -> end target)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # F. Create the Cleaned DataFrame
        # The result includes the 100 selected start drones mapped to their assigned targets
        df_cleaned_drone = pd.DataFrame({
            'Drone_ID': df_start_120.iloc[row_ind]['Drone_ID'].values,
            'X_start': df_start_120.iloc[row_ind]['X_start'].values,
            'Y_start': df_start_120.iloc[row_ind]['Y_start'].values,
            'Z_start': df_start_120.iloc[row_ind]['Z_start'].values,
            'X_end': df_end_100.iloc[col_ind]['X_end'].values,
            'Y_end': df_end_100.iloc[col_ind]['Y_end'].values,
            'Z_end': df_end_100.iloc[col_ind]['Z_end'].values,
        })

        # Calculate distance for verification (optional)
        df_cleaned_drone['Distance'] = np.sqrt(
            (df_cleaned_drone['X_end'] - df_cleaned_drone['X_start']) ** 2 +
            (df_cleaned_drone['Y_end'] - df_cleaned_drone['Y_start']) ** 2 +
            (df_cleaned_drone['Z_end'] - df_cleaned_drone['Z_start']) ** 2
        ).round(2)

        # G. Export the final, cleaned file
        df_cleaned_drone = df_cleaned_drone.sort_values(by='Drone_ID').reset_index(drop=True)
        df_cleaned_drone.to_csv(OUTPUT_FILE, index=False)

        print("=" * 70)
        print(f"✅ Data processing complete! File created: {OUTPUT_FILE}")
        print(f"Total Active Drones: {len(df_cleaned_drone)}")
        print(f"Average flight distance: {df_cleaned_drone['Distance'].mean().round(2)} meters.")
        print("\nFirst 5 entries of the cleaned data:")
        print(df_cleaned_drone.head())

    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_FILE}' not found. Make sure you ran the first code block to create it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")