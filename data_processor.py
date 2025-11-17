import pandas as pd
import numpy as np

# --- 1. CONFIGURATION (20 DRONES FOR STAR OF DAVID) ---
NUM_DRONES = 20
DRONE_SPACING_START = 1.5
DRONE_SPACING_END = 5.0
Z_START = 10
Z_END = 170  # CRITICAL FIX: Increased Z_END from 150 to 170 to provide vertical margin
OUTPUT_FILE_CLEANED = 'cleaned_drone.csv'
OUTPUT_FILE_INITIAL = 'drone_initial_status.csv'


# --- 2. GENERATE START POSITIONS (P_start: 4x5 Grid) ---
def generate_start_grid():
    rows = 4
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


# --- 3. GENERATE TARGET POSITIONS (P_end: Star of David Shape) ---
def generate_star_of_david():
    points = []
    d = DRONE_SPACING_END

    # Define the shape (Note: the vertical positioning is now inherently higher)
    outer_coords = [
        # Upward triangle
        (0, d * 3.5), (d * 3, d * 3.5), (d * 1.5, d * 5.5),

        # Downward triangle - CRITICAL: These coordinates were too low
        (0, d * 1.5), (-d * 3, d * 1.5), (-d * 1.5, -d * 0.5),
    ]

    center_x = d * 0.75
    center_y = d * 2.5
    center_coords = [
        (center_x, center_y), (-center_x, center_y),
        (center_x * 2, center_y), (-center_x * 2, center_y),
        (0, d * 3.0), (0, d * 2.0),
        (d * 1.5, d * 1.5), (-d * 1.5, d * 1.5)
    ]

    all_coords = outer_coords[:10] + center_coords[:10]

    if len(all_coords) < NUM_DRONES:
        all_coords.extend([(0, 0)] * (NUM_DRONES - len(all_coords)))

    # Apply Z and center X
    max_x = max(c[0] for c in all_coords)
    min_x = min(c[0] for c in all_coords)
    center_offset = (max_x + min_x) / 2

    for i, (x, y) in enumerate(all_coords):
        points.append({
            'Drone_ID': i + 1,
            'X_end': round(x - center_offset, 2),
            'Y_end': 0,
            'Z_end': round(y + Z_END - 150, 2)  # Center Z around the new Z_END (170m)
        })

    return pd.DataFrame(points)


# --- 4. MAIN EXECUTION (Direct Assignment - NO HUNGARIAN) ---
if __name__ == '__main__':
    df_start = generate_start_grid()
    df_end = generate_star_of_david()

    df_cleaned = df_start.merge(df_end, on='Drone_ID')

    df_start.to_csv(OUTPUT_FILE_INITIAL, index=False)
    df_cleaned.to_csv(OUTPUT_FILE_CLEANED, index=False)

    print("=" * 70)
    print("✅ DATA PIPELINE RESET: 20 Drones for Star of David.")
    print(f"Output saved to {OUTPUT_FILE_CLEANED}")
    print(f"Start/End pairs created: {len(df_cleaned)}")