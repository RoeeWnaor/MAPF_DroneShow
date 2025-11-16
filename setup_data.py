import pandas as pd
import numpy as np

# --- Configuration Parameters ---
DRONE_DISTANCE_START = 1.5  # Initial drone spacing in meters
Z_START = 10  # Takeoff altitude (meters)
ROWS = 10  # Number of rows in the starting grid
COLS = 12  # Number of columns in the starting grid
FILE_NAME = 'drone_initial_status.csv'  # Output file name

# --- 1. Generate Initial Start Position Table (120 Drones) ---

start_positions = []

# Calculate offset to center the grid around X=0 and Y=0
x_offset = (COLS - 1) * DRONE_DISTANCE_START / 2
y_offset = (ROWS - 1) * DRONE_DISTANCE_START / 2

for i in range(ROWS):
    for j in range(COLS):
        # Calculate a unique ID for each drone (1 to 120)
        drone_id = i * COLS + j + 1

        # Calculate X and Y coordinates based on the grid position (i, j) and offset
        x_start = j * DRONE_DISTANCE_START - x_offset
        y_start = i * DRONE_DISTANCE_START - y_offset
        z_start = Z_START

        start_positions.append({
            'Drone_ID': drone_id,
            'X_start': round(x_start, 2),
            'Y_start': round(y_start, 2),
            'Z_start': z_start
        })

df_start_120 = pd.DataFrame(start_positions)
df_start_120.to_csv(FILE_NAME, index=False)

print("=" * 70)
print(f"✅ Created Initial Status File: {FILE_NAME}")
print(df_start_120.head(3))
print("...")