import pandas as pd
import numpy as np

# --- 1. Load the Data ---
try:
    # pd.read_csv: Reads the CSV file and converts it into a powerful Pandas DataFrame (table).
    df_input = pd.read_csv('drone_initial_status.csv')
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("Error: The file drone_initial_status.csv was not found.")
    exit()

print("\n--- A. Raw Data (Before Cleaning) ---")
print(df_input)

# --- 2. Data Cleaning: Handling Missing Values (NaN) ---
# .dropna(subset=...): Removes rows where one of the critical 'start_x' or 'start_y' values is missing (NaN).
# If a drone doesn't have a starting position, it cannot participate.
# inplace=True: Applies the change directly to the original DataFrame.
df_input.dropna(subset=['start_x', 'start_y'], inplace=True)

# --- 3. Data Cleaning: Type Casting ---
# .astype(int): Ensures all coordinates are integers, as required for our movement grid.
df_input[['start_x', 'start_y']] = df_input[['start_x', 'start_y']].astype(int)

# --- 4. Preparing Target Columns ---
# Add empty columns for the target coordinates (which we will fill in the next step).
df_input['target_x'] = np.nan
df_input['target_y'] = np.nan

# --- 5. Summarize Result ---
N_drones = len(df_input)
print("\n--- B. Data After Cleaning and Processing ---")
print(f"Total valid drones remaining (N): {N_drones}")
print(df_input)

# Saving the cleaned data to a new file for safety/traceability
df_input.to_csv('cleaned_drone_data.csv', index=False)