# MAPF Drone Show - "Am Yisrael Chai"

This project demonstrates the core data pipeline for a large-scale drone light show, using the Hungarian Algorithm for optimal path assignment (MAPF - Multi-Agent Path Finding).

---

## 🎯 Goal
To select the 100 most optimally positioned drones out of 120 available initial positions to form the Hebrew inscription "עם ישראל חי" (Am Yisrael Chai) at a final altitude of 150 meters.

## 📁 Project Files Breakdown

| File Name | Description | Key Metric |
| :--- | :--- | :--- |
| `setup_data.py` | **Initial State Generator.** Python script that creates the starting grid data for 120 drones. | P_start (120 points) |
| `drone_initial_status.csv` | **Raw Input.** Contains the (X, Y, Z) starting coordinates of the 120 drones in a 10x12 grid. | |
| `data_processor.py` | **Core Processor & Assignment Logic.** Reads the initial data, generates the 100 target points (P_end), and runs the **Hungarian Algorithm** to find the shortest total flight distance. | Optimal Assignment |
| `cleaned_drone.csv` | **Final Output.** Contains the 100 optimally selected drones with their mapped P_start and P_end coordinates, ready for trajectory calculation. | P_start -> P_end (100 pairs) |

---