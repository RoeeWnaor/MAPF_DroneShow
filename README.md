# MAPF Drone Show - Star of David (20 Agents)

This project demonstrates the core data pipeline for a complex Multi-Agent Path Finding (MAPF) problem, focusing on comparing algorithm efficiency versus collision safety.

---

## 🎯 Project Goal
The primary objective is to select the 20 most optimal starting positions out of 25 available drones and compute a collision-free trajectory for the agents to form a **clear, wide Star of David shape (Magen David)** at a high altitude.

## 🔬 Experimental Findings (Summary)

We tested four MAPF planning strategies to determine the most practical solution for the 20-drone layout:

| Algorithm | Method | Execution Time | Makespan | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Simple A\*** | Decoupled (No Safety) | 0.003s | 16.0s | **Ideal Baseline** (Collision detected: INFEASIBLE) |
| **Simple PRP (Fixed Priority)** | Heuristic (Fixed Priority) | 40s+ | N/A | **Incomplete/Deadlock.** Failed to find a path for Agent 5.0, proving fixed priority is unreliable. |
| **PRP (MFEA Priority)** | Heuristic (Dynamic Priority) | 0.007s | **16.0s** | **Practical & Optimal.** Achieved collision-free path in minimum theoretical time. |
| **CBS** | Optimal (Hierarchical Search) | Failed / Timeout | N/A | **Non-Scalable.** Computationally too expensive for real-time use. |

## 📁 Repository Files

This project contains all the planners, data processors, and visualization scripts used throughout the research:

| File Name | Description | Status |
| :--- | :--- | :--- |
| `data_processor.py` | **Assignment & Geometry Logic:** Implements the **Hungarian Algorithm** to filter 25 starting drones down to the 20 most efficient; generates Star of David coordinates. | Final Data Generator |
| `a_star_searcher.py` | **Search Engine:** The robust A* function used by all planning algorithms. | Engine Core |
| `cbs_planner.py` | **Optimal Planner Test:** Contains the CBS logic that was tested and failed on the 100-drone scale (used for analysis). | Scalability Test |
| `prp_planner.py` | **Incomplete Planner Test:** Contains the Simple PRP logic that failed due to fixed-priority deadlock (used for analysis). | Deadlock Test |
| `simple_astar_planner.py` | **Baseline Planner:** Decoupled A* used to establish the 16.0s minimum time cost. | Unsafe Baseline |
| `prp_mfea_planner.py` | **Winning Algorithm:** Implements the successful Priority-Based Planner using a dynamic MFEA-like heuristic. | Successful Planner |
| `visualize_matplotlib.py` | **Visualization Script:** Contains the logic to generate the final, time-synchronized GIF animations, including the collision effect and the jump to final position. | Project Proof |
| `prp_mfea_trajectory.csv` | **Final Output:** The safe, optimal (X, Y, Z, Time) trajectory data for the 20 drones. | Visualization Input |
| `simple_astar_trajectory.csv` | **Baseline Trajectory Data:** Contains the short, fastest trajectory that will collide (used for collision GIF). | Collision Input |
| `cleaned_drone.csv` | **Final Assignment:** List of the 20 selected drones with their final assigned start and end coordinates. | Final Data |
| `drone_initial_status.csv` | **Initial Status:** The list of the 25 possible starting drones. | Data Input |
| `README.md` | This file: Project description, experimental summary, and file structure. | Documentation |