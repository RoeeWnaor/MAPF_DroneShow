# MAPF Drone Show - Star of David (20 Agents)

This project demonstrates the core data pipeline for a complex Multi-Agent Path Finding (MAPF) problem, focusing on comparing algorithm efficiency versus collision safety.

---

## 🎯 Project Goal
The primary objective is to select the 20 most optimal starting positions out of 25 available drones and compute a collision-free trajectory for the agents to form a **clear, wide Star of David shape (Magen David)** at a high altitude.

## 🔬 Experimental Findings (Summary)

We tested three MAPF planning strategies to determine the most practical solution for the 20-drone layout:

| Algorithm | Method | Execution Time | Makespan | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Simple A\*** | Decoupled (No Safety) | 0.003s | 16.0s | **Ideal Baseline** (Collision detected: INFEASIBLE) |
| **PRP (MFEA Priority)**| Heuristic (Dynamic Priority) | 0.007s | **16.0s** | **Practical & Optimal.** Achieved collision-free path in minimum theoretical time. |
| **CBS** | Optimal (Hierarchical Search) | Failed / Timeout | N/A | **Non-Scalable.** Computationally too expensive for real-time use. |

## 📁 Repository Files

| File Name | Description | Status |
| :--- | :--- | :--- |
| `data_processor.py` | **Assignment Logic:** Implements the **Hungarian Algorithm** to filter 25 starting drones down to the 20 most efficient for the mission. | Final Data Generator |
| `a_star_searcher.py` | **Search Engine:** The robust A* function used by all planning algorithms. | Engine Core |
| `prp_mfea_planner.py` | **Winning Algorithm:** Implements the successful Priority-Based Planner using a dynamic MFEA-like heuristic. | Successful Planner |
| `simple_astar_planner.py` | **Baseline Test:** Decoupled A* used to establish the 16.0s minimum time cost. | Unsafe Baseline |
| `prp_mfea_trajectory.csv` | **Final Output:** The safe, optimal (X, Y, Z, Time) trajectory data for the 20 drones. | **Visualization Input** |
| `PRP_MFEA_SAFE_animation.gif` | **Final Product:** Visual demonstration of the collision-free Star of David formation. | **Project Proof** |

