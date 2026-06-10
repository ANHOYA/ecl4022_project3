# Project 3: Reinforcement Learning in Custom Maze (SimpleEnv)

This project implements and compares various Reinforcement Learning (RL) algorithms in a custom `MiniGrid` maze environment (`SimpleEnv`, 6x6 internal navigable space) for Project 3.

---

## 1. Project Overview & Environment
- **Environment**: Custom 8x8 MiniGrid layout (6x6 navigable tiles surrounded by external walls).
- **Features**: Includes multiple crossroads, dead ends to deceive the agent, and obstacles placed directly in front of the final destination (Goal) at position (6,6).
- **Implemented Algorithms**:
  1. **Tabular RL**: Q-Learning (Off-policy TD) and SARSA (On-policy TD) (each trained for 50,000 episodes).
  2. **Deep RL**: Grayscale (1-channel) and RGB (3-channel) input Deep Q-Networks (DQN).
  3. **Early Stopping DQN**: A DQN variant that halts training once the moving average reward over 100 episodes reaches 0.84 or above and Epsilon is below 0.05.
  4. **Hyperparameter Tuning DQN**: DQN variants evaluating Batch Size = 256 and Epsilon Decay = 0.95.

---

## 2. Policy Evaluation and Execution (`Project3_evaluation.ipynb`)

For grading and testing without waiting for hours of training, we provide a dedicated evaluation-only Jupyter notebook ([Project3_evaluation.ipynb](file:///home/ash/projects/51_optimization_project3/Project3_evaluation.ipynb)). It loads pre-trained policy arrays and neural network weights directly.

### 🚀 Steps to Run
1. Open the project directory in VS Code or Jupyter.
2. **Select Jupyter Kernel**:
   - In the top-right corner of the notebook, choose **Select Kernel**.
   - Select the Python interpreter inside the local virtual environment folder **`.venv`** in the project root.
   - All necessary libraries (such as `minigrid`, `torch`, `imageio`, `tqdm`) are pre-installed in this virtual environment.
3. **Run All Cells**:
   - Run the cells sequentially.
   - It will automatically load the stored tabular policies and network weights from `policy_obs3/` and run evaluations for all agents.
   - The final shortest path navigation of each agent will be displayed as HTML videos directly inside the notebook.

---

## 3. Directory Structure
- `Project3_evaluation.ipynb`: The evaluation-only notebook for immediate policy execution (used for submission).
- `Project3_final_obs3.ipynb`: The primary training notebook containing model designs, training loops, tensorboard logs, and comparison plots.
- `policy_obs3/`: Pre-trained policy weights directory.
  - `learned_Q_q_learning.npy` / `learned_Q_sarsa.npy`: Stored tabular Q-tables.
  - `*.pth`: Saved weights for DQN and its variants.
- `video_obs3/`: Generated gameplay/navigation MP4 videos during evaluation.
- `.venv/`: The local virtual environment directory containing the pre-configured Python interpreter.
