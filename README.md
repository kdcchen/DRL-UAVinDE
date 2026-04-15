# DRL-UAVinDE: Deep Reinforcement Learning for UAV in Dynamic Environments

This project implements a Deep Reinforcement Learning (DRL) system for Unmanned Aerial Vehicle (UAV) navigation and obstacle avoidance in dynamic environments. It utilizes Proximal Policy Optimization (PPO) from Stable-Baselines3 and features a highly structured Curriculum Learning approach to train the UAV progressively.

## Installation

1. Create Conda environment
   ```bash
   conda create -n uav python=3.10
   conda activate uav
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All environment parameters, PPO hyperparameters, and Curriculum Learning stages are centralized in:
`code/config/config.py`

You can customize map size, wind variations (with AR1 smoothing), obstacle layouts (random, algorithmic blocking, etc.), and training steps for each stage here.

## Training the Agent (Curriculum Learning)

The primary way to train the UAV is through the curriculum script, which guides the agent from simple empty maps to complex dynamic environments with wind.

1. Full Curriculum Training
To train the agent through all stages defined in `config.py` sequentially:
```bash
python code/train/curriculum.py
```

2. Resume or Start from a Specific Stage
You can start training from a specific stage and load a pre-trained model as the baseline. 
For example, to start from Stage 2 using the model trained in Stage 1:
```bash
python code/train/curriculum.py --start_stage 2 --load_model ./models/stage1_empty.zip
```
Note: Models are automatically saved in the `./models/` directory with their stage names.

## Evaluation & Visualization

1. Dynamic Interactive Animation
To visualize the UAV's behavior in real-time with an interactive Matplotlib GUI (includes a "Restart" button). You can specify which stage's environment and model you want to test:
```bash
python code/train/eval_animation.py --stage 7
```
(This will automatically load the configuration for Stage 7 and search for a model starting with `stage7_` in the `./models/` folder).

2. Static Evaluation & Trajectory Analysis
To run the model without animation, save the trajectory, and analyze the flight path (distance, steps, success rate):
```bash
python code/train/eval.py
python code/train/analyze_traj.py
```

## Key Features

* Custom Gymnasium Environment: 2D continuous physics simulation with acceleration control.
* Curriculum Learning: 7 distinct stages (from basic navigation, to static obstacle avoidance, to dynamic obstacles with wind turbulence).
* Algorithmic Obstacle Generation: Dynamic topological generation of obstacles to force specific learning behaviors (e.g., detour routing).
* Advanced Reward Shaping: Artificial Potential Fields (4th-power smooth penalty) for obstacles and progressive speed rewards for racing behavior.
