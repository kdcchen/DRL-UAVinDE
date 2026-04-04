This README explains how to install, run, and reproduce the entire workflow.

## Installation

1. **Create Conda environment**
   ```bash
   conda create -n uav python=3.10
   conda activate uav
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training the PPO Agent

Run:
```bash
python code/train/train.py
```

After training, the model will be saved to:
`models/ppo_uav_final.zip`

You can change the save path in `config.py`.

## Static Evaluation (Trajectory Plot)

To visualize a single episode trajectory:
```bash
python code/train/eval_animation.py
```

This script plots:
* UAV path
* Obstacles
* Goal position
