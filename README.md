# dqn-pytorch

## How to run
Note: These are just toy hyperparameters that have not been well tuned.
Cartpole:
```
python main.py --env=CartPole-v0 --max_timesteps=1000000 --memory=50000 --batch_size=32 --gamma=0.99 --hard_update_freq=1 --save_dir=./data_cartpole_er

python main.py --env=CartPole-v0 --max_timesteps=1000000 --memory=40 --warmup_mem=35 --batch_size=32 --gamma=0.99 --hard_update_freq=1 --save_dir=./data_cartpole
```

Mountain Car:
```python


python main.py --env=MountainCar-v0 --memory=40 --warmup_mem=35 --batch_size=32 --gamma=1 --max_episode=5000 --hard_update_freq=1 --max_epsilon_decay_steps=500000 --save_dir=./data_mountaincar

python main.py --env=MountainCar-v0 --memory=50000 --batch_size=32 --gamma=1 --max_episode=5000 --hard_update_freq=1 --max_epsilon_decay_steps=500000 --save_dir=./data_mountaincar_er
```

SpaceInvader:
```python
python main.py --save_dir=./lr_0_00025_hard_frame_1 --frame_skip=1 --memory=1000000 --hard_update_freq=10000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.1 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1

python main.py --double_q --save_dir=./lr_0_00025_hard_double_frame_1 --frame_skip=1 --memory=1000000 --hard_update_freq=30000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.01 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1

python main.py --dueling_net --double_q --max_grad_norm=10 --save_dir=./lr_0_00025_hard_dd_frame_1 --frame_skip=1 --memory=1000000 --hard_update_freq=30000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.01 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1

python main.py --dueling_net --double_q --max_grad_norm=10 --save_dir=./lr_0_00025_hard_dd --env=SpaceInvaders-v0 --memory=1000000 --hard_update_freq=30000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.01 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1
```