# dqn-pytorch

## How to run

Note that we used the same code structure for both linear network and DQN. The only difference lies in the class named as DQNetworkFC. Comment out the other one first to run linear net or DQN.

Cartpole:
```
python main.py --env=CartPole-v0 --max_timesteps=1000000 --memory=32 --warmup_mem=31 --batch_size=32 --gamma=0.99 --hard_update_freq=1 --save_dir=./data_cartpole_linear

python main.py --env=CartPole-v0 --max_timesteps=1000000 --memory=50000 --batch_size=32 --gamma=0.99 --hard_update_freq=1 --save_dir=./data_cartpole_linear_er

python main.py --env=CartPole-v0 --max_timesteps=1000000 --memory=50000 --batch_size=32 --gamma=0.99 --hard_update_freq=2000 --save_dir=./data_cartpole_dqn
```

Mountain Car:
```python

python main.py --env=MountainCar-v0 --memory=32 --warmup_mem=31 --batch_size=32 --gamma=1 --max_episode=5000 --hard_update_freq=1 --max_epsilon_decay_steps=500000 --save_dir=./mountaincar_linear

python main.py --env=MountainCar-v0 --memory=50000 --batch_size=32 --gamma=1 --max_episode=5000 --hard_update_freq=1 --max_epsilon_decay_steps=500000 --save_dir=./mountaincar_linear_er

python main.py --env=MountainCar-v0 --memory=50000 --batch_size=32 --gamma=1 --max_episode=5000 --hard_update_freq=2000 --max_epsilon_decay_steps=500000 --save_dir=./mountaincar_dqn
```

SpaceInvader:
```python
python main.py --save_dir=./dqn --frame_skip=3 --memory=1000000 --hard_update_freq=10000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.1 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1

python main.py --double_q --save_dir=./double_dqn --frame_skip=3 --memory=1000000 --hard_update_freq=30000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.01 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1

python main.py --dueling_net --double_q --max_grad_norm=10 --save_dir=./dueling_double_dqn --frame_skip=3 --memory=1000000 --hard_update_freq=30000 --gamma=0.99 --train_freq=4 --lr=0.00025 --initial_epsilon=1.0 --final_epsilon=0.01 --max_epsilon_decay_steps=1000000 --warmup_mem=50000 --max_timesteps=100000000 --save_interval=500 --render=1
```
