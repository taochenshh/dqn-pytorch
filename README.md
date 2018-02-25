# dqn-pytorch

## How to run
Note: These are just toy hyperparameters that have not been well tuned.
Mountain Car:
```
python main.py --env=MountainCar-v0 --memory=50000 --batch_size=32 --gamma=1 --max_timesteps=1000000 --hard_update_freq=10000 --max_epsilon_decay_steps=500000
```

Cartpole:
```python
python main.py --env=CartPole-v0 --max_episode=6000 --max_episode_steps=200 --memory=50000 --batch_size=32 --gamma=0.99
```