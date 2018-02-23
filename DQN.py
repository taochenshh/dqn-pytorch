import pyglet
from pyglet.gl import *
import gym
from gym.wrappers import Monitor
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np

from collections import deque
import os, sys, copy, argparse, shutil
from datetime import datetime

def print_red(skk):
    print("\033[91m {}\033[00m" .format(skk))


def print_green(skk):
    print("\033[92m {}\033[00m" .format(skk))


def print_yellow(skk):
    print("\033[93m {}\033[00m" .format(skk))


def print_blue(skk):
    print("\033[94m {}\033[00m" .format(skk))


def print_purple(skk):
    print("\033[95m {}\033[00m" .format(skk))


def print_cyan(skk):
    print("\033[96m {}\033[00m" .format(skk))

class DQNetworkFC(nn.Module):
    def __init__(self, ob_space, act_space):
        super(DQNetworkFC, self).__init__()
        ob_dim = ob_space.shape[0]
        act_dim = act_space.n
        hid_dim = 64
        self.fc1 = nn.Linear(ob_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, act_dim)

    def forward(self, st):
        out = F.selu(self.fc1(st))
        out = self.fc2(out)
        return out

class CyclicBuffer:
    def __init__(self, shape, dtype='float32'):
        self.cur_pos = 0
        self.cur_len = 0
        self.buffer_size = shape[0]
        self.data = np.zeros(shape).astype(dtype)

    def __len__(self):
        return self.cur_len

    def get_batch(self, ids):
        return self.data[ids]

    def append(self, v):
        if self.cur_len < self.buffer_size:
            self.cur_len += 1
        self.data[self.cur_pos] = v
        self.cur_pos = (self.cur_pos + 1) % self.buffer_size


class ReplayMemory:
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity
        self.ob0 = CyclicBuffer(shape=(capacity, ) + observation_shape)
        self.acts = CyclicBuffer(shape=(capacity, 1))
        self.rewards = CyclicBuffer(shape=(capacity, ) + (1, ))
        self.terminals1 = CyclicBuffer(shape=(capacity, ) + (1,))
        self.ob1 = CyclicBuffer(shape=(capacity, ) + observation_shape)

    def sample(self, batch_size):
        batch_inds = np.random.randint(len(self.ob0), size=batch_size)

        obs0_batch = self.ob0.get_batch(batch_inds)
        obs1_batch = self.ob1.get_batch(batch_inds)
        action_batch = self.acts.get_batch(batch_inds)
        reward_batch = self.rewards.get_batch(batch_inds)
        terminal1_batch = self.terminals1.get_batch(batch_inds)

        result = {
            'obs0': obs0_batch,
            'obs1': obs1_batch,
            'rewards': reward_batch,
            'actions': action_batch,
            'terminals1': terminal1_batch,
        }
        return result

    def append(self, ob0, action, reward, ob1, terminal1):
        self.ob0.append(ob0)
        self.acts.append(action)
        self.rewards.append(reward)
        self.ob1.append(ob1)
        self.terminals1.append(terminal1)

    def __len__(self):
        return len(self.ob0)

class DQNAgent:
    def __init__(self, env, qnet, args):
        self.env = env
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.ac_dim = self.ac_space.n
        self.epsilon = self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.max_epsilon_decay_steps = args.max_epsilon_decay_steps
        self.batch_size = args.batch_size
        self.warmup_mem = args.warmup_mem
        self.train_freq = args.train_freq
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.gamma = args.gamma
        self.max_grad_norm = args.max_grad_norm
        self.softupdate = args.soft_update
        self.tau = args.tau
        self.hardupdate_freq = args.hard_update_freq
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.max_episode = args.max_episode
        self.max_timesteps = args.max_timesteps

        self.save_dir = args.save_dir
        self.model_dir = os.path.join(self.save_dir, 'model')
        self.log_dir = os.path.join(self.save_dir, 'log')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.best_eval_reward = -np.inf
        self.q_net = qnet(ob_space=self.ob_space, act_space=self.ac_space)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.998, last_epoch=-1)
        self.target_q_net = qnet(ob_space=self.ob_space, act_space=self.ac_space)
        self.hard_update(self.target_q_net, self.q_net)

        self.memory = ReplayMemory(capacity=int(args.memory),
                                   observation_shape=self.ob_space.shape)

        self.q_net_loss = nn.SmoothL1Loss()
        self.q_net.cuda()
        self.target_q_net.cuda()
        self.q_net_loss.cuda()
        log_dir = os.path.join(self.log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def train(self):

        episode_loss = []
        episode_num = 0
        episode_num_inc = False
        episode_step = 0
        episode_reward = 0.0
        episode_rewards = deque(maxlen=50)
        episode_steps = deque(maxlen=50)

        ob = self.env.reset()
        for t in range(self.max_timesteps):
            self.decay_epsilon(t)
            q_val = self.get_q_values(ob)
            act = self.epsilon_greedy_policy(q_val)
            new_ob, rew, done, _ = self.env.step(act)
            episode_step += 1
            self.memory.append(ob, act, rew, new_ob, done)
            episode_reward += rew
            if done:
                ob = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_steps.append(episode_step)
                episode_loss = []
                episode_num += 1
                episode_num_inc = True
                episode_step = 0
                episode_reward = 0
            else:
                ob = new_ob

            if len(self.memory) > self.warmup_mem:
                if t % self.train_freq == 0:
                    loss = self.train_q_net()
                    episode_loss.append(loss)
                    if self.lr_decay:
                        self.scheduler.step()
                self.update_target_network(t)

                if episode_num % self.log_interval == 0 and episode_num_inc:
                    log_info = {}
                    log_info['steps'] = t
                    log_info['episode'] = episode_num
                    log_info['episode_loss'] = np.mean(episode_loss)
                    log_info['episode_reward'] = np.mean([rew for rew in episode_rewards])
                    log_info['episode_step'] = np.mean([sp for sp in episode_steps])
                    log_info['lr'] = self.optimizer.param_groups[0]['lr']
                    log_info['epsilon'] = self.epsilon
                    self.nice_log(log_info, step=t)

                if episode_num % self.save_interval == 0 and episode_num_inc:
                    log_info = {}
                    eval_reward = self.eval_q_net()
                    log_info['eval/reward'] = eval_reward
                    self.nice_log(log_info, step=t)
                    is_best = False
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        is_best = True
                    self.save_model(is_best=is_best, step=t)
                    self.log_model_weights(t)
                episode_num_inc = False

            if self.max_episode is not None and episode_num >= self.max_episode:
                break
        print_blue('Training done ...')

    def get_q_values(self, ob):
        ob = Variable(torch.from_numpy(ob)).float().cuda().view(1, -1)
        q_val = self.q_net(ob)
        return q_val

    def eval_q_net(self):
        ori_epsilon = self.epsilon
        self.epsilon = 0.05
        rewards = []
        ob = self.env.reset()
        for i in range(20):
            reward = 0
            episode_step = 0
            while True:
                q_val = self.get_q_values(ob)
                act = self.epsilon_greedy_policy(q_val)
                new_ob, rew, done, _ = self.env.step(act)
                episode_step += 1
                reward += rew
                ob = new_ob
                if done:
                    ob = self.env.reset()
                    rewards.append(reward)
                    break
        self.epsilon = ori_epsilon
        return np.mean(rewards)

    def decay_epsilon(self, step):
        frac = min(float(step) / self.max_epsilon_decay_steps, 1.0)
        self.epsilon = self.initial_epsilon + frac * (self.final_epsilon - self.initial_epsilon)

    def update_target_network(self, step):
        if self.softupdate:
            self.soft_update(self.target_q_net, self.q_net, self.tau)
        elif step % self.hardupdate_freq == 0:
            self.hard_update(self.target_q_net, self.q_net)

    def train_q_net(self):
        batch_data = self.memory.sample(batch_size=self.batch_size)
        for key, value in batch_data.items():
            batch_data[key] = torch.from_numpy(value)
        obs0 = Variable(batch_data['obs0']).float().cuda()
        vol_obs1 = Variable(batch_data['obs1'], volatile=True).float().cuda()
        rewards = Variable(batch_data['rewards']).float().cuda()
        actions = Variable(batch_data['actions']).long().cuda().view(-1, 1)
        terminals = Variable(batch_data['terminals1']).float().cuda()

        q_vals = self.q_net(obs0)
        action_q_vals = torch.gather(q_vals, 1, actions)

        target_net_q_vals = self.target_q_net(vol_obs1)
        target_action_q_vals, _ = torch.max(target_net_q_vals, 1)
        target_action_q_vals.volatile = False
        target_action_q_vals = target_action_q_vals.view(-1, 1)
        q_label = rewards + self.gamma * target_action_q_vals * (1 - terminals)

        self.q_net.zero_grad()
        net_loss = self.q_net_loss(action_q_vals, q_label)
        net_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return net_loss.cpu().data.numpy()

    def epsilon_greedy_policy(self, q_values):
        ran = np.random.random(1)[0]
        if ran < self.epsilon:
            act = np.random.randint(self.ac_dim, size=q_values.shape[0])
            act = Variable(torch.from_numpy(act))
        else:
            max_val, act = torch.max(q_values, dim=-1)
        act = act.cpu().data.numpy()[0]
        return act

    def greedy_policy(self, q_values):
        max_val, act = torch.max(q_values, dim=-1)
        act = act.cpu().data.numpy()[0]
        return act

    def log_model_weights(self, step):
        for name, param in self.q_net.named_parameters():
            self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

    def save_model(self, is_best, step):
        ckpt_file = os.path.join(self.model_dir,
                                 'ckpt_{:010d}.pth'.format(step))
        self.save_checkpoint({
            'ckpt_step': step,
            'state_dict': self.q_net.state_dict(),
            'optimizer': self.optimizer.state_dict()},
            is_best=is_best,
            filename=ckpt_file)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        print_yellow('Saving checkpoint: %s' % filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.model_dir, 'model_best.pth'))

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def nice_log(self, stats, step):
        for key, val in stats.items():
            self.summary_writer.add_scalar(key, float(val), step)
        # This part of logging code comes from openai/baselines
        def truncate_string(s):
            return s[:20] + '...' if len(s) > 23 else s
        stats_str = {}
        for key, val in stats.items():
            if isinstance(val, float):
                str_val = '{:<10.4f}'.format(val)
            else:
                str_val = str(val)
            stats_str[truncate_string(key)] = truncate_string(str_val)

        # Find max widths
        if len(stats_str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, stats_str.keys()))
            valwidth = max(map(len, stats_str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(stats_str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        sys.stdout.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        sys.stdout.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--save_interval', type=int, default=50, help='save model every n episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='logging every n episodes')
    parser.add_argument('--render', help='render', type=int, default=1)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--train_freq', help='train_frequency', type=int, default=1)
    parser.add_argument('--max_episode', help='maximum episode', type=int, default=None)
    parser.add_argument('--max_timesteps', help='maximum timestep', type=int, default=100000000)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate')
    parser.add_argument('--gamma', help='discount_factor', type=float, default=0.99)
    parser.add_argument('--warmup_mem', type=int, help='warmup memory size', default=1000)
    parser.add_argument('--memory', help='memory size', type=int, default=1000000)
    parser.add_argument('--initial_epsilon', '-ie', help='initial_epsilon', type=float, default=0.5)
    parser.add_argument('--final_epsilon', '-fe', help='final_epsilon', type=float, default=0.05)
    parser.add_argument('--max_epsilon_decay_steps', '-eds', help='maximum steps to decay epsilon', type=int, default=100000)
    parser.add_argument('--max_grad_norm', type=float, default=None, help='maximum gradient norm')
    parser.add_argument('--soft_update', '-su', action='store_true', help='soft update target network')
    parser.add_argument('--tau', type=float, default=0.01, help='tau for soft target network update')
    parser.add_argument('--hard_update_freq', '-huf', type=int, default=500, help='hard target network update frequency')
    parser.add_argument('--save_dir', type=str, default='./data')
    return parser.parse_args()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def wrap_env(env, args):
    monitor_dir = os.path.join(args.save_dir, 'monitor')
    os.makedirs(monitor_dir, exist_ok=True)
    if args.render:
        if args.max_episode is not None:
            video_save_interval = int(args.max_episode / 3)
        else:
            video_save_interval = int(args.max_timesteps / env._max_episode_steps / 3)
        env = Monitor(env, directory=monitor_dir,
                      video_callable=lambda episode_id: episode_id % video_save_interval == 0,
                      force=True)
    else:
        env = Monitor(env, directory=monitor_dir, video_callable=False, force=True)
    return env

def main(args):
    args = parse_arguments()
    print_green('Program starts at: \033[92m %s \033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    set_random_seed(args.seed)
    env = gym.make(args.env)

    dele = input("Do you wanna recreate ckpt and log folders? (y/n)")
    if dele == 'y':
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)

    env = wrap_env(env, args)
    q_net = DQNetworkFC
    agent = DQNAgent(env=env, qnet=q_net, args=args)
    agent.train()
    print_green('Program ends at: \033[92m %s \033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == '__main__':
    main(sys.argv)
