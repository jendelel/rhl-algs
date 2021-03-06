from PyQt5 import QtGui, QtCore, QtWidgets
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import scipy.signal
from utils import utils, logx


def parse_args(parser):
    parser.add_argument(
            '--epochs',
            type=int,
            default=400,
            help='Number of epochs of interaction (equivalent to'
            'number of policy updates) to perform. default:400')
    parser.add_argument(
            '--eval_epochs', type=int, default=10, help='Number of epochs to render for evaluation. default:200')
    parser.add_argument(
            '--start_steps',
            type=int,
            default=1000,
            help="Number of steps for uniform-random action selection,"
            "before running real policy. Helps exploration. default:1000")
    parser.add_argument(
            '--replay_size', type=int, default=int(1e6), help='Maximum size of the replay buffer. default:1e6')
    parser.add_argument(
            '--batch_size', type=int, default=100, help='Batch size (how many episodes per batch). default: 100')
    parser.add_argument('--lr_pi', type=float, default=1e-3, help='Learning rate for policy optimizer. (default:1e-3)')
    parser.add_argument(
            '--lr_q', type=float, default=1e-3, help='Learning rate for the Q-networks optimizer. (default:1e-3)')
    parser.add_argument(
            '--noise_clip',
            type=float,
            default=0.5,
            help='Limit for absolute value of target policy smoothing noise.(default:0.5)')
    parser.add_argument(
            "--target_noise",
            type=float,
            default=0.2,
            help="Stddev for smoothing noise added to target policy. default: 0.2)")
    parser.add_argument(
            "--gamma", type=float, default=0.99, help="Discount factor. (Always between 0 and 1., default: 0.999")
    parser.add_argument(
            "--action_noise",
            type=float,
            default=0.1,
            help="Stddev for Gaussian exploration noise added to"
            "policy at training time. (At test time, no noise is added.) default:0.1")
    parser.add_argument(
            "--polyak",
            type=float,
            default=0.995,
            help="Interpolation factor in polyak averaging for target"
            "networks. Target networks are updated towards main networks"
            "according to:"
            ".. math:: \\theta_{\\text{targ}} \\leftarrow"
            "    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta"
            "where :math:`\\rho` is polyak. (Always between 0 and 1, usually"
            "close to 1.) default:0.995")
    parser.add_argument(
            '--policy_delay',
            type=int,
            default=2,
            help="Policy will only be updated once every"
            "policy_delay times for each update of the Q-networks. Default: 2")
    parser.add_argument(
            '--max_episode_len',
            type=int,
            default=1000,
            help='Maximum length of trajectory / episode / rollout. default: 1000')
    parser.add_argument(
            '--save_freq',
            type=int,
            default=50,
            help='How often (in terms of gap between epochs) to save the current policy and value function. default: 50'
    )
    parser.add_argument(
            '--log_freq', type=int, default=5, help='How often (in terms of gap between epochs) to log. default: 5')
    parser.add_argument('--her_k', type=int, default=0, help='K > 0 enables hindsight experience replay default:0.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')


class TD3():

    def __init__(self, window, args, env):
        self.window = window
        self.args = args
        self.env = env
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.device = torch.device("cuda" if not args.no_cuda else "cpu")
        self.render_enabled = True
        self.renderSpin = None
        self.logger = logx.EpochLogger()

        if window is not None:
            self.setup_ui(window)

        self.obs_dim = self.env.unwrapped.observation_space.shape[0]
        if self.args.her_k > 0 and self.args.env.startswith("MountainCar"):
            self.her_goal_f = lambda obs: [obs[0]]
            self.env_target = [self.env.unwrapped.goal_position]
            self.obs_dim = self.env.unwrapped.observation_space.shape[0] + len(self.env_target)
        elif self.args.her_k > 0 and self.args.env.startswith("LunarLander"):
            self.her_goal_f = lambda obs: obs
            self.env_target = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
            self.obs_dim = self.env.unwrapped.observation_space.shape[0] * 2
        elif self.args.her_k > 0:
            raise ValueError("Can't use her for {}. Please setup the target state!".format(self.args.env))

        self.main_net = ActorCritic(
                observation_space_shape=self.obs_dim,
                action_space=self.env.unwrapped.action_space).to(device=self.device)

        self.target_net = ActorCritic(
                observation_space_shape=self.obs_dim,
                action_space=self.env.unwrapped.action_space).to(device=self.device)

        self.optimizer_pi = torch.optim.Adam(self.main_net.policy.parameters(), lr=args.lr_pi)
        self.optimizer_q = torch.optim.Adam(
                itertools.chain(self.main_net.q1.parameters(), self.main_net.q2.parameters()), lr=args.lr_q)
        self.target_net.load_state_dict(self.main_net.state_dict())

    def setup_ui(self, window):

        @QtCore.pyqtSlot(QtGui.QKeyEvent)
        def keyPressed(event):
            print("ERROR: Unknown key: ", event)

        @QtCore.pyqtSlot(int)
        def checkedChanged(state):
            print("State: ", state)
            self.render_enabled = state > 0

        hor = QtWidgets.QHBoxLayout()
        self.renderSpin = QtWidgets.QSpinBox()
        self.renderSpin.setRange(1, 1000)
        self.renderSpin.setSingleStep(5)
        self.renderSpin.setValue(100)
        renderCheck = QtWidgets.QCheckBox()
        renderCheck.setChecked(True)
        renderCheck.stateChanged.connect(checkedChanged)
        hor.addWidget(self.renderSpin)
        hor.addWidget(renderCheck)

        window.feedback_widget.setLayout(hor)
        window.keyPressedSignal.connect(keyPressed)

    def get_action(self, obs, noise_scale):
        action_limit = self.env.unwrapped.action_space.high[0]
        if self.args.her_k > 0:
            obs = np.concatenate([obs, self.env_target], axis=0)
        pi = self.main_net.policy(torch.Tensor(obs).to(self.device).unsqueeze(dim=0))
        action = pi.detach().cpu().numpy()[0] + noise_scale * np.random.randn(self.env.unwrapped.action_space.shape[0])
        return np.clip(action, -action_limit, action_limit)

    def test_agent(self, n=10):
        for test_epoch in range(n):
            obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0
            while not (done or (episode_len == self.args.max_episode_len)):
                obs, reward, done, _, = self.env.step(self.get_action(obs, 0))
                self.window.processEvents()
                if self.render_enabled and test_epoch % self.renderSpin.value() == 0:
                    self.window.render(self.env)
                    time.sleep(self.window.renderSpin.value())
                if not self.window.isVisible():
                    return
                episode_ret += reward
                episode_len += 1
            self.logger.store(TestEpRet=episode_ret, TestEpLen=episode_len)

    def update_net(self, buffer, episode_len):

        def t(arr):
            return torch.Tensor(arr).to(self.device)

        for j in range(episode_len):
            batch = buffer.sample_batch(self.args.batch_size)
            (obs1, obs2, actions, rewards, done) = (t(batch['obs1']), t(batch['obs2']), t(batch['acts']),
                                                    t(batch['rews']), t(batch['done']))
            q1 = self.main_net.q1(torch.cat([obs1, actions], dim=1))
            q2 = self.main_net.q2(torch.cat([obs1, actions], dim=1))

            pi_target = self.target_net.policy(obs2)
            # Target policy smoothing by adding clipped noise to target actions
            action_limit = self.env.unwrapped.action_space.high[0]
            epsilon = torch.normal(torch.zeros_like(pi_target), self.args.target_noise * torch.ones_like(pi_target))
            epsilon = torch.clamp(epsilon, -self.args.noise_clip, self.args.noise_clip)
            a2 = torch.clamp(pi_target + epsilon, -action_limit, action_limit)

            # Target Q-values, using action form target policy
            q1_target = self.target_net.q1(torch.cat([obs2, a2], dim=1))
            q2_target = self.target_net.q2(torch.cat([obs2, a2], dim=1))

            # Bellman backup for Q functions, using Clipped Double-Q targets
            min_q_targ = torch.min(q1_target, q2_target)
            backup = (rewards + self.args.gamma * (1 - done) * min_q_targ).detach()

            # TD3 Q losses
            q1_loss = F.mse_loss(q1, backup)
            q2_loss = F.mse_loss(q2, backup)
            q_loss = q1_loss + q2_loss

            self.optimizer_q.zero_grad()
            q_loss.backward()
            self.optimizer_q.step()

            self.logger.store(LossQ=q_loss.item(), Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy())

            if j % self.args.policy_delay == 0:
                q1_pi = self.main_net.q1(torch.cat([obs1, self.main_net.policy(obs1)], dim=1))
                pi_loss = -q1_pi.mean()  # Maximize the Q function.

                self.optimizer_pi.zero_grad()
                pi_loss.backward()
                self.optimizer_pi.step()

                # Polyak averaging for target variables
                for p_main, p_target, in zip(self.main_net.parameters(), self.target_net.parameters()):
                    p_target.data.copy_(self.args.polyak * p_target.data + (1 - self.args.polyak) * p_main.data)

                self.logger.store(LossPi=pi_loss.item())

    def train(self):
        self.logger.save_config({"args:": self.args})
        if self.args.her_k > 0:
            buffer = ReplayBuffer(
                    obs_dim=self.obs_dim, act_dim=self.env.unwrapped.action_space.shape[0], size=self.args.replay_size)
        else:
            buffer = ReplayBuffer(
                    obs_dim=self.obs_dim, act_dim=self.env.unwrapped.action_space.shape[0], size=self.args.replay_size)

        var_counts = tuple(
                utils.count_parameters(module)
                for module in [self.main_net.policy, self.main_net.q1, self.main_net.q2, self.main_net])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        start_time = time.time()
        obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0
        tot_steps = 0
        for epoch in range(0, self.args.epochs):
            episode_buffer = []
            # Set the network in eval mode (e.g. Dropout, BatchNorm etc.)
            for t in range(self.args.max_episode_len):
                self.window.processEvents()
                if not self.window.isVisible():
                    return
                if tot_steps > self.args.start_steps:
                    action = self.get_action(obs, self.args.action_noise)
                else:
                    action = self.env.action_space.sample()

                # Step in the env
                obs2, reward, done, _ = self.env.step(action)
                tot_steps += 1
                episode_len += 1
                episode_ret += reward

                done = False if episode_len == self.args.max_episode_len else done
                # Save and log
                episode_buffer.append((obs, action, reward, obs2, done))
                # buffer.store(obs, action, reward, obs2, done)
                obs = obs2

                if done or (episode_len == self.args.max_episode_len):
                    if self.args.her_k > 0:
                        for trans_id, (obs, action, reward, obs2, done) in enumerate(episode_buffer):
                            buffer.store(
                                    np.concatenate([obs, self.env_target], axis=0), action, reward,
                                    np.concatenate([obs2, self.env_target], axis=0), done)
                            for k in range(self.args.her_k):
                                future_exp = np.random.randint(trans_id, len(episode_buffer))
                                _, _, _, her_obs2, _ = episode_buffer[future_exp]
                                her_reward, her_done = (reward + 100, True) if np.allclose(
                                        self.her_goal_f(her_obs2), self.her_goal_f(obs2)) else (reward, done)
                                buffer.store(
                                        np.concatenate([obs, self.her_goal_f(her_obs2)], axis=0), action, her_reward,
                                        np.concatenate([obs2, self.her_goal_f(her_obs2)], axis=0), her_done)

                    else:
                        for obs, action, reward, obs2, done in episode_buffer:
                            buffer.store(obs, action, reward, obs2, done)

                    self.update_net(buffer, episode_len)
                    self.logger.store(EpRet=episode_ret, EpLen=episode_len)
                    obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0

                    if (epoch % self.args.save_freq == 0) or (epoch == self.args.epochs - 1):
                        self.logger.save_state({'env': self.env}, self.main_net, None)

                    if epoch % self.args.log_freq == 0:
                        self.test_agent(self.args.eval_epochs)
                        # Log info about epoch
                        self.logger.log_tabular(tot_steps, 'Epoch', epoch)
                        self.logger.log_tabular(tot_steps, 'EpRet', with_min_and_max=True)
                        self.logger.log_tabular(tot_steps, 'TestEpRet', with_min_and_max=True)
                        self.logger.log_tabular(tot_steps, 'EpLen', average_only=True)
                        self.logger.log_tabular(tot_steps, 'TestEpLen', average_only=True)
                        self.logger.log_tabular(tot_steps, 'TotalEnvInteracts', tot_steps)
                        self.logger.log_tabular(tot_steps, 'Q1Vals', with_min_and_max=True)
                        self.logger.log_tabular(tot_steps, 'Q2Vals', with_min_and_max=True)
                        self.logger.log_tabular(tot_steps, 'LossPi', average_only=True)
                        self.logger.log_tabular(tot_steps, 'LossQ', average_only=True)
                        self.logger.log_tabular(tot_steps, 'Time', time.time() - start_time)
                        self.logger.dump_tabular()
                    break


def start(window, args, env):
    alg = TD3(window, args, env)
    print("Number of trainable parameters:", utils.count_parameters(alg.main_net))
    alg.train()
    print("Done")
    env.close()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
                obs1=self.obs1_buf[idxs],
                obs2=self.obs2_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs])


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        self.output_scale = output_scale

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


class ActorCritic(nn.Module):

    def __init__(self,
                 observation_space_shape,
                 action_space,
                 hidden_sizes=[32, 32],
                 activation=F.relu,
                 output_activation=torch.tanh,
                 policy=None):
        super(ActorCritic, self).__init__()

        action_dim = action_space.shape[0]
        action_scale = action_space.high[0]

        self.policy = MLP(
                layers=[observation_space_shape] + list(hidden_sizes) + [action_dim],
                activation=activation,
                output_activation=output_activation,
                output_scale=action_scale)
        self.q1 = MLP(
                layers=[observation_space_shape + action_dim] + list(hidden_sizes) + [1],
                activation=activation,
                output_squeeze=True)
        self.q2 = MLP(
                layers=[observation_space_shape + action_dim] + list(hidden_sizes) + [1],
                activation=activation,
                output_squeeze=True)

    def forward(self, x, action_taken):
        print(x)
        pi = self.policy(x)

        q1 = self.q1(torch.cat([x, action_taken], dim=1))
        q2 = self.q2(torch.cat([x, action_taken], dim=1))

        q1_pi = self.q1(torch.cat([x, pi], dim=1))

        return pi, q1, q2, q1_pi
