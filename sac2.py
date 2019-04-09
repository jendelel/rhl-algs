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
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. (default:1e-3)')
    parser.add_argument(
            "--gamma", type=float, default=0.99, help="Discount factor. (Always between 0 and 1., default: 0.99")
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
            '--alpha',
            type=float,
            default=-1.0,
            help="Entropy regularization coefficient. (Equivalent to"
            "inverse of reward scale in the original SAC paper.) When negative, alpha is trainable. Default: -1")
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


class SAC():

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

        if self.args.alpha < 0:
            print("Using trainable alpha")
            self.target_entropy = -torch.prod(torch.Tensor(self.env.unwrapped.action_space.shape).to(
                    self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=self.args.lr)
        else:
            self.alpha = self.args.alpha

        self.main_net = ActorCritic(
                in_features=self.obs_dim, action_space=self.env.unwrapped.action_space).to(device=self.device)

        # Only critique (Q-networks) from the target network is used!
        self.target_net = ActorCritic(
                in_features=self.obs_dim, action_space=self.env.unwrapped.action_space).to(device=self.device)

        self.optimizer_pi = torch.optim.Adam(self.main_net.policy.parameters(), lr=args.lr)
        self.optimizer_q = torch.optim.Adam(
                itertools.chain(self.main_net.q1.parameters(), self.main_net.q2.parameters()), lr=args.lr)
        self.target_net.q1.load_state_dict(self.main_net.q1.state_dict())
        self.target_net.q2.load_state_dict(self.main_net.q2.state_dict())

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

    def get_action(self, obs, deterministic=False):
        if self.args.her_k > 0:
            obs = np.concatenate([obs, self.env_target], axis=0)
        pi, mu, _ = self.main_net.policy(torch.Tensor(obs).to(self.device).unsqueeze(dim=0))
        action = mu.detach().cpu().numpy()[0] if deterministic else pi.detach().cpu().numpy()[0]
        return action

    def test_agent(self, n=10):
        for test_epoch in range(n):
            obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0
            while not (done or (episode_len == self.args.max_episode_len)):
                obs, reward, done, _, = self.env.step(self.get_action(obs, deterministic=True))
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
            _, _, logp_pi, q1, q2, q1_pi, q2_pi = self.main_net(obs1, actions)

            if self.args.alpha < 0:
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                self.optimizer_alpha.zero_grad()
                alpha_loss.backward()
                self.optimizer_alpha.step()
                alpha = self.log_alpha.exp()  # .detach() ?? (See github issue.)
                self.logger.store(LossAlpha=alpha_loss.item())
            else:
                alpha = self.args.alpha

            # Targets for Q regression
            actions_next, _, logp_pi_next = self.main_net.policy(obs2)
            q1_pi_next = self.target_net.q1(torch.cat((obs2, actions_next), dim=1))
            q2_pi_next = self.target_net.q2(torch.cat((obs2, actions_next), dim=1))

            q_next_value = torch.min(q1_pi_next, q2_pi_next) - alpha * logp_pi_next
            q_backup = (rewards + self.args.gamma * (1 - done) * q_next_value).detach()

            # We are using the current policy (sampling from fresh actions)
            min_q_pi = torch.min(q1_pi, q2_pi)

            # SAC losses
            pi_loss = (alpha * logp_pi - min_q_pi).mean()
            q1_loss = 0.5 * F.mse_loss(q1, q_backup)  # Why times 0.5?
            q2_loss = 0.5 * F.mse_loss(q2, q_backup)  # Why times 0.5?
            value_loss = q1_loss + q2_loss

            # Train policy
            self.optimizer_pi.zero_grad()
            pi_loss.backward()
            self.optimizer_pi.step()

            # Q losses
            self.optimizer_q.zero_grad()
            value_loss.backward()
            self.optimizer_q.zero_grad()
            # Polyak averaging for target variables
            for p_main, p_target, in zip(
                    itertools.chain(self.main_net.q1.parameters(), self.main_net.q2.parameters()),
                    itertools.chain(self.target_net.q1.parameters(), self.target_net.q2.parameters())):
                p_target.data.copy_(self.args.polyak * p_target.data + (1 - self.args.polyak) * p_main.data)

            self.logger.store(
                    LossPi=pi_loss.item(),
                    LossQ1=q1_loss.item(),
                    LossQ2=q2_loss.item(),
                    Alpha=alpha.item(),
                    Q1Vals=q1.detach().cpu().numpy(),
                    Q2Vals=q2.detach().cpu().numpy(),
                    LogPi=logp_pi.detach().cpu().numpy())

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
                    action = self.get_action(obs)
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
                        self.logger.log_tabular(tot_steps, 'LogPi', with_min_and_max=True)
                        self.logger.log_tabular(tot_steps, 'Alpha', average_only=True)
                        self.logger.log_tabular(tot_steps, 'LossPi', average_only=True)
                        self.logger.log_tabular(tot_steps, 'LossQ1', average_only=True)
                        self.logger.log_tabular(tot_steps, 'LossQ2', average_only=True)
                        if self.args.alpha < 0:
                            self.logger.log_tabular(tot_steps, 'LossAlpha', average_only=True)
                        self.logger.log_tabular(tot_steps, 'Time', time.time() - start_time)
                        self.logger.dump_tabular()
                    break


def start(window, args, env):
    alg = SAC(window, args, env)
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


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6


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


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation, output_activation, action_space):
        super(GaussianPolicy, self).__init__()

        action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]
        self.output_activation = output_activation

        self.net = MLP(layers=[in_features] + list(hidden_sizes), activation=activation, output_activation=activation)

        self.mu = nn.Linear(in_features=list(hidden_sizes)[-1], out_features=action_dim)
        """
        Because this algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used torch.clamp instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        self.log_std = nn.Sequential(nn.Linear(in_features=list(hidden_sizes)[-1], out_features=action_dim), nn.Tanh())

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        if self.output_activation:
            mu = self.output_activation(mu)
        log_std = self.log_std(output)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        policy = torch.distributions.Normal(mu, torch.exp(log_std))
        pi = policy.rsample()  # Critical: must be rsample() and not sample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)

        mu, pi, logp_pi = self._apply_squashing_func(mu, pi, logp_pi)

        # make sure actions are in correct range
        mu_scaled = mu * self.action_scale
        pi_scaled = pi * self.action_scale

        return pi_scaled, mu_scaled, logp_pi

    def _clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(torch.log(self._clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS), dim=1)

        return mu, pi, logp_pi


class ActorCritic(nn.Module):

    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=torch.relu,
                 output_activation=None,
                 policy=GaussianPolicy):
        super(ActorCritic, self).__init__()

        action_dim = action_space.shape[0]

        self.policy = policy(in_features, hidden_sizes, activation, output_activation, action_space)

        self.q1 = MLP([in_features + action_dim] + list(hidden_sizes) + [1], activation, output_squeeze=True)

        self.q2 = MLP([in_features + action_dim] + list(hidden_sizes) + [1], activation, output_squeeze=True)

    def forward(self, x, a):
        pi, mu, logp_pi = self.policy(x)

        q1 = self.q1(torch.cat((x, a), dim=1))
        q1_pi = self.q1(torch.cat((x, pi), dim=1))

        q2 = self.q2(torch.cat((x, a), dim=1))
        q2_pi = self.q2(torch.cat((x, pi), dim=1))

        return pi, mu, logp_pi, q1, q2, q1_pi, q2_pi
