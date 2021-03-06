from PyQt5 import QtGui, QtCore, QtWidgets
import time
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
            default=200,
            help='Number of epochs of interaction (equivalent to'
            'number of policy updates) to perform. default:200')
    parser.add_argument(
            '--eval_epochs', type=int, default=10, help='Number of epochs to render for evaluation. default:200')
    parser.add_argument(
            '--batch_size', type=int, default=10, help='Batch size (how many episodes per batch). default: 10')
    parser.add_argument('--lr_pi', type=float, default=0.05, help='Learning rate for policy optimizer. (default:0.05)')
    parser.add_argument(
            '--lr_V', type=float, default=0.05, help='Learning rate for value function optimizer. (default:0.05)')
    parser.add_argument(
            '--train_v_iters',
            type=int,
            default=1,
            help='Number of gradient descent steps to take on value function per epoch.(default:80)')
    parser.add_argument(
            "--gae_lambda",
            type=float,
            default=0.97,
            help="Lambda for GAE-Lambda. (Always between 0 and 1, close to 1., default: 0.97)")
    parser.add_argument(
            "--gae_gamma", type=float, default=0.99, help="Discount factor. (Always between 0 and 1., default: 0.99")
    parser.add_argument(
            '--max_episode_len',
            type=int,
            default=500,
            help='Maximum length of trajectory / episode / rollout. default: 1000')
    parser.add_argument(
            '--save_freq',
            type=int,
            default=10,
            help='How often (in terms of gap between epochs) to save the current policy and value function. default: 10'
    )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')


class VPG():

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
        self.actor_critic = ActorCritic(
                observation_space_shape=self.env.unwrapped.observation_space.shape[0],
                action_space=self.env.unwrapped.action_space).to(device=self.device)

        self.optimizer_pi = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=args.lr_pi)
        self.optimizer_V = torch.optim.Adam(self.actor_critic.value_function.parameters(), lr=args.lr_V)

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

    def select_action(self, obs, action_taken=None):
        action, logp, logp_pi = self.actor_critic.policy(obs, action_taken)
        return action, logp, logp_pi

    def update_net(self, buffer_minibatch):
        obs, act, adv, ret, logp_old = [torch.Tensor(x).to(self.device) for x in buffer_minibatch]
        _, logp, _ = self.select_action(obs, action_taken=act)
        # Estimate the entropy E[-logp]
        entropy_est = (-logp).mean()

        # Policy gradient step
        pi_loss = -(logp * adv).mean()
        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        # Value function learning
        # MSE of the value function and the returns
        v = self.actor_critic.value_function(obs)
        v_loss_old = F.mse_loss(v, ret)
        for _ in range(self.args.train_v_iters):
            v = self.actor_critic.value_function(obs)
            v_loss = F.mse_loss(v, ret)

            # V function gradient step
            self.optimizer_V.zero_grad()
            v_loss.backward()
            self.optimizer_V.step()

        _, logp, _, v = self.actor_critic(obs, act)
        pi_loss_new = -(logp * adv).mean()
        v_loss_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        self.logger.store(
                LossPi=pi_loss,
                LossV=v_loss_old,
                KL=kl,
                Entropy=entropy_est,
                DeltaLossPi=(pi_loss_new - pi_loss),
                DeltaLossV=(v_loss_old - v_loss_old))

    def train(self):
        self.logger.save_config({"args:": self.args})
        buffer = VPGBuffer(
                obs_dim=self.env.unwrapped.observation_space.shape,
                act_dim=self.env.unwrapped.action_space.shape,
                size=(self.args.batch_size + 1) * self.args.max_episode_len,
                gamma=self.args.gae_gamma,
                lam=self.args.gae_lambda)

        var_counts = tuple(
                utils.count_parameters(module)
                for module in [self.actor_critic.policy, self.actor_critic.value_function])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        start_time = time.time()
        tot_steps = 0
        obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0
        for epoch in range(0, self.args.epochs):
            # Set the network in eval mode (e.g. Dropout, BatchNorm etc.)
            self.actor_critic.eval()
            for t in range(self.args.max_episode_len):
                action, _, logp_t, v_t = self.actor_critic(torch.Tensor(obs).unsqueeze(dim=0).to(self.device))

                # Save and log
                buffer.store(obs, action.detach().cpu().numpy(), reward, v_t.item(), logp_t.detach().cpu().numpy())
                self.logger.store(VVals=v_t)

                obs, reward, done, _ = self.env.step(action.detach().cpu().numpy()[0])
                tot_steps += 1
                episode_ret += reward
                episode_len += 1

                self.window.processEvents()
                if self.render_enabled and epoch % self.renderSpin.value() == 0:
                    self.window.render(self.env)
                    time.sleep(self.window.renderSpin.value())
                if not self.window.isVisible():
                    return

                terminal = done or (episode_len == self.args.max_episode_len)
                if terminal:
                    if not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % episode_len)
                    last_val = reward if done else self.actor_critic.value_function(
                            torch.Tensor(obs).to(self.device).unsqueeze(dim=0)).item()
                    buffer.finish_path(last_val=last_val)

                    if epoch % self.args.batch_size == 0:
                        self.actor_critic.train()  # Switch module to training mode
                        self.update_net(buffer.get())
                        self.actor_critic.eval()
                    if terminal:
                        self.logger.store(EpRet=episode_ret, EpLen=episode_len)
                    obs, reward, done, episode_ret, episode_len = self.env.reset(), 0, False, 0, 0
                    break

            if (epoch % self.args.save_freq == 0) or (epoch == self.args.epochs - 1):
                self.logger.save_state({'env': self.env.unwrapped}, self.actor_critic, None)
                pass

            # Log info about epoch
            self.logger.log_tabular(tot_steps, 'Epoch', epoch)
            self.logger.log_tabular(tot_steps, 'EpRet', with_min_and_max=True)
            self.logger.log_tabular(tot_steps, 'EpLen', average_only=True)
            self.logger.log_tabular(tot_steps, 'VVals', with_min_and_max=True)
            self.logger.log_tabular(tot_steps, 'TotalEnvInteracts', tot_steps)
            if epoch % self.args.batch_size == 0:
                self.logger.log_tabular(tot_steps, 'LossPi', average_only=True)
                self.logger.log_tabular(tot_steps, 'LossV', average_only=True)
                self.logger.log_tabular(tot_steps, 'DeltaLossPi', average_only=True)
                self.logger.log_tabular(tot_steps, 'DeltaLossV', average_only=True)
                self.logger.log_tabular(tot_steps, 'Entropy', average_only=True)
                self.logger.log_tabular(tot_steps, 'KL', average_only=True)
            self.logger.log_tabular(tot_steps, 'Time', time.time() - start_time)
            self.logger.dump_tabular()

    def eval(self):
        # Final evaluation
        print("Eval")
        episode_reward = 0
        for t in range(self.args.eval_epochs):
            state, done = self.env.reset(), False
            while not done:
                action = self.actor_critic.policy.eval(torch.Tensor(state).to(self.device)).detach().cpu().numpy()

                # Choose greedy action this time
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if self.render_enabled:
                    self.window.render(self.env)
                    time.sleep(self.window.renderSpin.value())
                if not self.window.isVisible():
                    return


def start(window, args, env):
    alg = VPG(window, args, env)
    print("Number of trainable parameters:", utils.count_parameters(alg.actor_critic))
    alg.train()
    alg.eval()
    print("Done")
    env.close()


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # \delta_t =  - V(s_t) + r_t + \gamma * V_(s_{t+1})
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # GAE_advantage = \Sum_l (\lambda * \gamma)^l * delta_{t+1}
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        # self.adv_buf[path_slice] = self.ret_buf[path_slice] - self.val_buf[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        buffer_slice = slice(0, self.ptr)
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf[buffer_slice]), np.std(self.adv_buf[buffer_slice])
        self.adv_buf[buffer_slice] = (self.adv_buf[buffer_slice] - adv_mean) / (adv_std + 1e-5)
        # TODO: Consider returning a dictionary.
        return [
                self.obs_buf[buffer_slice], self.act_buf[buffer_slice], self.adv_buf[buffer_slice],
                self.ret_buf[buffer_slice], self.logp_buf[buffer_slice]
        ]

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=None, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicyNet(nn.Module):

    def __init__(self, observation_space_shape, hidden_sizes, activation, output_activation, action_dim):
        super(CategoricalPolicyNet, self).__init__()
        self.logits = MLP(layers=[observation_space_shape] + list(hidden_sizes) + [action_dim], activation=activation)

    def forward(self, x, action_taken=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        # Sample the action.
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if action_taken is not None:
            logp = policy.log_prob(action_taken).squeeze()
        else:
            logp = None
        return pi, logp, logp_pi

    def eval(self, x):
        logits = self.logits(x)
        return torch.argmax(logits, dim=-1)


class GaussianPolicyNet(nn.Module):

    def __init__(self, observation_space_shape, hidden_sizes, activation, output_activation, action_dim):
        super(GaussianPolicyNet, self).__init__()

        self.mu = MLP(
                layers=[observation_space_shape] + list(hidden_sizes) + [action_dim],
                activation=activation,
                output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, action_taken):
        policy = Normal(self.mu(x), self.log_std.exp())
        # Sample the action from the policy.
        pi = policy.sample()
        # Sum over the actions.
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if action_taken is not None:
            logp = policy.log_prob(action_taken).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

    def eval(self, x):
        return self.mu(x)


class ActorCritic(nn.Module):

    def __init__(self,
                 observation_space_shape,
                 action_space,
                 hidden_sizes=[64],
                 activation=torch.tanh,
                 output_activation=None,
                 policy=None):
        super(ActorCritic, self).__init__()

        if policy is None and hasattr(action_space, 'n'):
            self.policy = CategoricalPolicyNet(
                    observation_space_shape, hidden_sizes, activation, output_activation, action_dim=action_space.n)
        elif policy is None:
            self.policy = GaussianPolicyNet(
                    observation_space_shape,
                    hidden_sizes,
                    activation,
                    output_activation,
                    action_dim=action_space.shape[0])
        else:
            self.policy = policy(
                    observation_space_shape,
                    hidden_sizes,
                    activation,
                    output_activation,
                    action_dim=action_space.shape[0])

        self.value_function = MLP(
                layers=[observation_space_shape] + list(hidden_sizes) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, action_taken=None):
        pi, logp, logp_pi = self.policy(x, action_taken)
        v = self.value_function(x)

        return pi, logp, logp_pi, v
