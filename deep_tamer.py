from PyQt5 import QtGui, QtCore, QtWidgets
from collections import namedtuple
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import utils

HumanFeedback = namedtuple('HumanFeedback', ['feedback_time', 'feedback_value'])
SavedAction = namedtuple('SavedAction', ['state', 'action_index', 'start_time', 'end_time'])
SavedActionWithReward = namedtuple('SavedActionWithReward', ['saved_action', 'reward'])

def parse_args(parser):
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch_size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default:0.01)')
    parser.add_argument('--tamer_lower_bound', type=float, default=0.2,
                            help='TAMER Window lower bound')
    parser.add_argument('--tamer_upper_bound', type=float, default=2.0,
                            help='TAMER Window lower bound')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')

class DeepTamer():
    def __init__(self, window, args, env):
        self.window = window
        self.args = args
        self.env = env
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if not args.no_cuda else "cpu") 

        self.setup_ui(window)
        self.reward_net = RewardNetwork(env.observation_space.shape[0], int(env.action_space.n)).to(device=self.device)
        self.optimizer = torch.optim.RMSprop(self.reward_net.parameters(), lr=args.learning_rate)
        self.replay_buffer = []
        self.feedback = None
    
    def setup_ui(self, window):
        @QtCore.pyqtSlot(QtGui.QKeyEvent)
        def keyPressed(event):
            numpad_mod = int(event.modifiers()) & QtCore.Qt.KeypadModifier
            if (event.key() == QtCore.Qt.Key_Minus and numpad_mod) or event.key() == QtCore.Qt.Key_M:
                self.buttonClicked(-1)
            elif (event.key() == QtCore.Qt.Key_Plus and numpad_mod) or event.key() == QtCore.Qt.Key_P:
                self.buttonClicked(1)
            else:
                print("ERROR: Unknown key: ", event)
        hor = QtWidgets.QHBoxLayout()
        for i in range(-1, 2):
            if i == 0: continue
            but = QtWidgets.QPushButton()
            but.setText(str(i))
            but.clicked.connect(lambda: self.buttonClicked(i))
            hor.addWidget(but)

        if window.feedback_widget.layout():
            l = window.feedback_widget.layout()
            del l
        window.feedback_widget.setLayout(hor)
        window.keyPressedSignal.connect(keyPressed)

    def buttonClicked(self, value):
        self.feedback = HumanFeedback(feedback_time=time.time(), feedback_value=value)

    def select_action(self, state):
        state = torch.from_numpy(state).to(device=self.device).float()
        action_rewards = self.reward_net(state)
        max_action = torch.argmax(action_rewards, dim=0)
        # print(action_rewards.detach().cpu().numpy(), max_action.item())
        return action_rewards, max_action.item()

    def update_net(self, savedActionsWithReward):
        if not savedActionsWithReward: return
        print("training")
        losses = []
        mseLoss = torch.nn.MSELoss()
        for sar in savedActionsWithReward:
            action_rewards, _ = self.select_action(sar.saved_action.state)
            action_reward_tensor = action_rewards[sar.saved_action.action_index]
            reward = torch.tensor(sar.reward, dtype=torch.float32).to(device=self.device)
            losses.append(mseLoss(action_reward_tensor, reward))
        self.optimizer.zero_grad()
        loss = torch.stack(losses).to(device=self.device).sum()
        loss.backward()
        self.optimizer.step()
    
    def processFeedback(self, savedActions, buffer):
        feedback_time = self.feedback.feedback_time
        new_data = [SavedActionWithReward(saved_action=sa, reward=self.feedback.feedback_value)
                    for sa in savedActions 
                    if (feedback_time-sa.start_time) > self.args.tamer_lower_bound and (feedback_time-sa.end_time) < self.args.tamer_upper_bound]
        self.update_net(new_data)
        buffer.extend(new_data)
        self.feedback = None

    # def calc_weight(start_time, end_time, feedback_time):
        #     # DeepTamer uses f_delay as uniform distribution [0.2, 4]
        #     # \int_{tf-te}^{tf-ts} f_delay(t) dt
        #     return (end_time - start_time) # * (4-0.2)

    def train(self):
        buffer = []
        running_reward = 10
        for i_episode in range(1, 10000):
                state, ep_reward, esp_reward = self.env.reset(), 0, 0
                savedActions = []
                for t in range(1, 10000):  # Don't infinite loop while learning
                    start_time = time.time()
                    rewards, action = self.select_action(state)
                    state, reward, done, _ = self.env.step(action)
                    end_time = time.time()
                    es_reward = rewards[action].item()
                    ep_reward += reward
                    esp_reward += es_reward
                    savedActions.append(SavedAction(state=state, action_index=action, start_time=start_time, end_time=end_time))
                    self.window.render(self.env)
                    if not self.window.isVisible():
                        break
                    if self.feedback:
                        self.processFeedback(savedActions, buffer)
                    time.sleep(0.1)
                    if t % 10 == 0 and len(buffer) > self.args.batch_size:
                        indicies = random.sample(range(len(buffer)), self.args.batch_size)
                        mini_batch = [buffer[i] for i in indicies]
                        self.update_net(mini_batch)
                    print("Action: %d, Reward: %d, es_reward: %d, ep_reward: %d, esp_reward: %d" % 
                                                            (action, reward, es_reward, ep_reward, esp_reward))
                    if done:
                        # if t < 190:
                        #     self.feedback = HumanFeedback(feedback_time=time.time(), feedback_value=-1)
                        #     processFeedback(savedActions, buffer)
                        break   
                if not self.window.isVisible():
                        break
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                print("Running reward %d"% running_reward) 

def start(window, args, env):
    alg = DeepTamer(window, args, env)
    print("Number of trainable parameters:", utils.count_parameters(alg.reward_net))
    alg.train()
    env.close()

class RewardNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(RewardNetwork, self).__init__()
        self.hidden = nn.Linear(observation_space, 16)
        self.reward_head = nn.Linear(16, action_space)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden(x))
        action_rewards = self.reward_head(x)
        return action_rewards