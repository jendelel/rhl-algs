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

HumanFeedback = namedtuple('HumanFeedback', ['feedback_value'])
SavedAction = namedtuple('SavedAction', ['state', 'action', 'prob'])
SavedActionsWithFeedback = namedtuple('SavedActionsWithFeedback', ['saved_actions', 'final_feedback'])

def parse_args(parser):
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.00025,
                        help='Learning rate (default:0.00025)')
    parser.add_argument('--eligibility_decay', type=float, default=0.35,
                        help='Learning rate (default:0.01)')
    parser.add_argument("--coach_window_size", type=int, default=10, 
                        help="Number of transitions in a window.")
    parser.add_argument('--entropy_reg', type=float, default=1.5,
                            help='Entropy regularization beta')
    parser.add_argument('--feedback_delay_factor', type=int, default=1,
                            help='COACH Feedback delay factor.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')

class DeepCoach():
    def __init__(self, window, args, env):
        self.window = window
        self.args = args
        self.env = env
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if not args.no_cuda else "cpu") 

        if window is not None: 
            self.setup_ui(window)
        self.policy_net = PolicyNet(env.observation_space.shape[0], int(env.action_space.n)).to(device=self.device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=args.learning_rate)
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
            but.clicked.connect(lambda bla, def_arg=i: self.buttonClicked(def_arg))
            hor.addWidget(but)

        if window.feedback_widget.layout():
            l = window.feedback_widget.layout()
            del l
        window.feedback_widget.setLayout(hor)
        window.keyPressedSignal.connect(keyPressed)

    def buttonClicked(self, value):
        self.feedback = HumanFeedback(feedback_value=value)

    def to_tensor(self, value):
        return torch.tensor(value).float().to(device=self.device)

    def select_action(self, state):
        state = torch.from_numpy(state).to(device=self.device).float()
        action_probs = self.policy_net(state)
        max_action_prob, max_action = torch.max(action_probs, dim=0)
        # print(action_rewards.detach().cpu().numpy(), max_action.item())
        return max_action_prob.item(), max_action.item(), action_probs

    def update_net(self, savedActionsWithFeedback, current_action_probs):
        if not savedActionsWithFeedback: return
        print("training")
        e_losses = []
        for saf in savedActionsWithFeedback:
            final_feedback = saf.final_feedback
            for n, sa in enumerate(saf.saved_actions[::-1]):
                p = sa.prob
                _, _, action_probs = self.select_action(sa.state)
                e_loss = (self.args.eligibility_decay ** (n)) / p * action_probs[sa.action] * final_feedback
                e_losses.append(e_loss)
        action_dist = Categorical(current_action_probs)
        loss =-(self.to_tensor(1/(len(savedActionsWithFeedback))) * torch.stack(e_losses).to(device=self.device).sum() + self.args.entropy_reg * action_dist.entropy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def processFeedback(self, savedActions, buffer):
        feedback = self.feedback.feedback_value
        if feedback is not None and len(savedActions) > 0:
            print("Feedback: ", feedback)
            window_size = min(len(savedActions), self.args.coach_window_size)
            del savedActions[:-(window_size+self.args.feedback_delay_factor)]
            window = savedActions[:-self.args.feedback_delay_factor] # Copy the list
            savedActionsWithFeedback = SavedActionsWithFeedback(saved_actions=window, final_feedback=feedback)
            buffer.append(savedActionsWithFeedback)
        self.feedback = None

    def train(self):
        buffer = []
        running_reward = 10
        for i_episode in range(1, 10000):
                state, ep_reward = self.env.reset(), 0
                savedActions = []
                for t in range(1, 10000):  # Don't infinite loop while learning
                    action_prob, action, action_probs = self.select_action(state)
                    old_state = state
                    state, reward, done, _ = self.env.step(action)
                    ep_reward += reward
                    savedActions.append(SavedAction(state=state, action=action, prob=action_prob))
                    self.window.render(self.env)
                    if not self.window.isVisible():
                        break
                    if self.feedback:
                        self.processFeedback(savedActions, buffer)
                        if len(buffer[-1].saved_actions) > 0 and self.window.trainCheck.isChecked():
                            self.update_net([buffer[-1]], self.select_action(old_state)[2])
                    time.sleep(self.args.render_sleep)
                    if len(buffer) >= self.args.batch_size and self.window.trainCheck.isChecked():
                        indicies = random.sample(range(len(buffer)), self.args.batch_size)
                        mini_batch = [buffer[i] for i in indicies]
                        self.update_net(mini_batch, action_probs)
                    print("Action: %d, Reward: %d, ep_reward: %d" % (action, reward, ep_reward))
                    if done:
                        if t <= 190:
                            self.feedback = HumanFeedback(feedback_value=-1)
                            self.processFeedback(savedActions, buffer)
                        break
                if not self.window.isVisible():
                        break
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                print("Running reward %d"% running_reward) 

def start(window, args, env):
    alg = DeepCoach(window, args, env)
    print("Number of trainable parameters:", utils.count_parameters(alg.policy_net))
    alg.train()
    env.close()

class PolicyNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNet, self).__init__()
        self.hidden1 = nn.Linear(observation_space, 16)
        # self.hidden2 = nn.Linear(30, 30)
        self.action_probs = nn.Linear(16, action_space)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        action_probs = F.softmax(self.action_probs(x), dim=0)
        return action_probs