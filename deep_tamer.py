"""
This module contains the definiton of a pyglet widget for a 
PySide application: QPygletWidget

It also provides a basic usage example.
"""
import sys
# from PyQt5.QtCore import (QPoint, QRect, pyqtSignal, QSize, Qt, pyqtSlot)
# from PyQt5.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QLabel, QFileDialog
from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL
# from PyQt5.QtGui import (QPixmap, QImage, QPainter, QColor, QPalette, QBrush)


import qimage2ndarray
class MyPygletViewer(QtWidgets.QLabel):
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        # self.setMinimumSize(QtCore.QSize(480, 640))

    def imshow(self, img):
        qimg = qimage2ndarray.array2qimage(img, True).scaled(self.size())
        self.setPixmap(QtGui.QPixmap(qimg))
        self.repaint()

from gym import Wrapper
from gym import error
class UIRenderer(Wrapper):
    def __init__(self, env, viewer):
        super(UIRenderer, self).__init__(env)
        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            raise error.Error("Cannot ui decorate env without rgb_array renderer.")
        self.viewer = viewer

    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs): 
        if mode != 'human':  
            return self.env.render(mode, **kwargs)
        else:
            img = self.env.render('rgb_array')
            self.viewer.imshow(img)
            return True
    
    def close(self):
        super(UIRenderer, self).close()

from collections import namedtuple
HumanFeedback = namedtuple('HumanFeedback', ['feedback_time', 'feedback_value'])

import gym
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
class MainWindow(QtWidgets.QMainWindow):
    keyPressedSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
              
        self.viewer = MyPygletViewer()
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        hor = QtWidgets.QHBoxLayout()
        self.startBut = QtWidgets.QPushButton()
        self.startBut.setText("Start!")

        hor.addWidget(self.startBut)
        ver = QtWidgets.QVBoxLayout()
        ver.addWidget(self.viewer)
        self.feedback_widget = QtWidgets.QWidget()
        self.feedback_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        ver.addWidget(self.feedback_widget)
        ver.addLayout(hor)
        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(ver)
        self.setCentralWidget(centralWidget)

        self.startBut.clicked.connect(self.startRl)
        self.setFixedSize(QtCore.QSize(640, 480))
        self.bringToFront()
        self.running = False
        self.feedback = None

        self.device = None
    
    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            event.accept()
            self.keyPressedSignal.emit(event)
        else:
            event.ignore()

    def bringToFront(self):
        self.setWindowState( (self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.raise_()  # for MacOS
        self.activateWindow() #  for Windows

    @QtCore.pyqtSlot()
    def startRl(self):
        if self.running: return
        self.running = True
        argv = [str(arg) for arg in QtCore.QCoreApplication.instance().arguments()]
        parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor (default: 0.99)')
        parser.add_argument('--batch_size', type=int, default=10,
                            help='batch_size (default: 32)')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Learning rate (default:0.00025)')
        parser.add_argument('--seed', type=int, default=543, metavar='N',
                            help='random seed (default: 543)')
        parser.add_argument('--render', action='store_true',
                            help='render the environment')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='disables CUDA training')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='interval between training status logs (default: 10)')
        args = parser.parse_args(argv[1:])

        # env = gym.make('BipedalWalker-v2')
        env = gym.make('CartPole-v0')
        env = UIRenderer(env, self.viewer)
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if not args.no_cuda else "cpu")

        self.deep_tamer(args, env)
        self.running = True

    def random(self, args, env):
        def select_action(state):
            return env.action_space.sample()

        for i_episode in range(1, 10):
                state, ep_reward = env.reset(), 0
                for t in range(1, 10000):  # Don't infinite loop while learning
                    action = select_action(state)
                    state, reward, done, _ = env.step(action)
                    env.render('human')
                    # self.bringToFront()
                    QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents)
                    if not self.isVisible():
                        break
                    ep_reward += reward
                    print("Reward: %d, ep_reward: %d" % (reward, ep_reward))
                    if done:
                        break
                if not self.isVisible():
                        break
        env.close()

    def buttonClicked(self, value):
        import time
        self.feedback = HumanFeedback(feedback_time=time.time(), feedback_value=value)

    @QtCore.pyqtSlot(QtGui.QKeyEvent)
    def keyPressed(self, event):
        numpad_mod = int(event.modifiers()) & QtCore.Qt.KeypadModifier
        if event.key() == QtCore.Qt.Key_Minus and numpad_mod:
            self.buttonClicked(-1)
        elif event.key() == QtCore.Qt.Key_Plus and numpad_mod:
            self.buttonClicked(1)
        else:
            print(event)

    def deep_tamer(self, args, env):
        import time
        LOWER_TIME_BOUND = 0.2
        UPPER_TIME_BOUND = 2.0
        hor = QtWidgets.QHBoxLayout()
        for i in range(-1, 2):
            if i == 0: continue
            but = QtWidgets.QPushButton()
            but.setText(str(i))
            but.clicked.connect(lambda: self.buttonClicked(i))
            hor.addWidget(but)

        if self.feedback_widget.layout():
            l = self.feedback_widget.layout()
            del l
        self.feedback_widget.setLayout(hor)
        print(self.feedback)
        self.keyPressedSignal.connect(self.keyPressed)
        reward_net = RewardNetwork(env.observation_space.shape[0], int(env.action_space.n)).to(device=self.device)
        optimizer = torch.optim.RMSprop(reward_net.parameters(), lr=args.learning_rate)
        SavedAction = namedtuple('SavedAction', ['state', 'action_index', 'start_time', 'end_time'])
        SavedActionWithReward = namedtuple('SavedActionWithReward', ['saved_action', 'reward'])
        replay_buffer = []
        def select_action(state):
            state = torch.from_numpy(state).to(device=self.device).float()
            action_rewards = reward_net(state)
            max_action = torch.argmax(action_rewards, dim=0)
            # print(action_rewards.detach().cpu().numpy(), max_action.item())
            return action_rewards, max_action.item()

        def calc_weight(start_time, end_time, feedback_time):
            # DeepTamer uses f_delay as uniform distribution [0.2, 4]
            # \int_{tf-te}^{tf-ts} f_delay(t) dt
            return (end_time - start_time) # * (4-0.2)

        def train(savedActionsWithReward):
            if not savedActionsWithReward: return
            print("training")
            losses = []
            mseLoss = torch.nn.MSELoss()
            for sar in savedActionsWithReward:
                action_rewards, _ = select_action(sar.saved_action.state)
                action_reward_tensor = action_rewards[sar.saved_action.action_index]
                reward = torch.tensor(sar.reward, dtype=torch.float32).to(device=self.device)
                losses.append(mseLoss(action_reward_tensor, reward))
            optimizer.zero_grad()
            loss = torch.stack(losses).to(device=self.device).sum()
            loss.backward()
            optimizer.step()

        def processFeedback(savedActions, buffer):
            feedback_time = self.feedback.feedback_time
            new_data = [SavedActionWithReward(saved_action=sa, reward=self.feedback.feedback_value)
                        for sa in savedActions 
                        if (feedback_time-sa.start_time) > LOWER_TIME_BOUND and (feedback_time-sa.end_time) < UPPER_TIME_BOUND]
            train(new_data)
            buffer.extend(new_data)
            self.feedback = None

        buffer = []
        running_reward = 10
        for i_episode in range(1, 10000):
                state, ep_reward, esp_reward = env.reset(), 0, 0
                savedActions = []
                for t in range(1, 10000):  # Don't infinite loop while learning
                    start_time = time.time()
                    rewards, action = select_action(state)
                    state, reward, done, _ = env.step(action)
                    end_time = time.time()
                    es_reward = rewards[action].item()
                    ep_reward += reward
                    esp_reward += es_reward
                    savedActions.append(SavedAction(state=state, action_index=action, start_time=start_time, end_time=end_time))
                    env.render('human')
                    QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents)
                    if not self.isVisible():
                        break
                    if self.feedback:
                        processFeedback(savedActions, buffer)
                    time.sleep(0.1)
                    if t % 10 == 0 and len(buffer) > args.batch_size:
                        indicies = random.sample(range(len(buffer)), args.batch_size)
                        mini_batch = [buffer[i] for i in indicies]
                        train(mini_batch)
                    print("Reward: %d, es_reward: %d, ep_reward: %d, esp_reward: %d" % (reward, es_reward, ep_reward, esp_reward))
                    if done:
                        if t < 190:
                            self.feedback = HumanFeedback(feedback_time=time.time(), feedback_value=-1)
                            processFeedback(savedActions, buffer)
                        break   
                if not self.isVisible():
                        break
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                print("Running reward %d"% running_reward) 
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()