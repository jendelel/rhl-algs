from PyQt5 import QtGui, QtCore, QtWidgets
from collections import namedtuple
import time
import random
import numpy as np
import utils

# Dummy function because random alg does need parameters
def parse_args(parser):
    pass

class ManualControl():
    def __init__(self, window, args, env):
        self.window = window
        self.args = args
        self.env = env

        if window is not None: 
            self.setup_ui(window)
        self.last_action = 0
    
    def setup_ui(self, window):
        @QtCore.pyqtSlot(QtGui.QKeyEvent)
        def keyPressed(event):
            numpad_mod = int(event.modifiers()) & QtCore.Qt.KeypadModifier
            if event.key() == QtCore.Qt.Key_0:
                self.last_action = 0
            elif event.key() == QtCore.Qt.Key_1:
                self.last_action = 1
            elif event.key() == QtCore.Qt.Key_2:
                self.last_action = 2
            elif event.key() == QtCore.Qt.Key_3:
                self.last_action = 3
            elif event.key() == QtCore.Qt.Key_4:
                self.last_action = 4
            elif event.key() == QtCore.Qt.Key_5:
                self.last_action = 5
            else:
                print("ERROR: Unknown key: ", event)
        window.keyPressedSignal.connect(keyPressed)

    def train(self):
        buffer = []
        running_reward = 10
        for i_episode in range(1, 10000):
                state, ep_reward = self.env.reset(), 0
                savedActions = []
                for t in range(1, 10000):  # Don't infinite loop while learning
                    action = np.clip(self.last_action, 0, int(self.env.action_space.n)-1)
                    state, reward, done, _ = self.env.step(action)
                    ep_reward += reward
                    self.window.render(self.env)
                    if not self.window.isVisible():
                        break
                    time.sleep(self.args.render_sleep)
                    print("Action: %d, Reward: %d, ep_reward: %d" % (action, reward, ep_reward))
                    if done:
                        break
                if not self.window.isVisible():
                        break

def start(window, args, env):
    alg = ManualControl(window, args, env)
    alg.train()
    env.close()