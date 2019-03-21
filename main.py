import functools, argparse
import random_alg
from ui import MainWindow, create_app

from gym_utils import make_env


running = False
def startRl(window):
    global running
    if running: return
    running = True
    
    parser = argparse.ArgumentParser(description='Reinforcement human learning')
    parser.add_argument('--alg', type=str, default="random_alg",
                                    help='Name of RL algorithm.')
    parser.add_argument('--env', type=str, default="CartPole-v0",
                                    help='Name of Gym environment.')
    parser.add_argument('--seed', type=int, default=543, 
                            help='random seed (default: 543)')
    alg, args = window.loadAlg(parser)

    env = make_env(args.env, window.viewer, record=False)
    print(args)
    env.seed(args.seed)

    alg.start(window, args, env)
    running = True

def main():
    app = create_app()
    window = MainWindow()
    window.startBut.clicked.connect(functools.partial(startRl, window=window))
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()