import functools, argparse
import random_alg
from ui import MainWindow, create_app
import time
from gym_utils import make_env, toggle_recording


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

    env = make_env(args.env, window.viewer, alg_name=args.alg, record=window.recordCheck.isChecked())
    window.recordCheck.stateChanged.connect(functools.partial(toggle_recording, env_object=env))
    print(args)
    env.seed(args.seed)
    window.viewer.start_time = time.time()
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