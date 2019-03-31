import time


# Dummy function because random alg does need parameters
def parse_args(parser):
    pass


def start(window, args, env):

    def select_action(state):
        return env.action_space.sample()

    for i_episode in range(1, 10):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            window.render(env)
            time.sleep(window.renderSpin.value())
            if not window.isVisible():
                break
            ep_reward += reward
            print("Action: %d, Reward: %d, ep_reward: %d" % (action, reward, ep_reward))
            if done:
                break
        if not window.isVisible():
            break
    env.close()
