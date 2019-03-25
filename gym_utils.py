import gym

class UIRenderer(gym.Wrapper):
    def __init__(self, env, viewer):
        super(UIRenderer, self).__init__(env)
        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            raise gym.error.Error("Cannot ui decorate env without rgb_array renderer.")
        self.viewer = viewer
        self.recoding_enabled = False
        self.acc_reward = 0

    def step(self, action):
        res = self.env.step(action)
        reward = res[1]
        self.acc_reward += reward
        self.viewer.update_info(action, reward, self.acc_reward)
        return res
    
    def reset(self, **kwargs):
        self.env.enabled = self.recoding_enabled
        self.acc_reward = 0
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


def make_env(env_name, viewer, alg_name="random_alg", record=False):
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, "/tmp/{}_{}_videos".format(env_name, alg_name), video_callable=lambda x:True)
    env = UIRenderer(env, viewer)
    env.recoding_enabled = record == True
    return env

def toggle_recording(env_object):
    print("Setting recording to: {}".format( not env_object.env.enabled))
    env_object.recoding_enabled = not env_object.env.enabled