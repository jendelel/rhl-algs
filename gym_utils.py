import gym

class UIRenderer(gym.Wrapper):
    def __init__(self, env, viewer):
        super(UIRenderer, self).__init__(env)
        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            raise gym.error.Error("Cannot ui decorate env without rgb_array renderer.")
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


def make_env(env_name, viewer, record=False):
    env = gym.make(env_name)
    if record:
        env = Monitor(env, "/tmp/tamer_cartpole_videos")
    return UIRenderer(env, viewer)
