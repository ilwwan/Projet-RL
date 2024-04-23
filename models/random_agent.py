class RandomAgent:
    def __init__(self, action_space, *args, **kwargs):
        self.action_space = action_space

    def get_action(self, state, *args, **kwargs):
        return self.action_space.sample()
