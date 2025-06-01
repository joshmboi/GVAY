class Roll:
    def __init__(self, obs, ac_idxs, ac_poses, rews, dones):
        self.observations = obs
        self.action_indexes = ac_idxs
        self.action_positions = ac_poses
        self.rewards = rews
        self.dones = dones

        self.num_steps = len(rews)
