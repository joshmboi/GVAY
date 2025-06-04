import random
from collections import deque
import numpy as np

import consts as consts


class ReplayBuffer:
    def __init__(self, cap, window=consts.WINDOW, fpa=consts.FPA):
        self.cap = cap
        self.cur_steps = 0

        self.window = window
        self.fpa = fpa

        # indices of starts of episodes
        self.inds = []

        # doublely-ended queues for automatic cap size
        self.obs = deque()
        self.ac_idxs = deque()
        self.ac_poses = deque()
        self.rews = deque()
        self.dones = deque()

        self.rolls = deque()

    def add(self, obs, ac_idx, ac_pos, rew, done):
        # add index if beginning of episode (observation length of window)
        if len(obs) == self.window:
            self.inds.append(len(self.dones))

        # append each observation
        for ob in obs:
            self.obs.append(ob)

        # append the actions, rewards and dones
        self.ac_idxs.append(ac_idx)
        self.ac_poses.append(ac_pos)
        self.rews.append(rew)
        self.dones.append(done)

        if len(self.ac_idxs) > self.cap:
            # reduce index of all indices
            for i in range(len(self.inds)):
                self.inds[i] -= 1

            # check if finished episode
            if self.dones[0]:
                self.inds.popleft()

                for _ in range(self.window):
                    self.obs.popleft()
            else:
                for _ in range(self.window - self.fpa):
                    self.obs.popleft()

            # pop to keep size
            self.ac_idxs.popleft()
            self.ac_poses.popleft()
            self.rews.popleft()
            self.dones.popleft()

    def reassign_rew(self, rew):
        self.rews[-1] = rew

    def reassign_done(self):
        self.dones[-1] = True

    def num_epis(self, idx):
        i = 0
        while i < len(self.inds) and idx >= self.inds[i]:
            i += 1
        return i - 1

    def sample(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            # random index for sampling
            idx = random.randint(0, len(self.ac_idxs) - 1)

            extra_idx = self.num_epis(idx) * (self.window - self.fpa)

            ob_start = extra_idx + idx * self.fpa
            nob_start = extra_idx + (idx + 1) * self.fpa

            ob = []
            for i in range(ob_start, ob_start + self.window):
                ob.append(self.obs[i])
            ac_idx = self.ac_idxs[idx]
            ac_pos = self.ac_poses[idx]
            rew = self.rews[idx]
            done = self.dones[idx]

            if nob_start + self.window > len(self.obs):
                nob = [np.zeros_like(ob[0]) for i in range(self.window)]
            else:
                nob = []
                for i in range(nob_start, nob_start + self.window):
                    nob.append(self.obs[i])

            batch.append((ob, ac_idx, ac_pos, rew, nob, done))

        # convert to friendly format
        obs, ac_idxs, ac_poses, rews, nobs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(ac_idxs),
            np.array(ac_poses),
            np.array(rews).astype(np.float32),
            np.array(nobs),
            np.array(dones)
        )

    def __len__(self):
        return max(len(self.ac_idxs) - 1, 0)
