import random
from collections import deque, namedtuple
import numpy as np

import consts as consts


PTBatch = namedtuple(
    "PTBatch", "obs states ac_idxs ac_poses rews n_states dones"
)
Batch = namedtuple("Batch", "obs ac_idxs ac_poses rews nobs dones")


class ReplayBuffer:
    def __init__(
            self, cap, pretrain=False, window=consts.WINDOW, fpa=consts.FPA
    ):
        self.cap = cap
        self.cur_steps = 0
        self.pretrain = pretrain

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

        if self.pretrain:
            self.states = deque()

        self.rolls = deque()

    def add(self, obs, state, ac_idx, ac_pos, rew, done):
        # add index if beginning of episode (observation length of window)
        if len(obs) == self.window:
            self.inds.append(len(self.dones))

        # append each observation
        for ob in obs:
            self.obs.append(ob)

        # append the states, actions, rewards and dones
        self.ac_idxs.append(ac_idx)
        self.ac_poses.append(ac_pos)
        self.rews.append(rew)
        self.dones.append(done)

        if self.pretrain:
            self.states.append(state)

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

            if self.pretrain:
                self.states.popleft()

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
            # random index for sampling (inclusive)
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

            if self.pretrain:
                state = self.states[idx]
                if idx + 1 >= len(self.states):
                    n_state = np.zeros_like(state)
                else:
                    n_state = self.states[idx + 1]

                batch.append((ob, state, ac_idx, ac_pos, rew, n_state, done))
            else:
                if nob_start + self.window >= len(self.obs):
                    nob = [np.zeros_like(ob[0]) for i in range(self.window)]
                else:
                    nob = []
                    for i in range(nob_start, nob_start + self.window):
                        nob.append(self.obs[i])

                batch.append((ob, ac_idx, ac_pos, rew, nob, done))

        # convert to friendly format
        if self.pretrain:
            obs, states, ac_idxs, ac_poses, rews, n_states, dones = zip(*batch)
            return PTBatch(
                np.array(obs),
                np.array(states),
                np.array(ac_idxs),
                np.array(ac_poses),
                np.array(rews).astype(np.float32),
                np.array(n_states),
                np.array(dones)
            )
        obs, ac_idxs, ac_poses, rews, nobs, dones = zip(*batch)
        return Batch(
            np.array(obs),
            np.array(ac_idxs),
            np.array(ac_poses),
            np.array(rews).astype(np.float32),
            np.array(nobs),
            np.array(dones)
        )

    def __len__(self):
        return max(len(self.ac_idxs) - 1, 0)
