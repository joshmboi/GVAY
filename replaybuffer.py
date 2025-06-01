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

        self.rolls = deque()

    def add_rolls(self, rolls):
        # add all rollouts
        for roll in rolls:
            self.rolls.append(roll)
            self.cur_steps += roll.num_steps

        # take away rollouts until fractional episode over cap
        while self.rolls[0].num_steps <= self.cur_steps - self.cap:
            roll = self.rolls.popleft()
            self.cur_steps -= roll.num_steps

    def sample(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            roll = random.choice(self.rolls)

            # choose action step
            step = random.choice(range(roll.num_steps))

            ob_end = (step + 2) * self.fpa
            nob_end = (step + 3) * self.fpa

            # skip transitions that go beyond recorded
            if nob_end > len(roll.observations):
                continue

            # get ob, ac, rew, and done
            ob = roll.observations[ob_end - self.window:ob_end]
            ac_idx = roll.action_indexes[step]
            ac_pos = roll.action_positions[step]
            rew = roll.rewards[step]
            done = roll.dones[step]

            # tensor of zeros for nob if terminal action
            if done:
                nob = [np.zeros_like(ob[0]) for i in range(self.window)]
            else:
                nob = roll.observations[nob_end - self.window:nob_end]

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
