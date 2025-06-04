from collections import deque

import consts as consts

from game import Game
from policy import Policy
from replaybuffer import ReplayBuffer
from logger import Logger


class Trainer:
    def __init__(self):
        self.tot_steps = 0
        self.max_steps = 1000000
        self.critic_updates = 0
        self.rebuff = ReplayBuffer(consts.REBUFF_SIZE)

        self.logger = Logger(name="sac")
        self.iters = 0
        self.train_step = 0
        self.eval_step = 0

        self.p_pol = Policy(player=True)
        self.e_pol = Policy(player=False)
        self.e_pol.update_policy(self.p_pol)

        # init game
        self.ac_mask = [0, 0, 1, 0, 0]
        self.game = Game(True, ac_mask=self.ac_mask)

    def sim_roll(self, max_ep_len=1080, render=False):
        # initialize steps and frames
        steps = 1
        frames = []

        # get initial game state
        ob = self.game.reset()

        # initialize capturing lists
        p_obs = []
        e_obs = []

        while True:
            # add observations
            p_obs.append(ob["player"])
            e_obs.append(ob["enemy"])

            if render:
                frames.append(self.game.player_screen())

            p_ac, e_ac = None, None

            # get action and take it if acting frame
            if steps % consts.FPA == 0 and steps >= consts.WINDOW:
                # poll policy
                p_ac = self.p_pol.get_action(p_obs[-1], self.ac_mask)
                e_ac = self.e_pol.get_action(e_obs[-1], self.ac_mask)

            else:
                self.p_pol.update_actor_hidden(p_obs[-1])
                self.e_pol.update_actor_hidden(e_obs[-1])

            # move forward step
            ob, rew, done, info = self.game.step(
                playing=False, p_ac=p_ac, e_ac=e_ac
            )

            # end episode if someone dies or hit max episode length
            if steps >= max_ep_len or done:
                break

            # increment steps
            steps += 1

        p_obs.append(ob["player"])
        e_obs.append(ob["enemy"])

        if render:
            self.logger.log_video(frames, f"{self.tot_steps}.mp4")

    def train(self, max_ep_len=1080):
        done = True
        first = True
        accum_p_rew_list = deque(maxlen=10)

        while self.tot_steps < self.max_steps:
            if done:
                # initialize steps and frames
                steps = 1

                # get initial game state
                ob = self.game.reset()

                # initialize reward accumulation
                tot_p_rew, tot_e_rew = 0.0, 0.0

                # reset total rewards
                accum_p_rew = 0.0
                if len(accum_p_rew_list) > 0:

                    self.logger.log_scalar(
                        "Eval/Mean Return",
                        sum(accum_p_rew_list) / len(accum_p_rew_list),
                        self.tot_steps
                    )
                    print(f"Eval/Mean Return: {sum(accum_p_rew_list) / len(accum_p_rew_list)}")

                # reset done and first
                done = False
                first = True

                # initialize capturing lists
                p_obs = deque(maxlen=consts.WINDOW)
                e_obs = deque(maxlen=consts.WINDOW)

            # add observations
            p_obs.append(ob["player"])
            e_obs.append(ob["enemy"])

            p_ac, e_ac = None, None

            # get action and take it if acting frame
            if steps % consts.FPA == 0 and steps >= consts.WINDOW:
                # poll policy
                p_ac = self.p_pol.get_action(p_obs[-1], self.ac_mask)
                e_ac = self.e_pol.get_action(e_obs[-1], self.ac_mask)

            else:
                self.p_pol.update_actor_hidden(p_obs[-1])
                self.e_pol.update_actor_hidden(e_obs[-1])

            # move forward step
            ob, rew, done, info = self.game.step(
                playing=False, p_ac=p_ac, e_ac=e_ac
            )

            # only when action taking turn
            if p_ac is not None and e_ac is not None:
                # add to replay buffer
                if first:
                    self.rebuff.add(
                        list(p_obs), p_ac[0], p_ac[1], tot_p_rew, done
                    )
                    first = False
                else:
                    self.rebuff.add(
                        list(p_obs)[-consts.FPA:], p_ac[0],
                        p_ac[1], tot_p_rew, done
                    )

                # TODO: enemy transitions too??

                # rewards assigning to actions
                if steps != consts.WINDOW:
                    # last rewards and reset
                    self.rebuff.reassign_rew(tot_p_rew)

                tot_p_rew, tot_e_rew = 0.0, 0.0

                tot_p_rew += rew["player"]
                tot_e_rew += rew["enemy"]
                accum_p_rew += rew["player"]
            else:
                # accumulate rewards
                tot_p_rew += rew["player"]
                tot_e_rew += rew["enemy"]
                accum_p_rew += rew["player"]

            # end episode if someone dies or hit max episode length
            if steps >= max_ep_len or done:
                # add assign termination credit for Bellman updates
                self.rebuff.reassign_done()

                # add rewards for last action
                self.rebuff.reassign_rew(tot_p_rew)

                # add accumulated rewards
                accum_p_rew_list.append(accum_p_rew)

                # set done
                done = True

            if (
                steps % consts.FPA == 0 and
                len(self.rebuff) >= consts.BATCH_SIZE * 10
            ):
                batch = self.rebuff.sample(consts.BATCH_SIZE)
                c_metrics = self.p_pol.update_critic(batch, self.ac_mask)

                self.logger.log_scalar(
                    "Critic Loss",
                    c_metrics["critic_loss"],
                    self.tot_steps
                )
                self.logger.log_scalar(
                    "Mean Q Value",
                    c_metrics["q_values"],
                    self.tot_steps
                )

                self.critic_updates += 1

                if (
                    self.critic_updates % 2 == 0 and
                    self.tot_steps > consts.CRITIC_ONLY
                ):
                    a_metrics = self.p_pol.update_actor(batch, self.ac_mask)

                    self.logger.log_scalar(
                        "Actor Loss",
                        a_metrics["actor_loss"],
                        self.tot_steps
                    )

                    self.logger.log_scalar(
                        "Entropy",
                        a_metrics["entropy"],
                        self.tot_steps
                    )

                    self.logger.log_scalar(
                        "Alpha Value",
                        a_metrics["alpha"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Alpha Loss",
                        a_metrics["alpha_loss"],
                        self.tot_steps
                    )

                    self.logger.log_scalar(
                        "Position/X Mean",
                        a_metrics["means_x"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Position/X Std",
                        a_metrics["stds_x"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Position/Y Mean",
                        a_metrics["means_y"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Position/Y Std",
                        a_metrics["stds_y"],
                        self.tot_steps
                    )

            if self.tot_steps % consts.ITERS_PER_EVAL == 0:
                print("Evaluating...")
                self.sim_roll(max_ep_len, render=True)

            if self.tot_steps % consts.SAVE_EVERY == 0:
                print("Saving...")
                self.logger.save_models(self.p_pol, self.e_pol, self.tot_steps)
                print()

            if self.tot_steps % consts.ENEMY_UPDATE == 0 and self.tot_steps != 0:
                print("Updating Enemy...")
                self.e_pol.update_policy(self.p_pol)
                print()

            # increment steps
            steps += 1
            self.tot_steps += 1


trainer = Trainer()
trainer.train()
