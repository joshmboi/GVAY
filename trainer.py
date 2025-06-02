import consts as consts

from game import Game
from rollout import Roll
from replaybuffer import ReplayBuffer
from logger import Logger


class Trainer:
    def __init__(self):
        self.ac_mask = [1, 1, 0, 1, 0]
        self.game = Game(True, ac_mask=self.ac_mask)

        self.tot_steps = 0
        self.rebuff = ReplayBuffer(consts.REBUFF_SIZE)

        self.logger = Logger(name="sac")
        self.iters = 0
        self.train_step = 0
        self.eval_step = 0

    def sim_roll(self, max_ep_len=1800, eps=0, render=False):
        # initialize steps and frames
        steps = 0
        frames = []

        # get initial game state
        ob = self.game.reset()

        # initialize reward accumulation
        tot_p_rew, tot_e_rew = 0.0, 0.0

        # initialize capturing lists
        p_obs, p_ac_idxs, p_ac_poses = [], [], []
        p_rews, p_dones = [], []
        e_obs, e_ac_idxs, e_ac_poses = [], [], []
        e_rews, e_dones = [], []

        while True:
            # add observations
            p_obs.append(ob["player"])
            e_obs.append(ob["enemy"])

            if render:
                frames.append(self.game.player_screen())

            ob, ac_idx, ac_pos, rew, done, info = self.game.step()

            if ac_idx is not None or ac_pos is not None:
                # add actions
                p_ac_idxs.append(ac_idx["player"])
                e_ac_idxs.append(ac_idx["enemy"])
                p_ac_poses.append(ac_pos["player"])
                e_ac_poses.append(ac_pos["enemy"])

                if len(p_dones) != 0:
                    # add rewards and reset
                    p_rews.append(tot_p_rew)
                    e_rews.append(tot_e_rew)
                tot_p_rew, tot_e_rew = 0.0, 0.0

                tot_p_rew += rew["player"]
                tot_e_rew += rew["enemy"]

                # add done flag
                p_dones.append(done)
                e_dones.append(done)
            else:
                # accumulate rewards
                tot_p_rew += rew["player"]
                tot_e_rew += rew["enemy"]

            # end episode if someone dies or hit max episode length
            if steps >= max_ep_len or done:
                if steps % consts.FPA != 0:
                    # add assign termination credit to last action
                    p_dones[-1] = True
                    e_dones[-1] = True
                break

            # increment steps
            steps += 1

        # add last observation
        p_obs.append(ob["player"])
        e_obs.append(ob["enemy"])

        # add rewards for last action
        p_rews.append(tot_p_rew)
        e_rews.append(tot_p_rew)

        # return in friendly format
        p_roll = Roll(p_obs, p_ac_idxs, p_ac_poses, p_rews, p_dones)
        e_roll = Roll(e_obs, e_ac_idxs, e_ac_poses, e_rews, e_dones)

        if render:
            self.logger.log_video(frames, f"{self.tot_steps}.mp4")

        return p_roll, e_roll, steps

    def sim_n_rolls(self, num_rolls, max_ep_len, render=False):
        # track mean rewards over eval rollouts
        rews = []
        for i in range(num_rolls):
            if i > 0:
                render = False
            p_roll, _, _ = self.sim_roll(max_ep_len, 0, render)

            rews.append(sum(p_roll.rewards))

        # log return stats
        self.logger.log_scalar(
            "Sim/Mean Return:",
            sum(rews) / len(rews),
            self.eval_step
        )
        self.logger.log_scalar(
            "Sim/Max Return:",
            max(rews),
            self.eval_step
        )
        self.logger.log_scalar(
            "Sim/Min Return:",
            min(rews),
            self.eval_step
        )

        # print out return stats
        print(f"Sim/Mean Return: {sum(rews) / len(rews)}")
        print(f"Sim/Max Return: {max(rews)}")
        print(f"Sim/Min Return: {min(rews)}")
        print()

        self.eval_step += 1

    def sim_rolls(self, batch_min_steps, max_ep_len, render=False):
        # initialize
        iter_steps = 0
        rolls = []
        rews = []

        while iter_steps < batch_min_steps:
            # simulate player and enemy rollouts for a game
            p_roll, e_roll, steps = self.sim_roll(max_ep_len, render)
            iter_steps += steps * 2

            # add to rollouts
            rolls.append(p_roll)
            rolls.append(e_roll)

            # add rewards
            rews.append(sum(p_roll.rewards))

        # log return stats
        self.logger.log_scalar(
            "Eval/Mean Return:",
            sum(rews) / len(rews),
            self.eval_step
        )
        self.logger.log_scalar(
            "Eval/Max Return:",
            max(rews),
            self.eval_step
        )
        self.logger.log_scalar(
            "Eval/Min Return:",
            min(rews),
            self.eval_step
        )

        # print out return stats
        print(f"Eval/Mean Return: {sum(rews) / len(rews)}")
        print(f"Eval/Max Return: {max(rews)}")
        print(f"Eval/Min Return: {min(rews)}")
        print()

        return rolls, iter_steps

    def train(self, num_train_steps):
        # initialize metrics tracking
        c_losses, q_vals, a_losses = [], [], []
        alphs, entropies, alph_losses, pos_means, pos_stds = [], [], [], [], []
        c_loss_min, c_loss_max, a_loss_min, a_loss_max = None, None, None, None
        q_max, q_min, entropy_min, entropy_max = None, None, None, None

        for i in range(num_train_steps):
            batch = self.rebuff.sample(consts.BATCH_SIZE)

            c_metrics = self.game.player.policy.update_critic(
                batch, ac_mask=self.ac_mask
            )

            c_losses.append(c_metrics["critic_loss"])
            q_vals.append(c_metrics["q_values"])

            if not c_loss_min:
                c_loss_min = c_metrics["critic_loss"]
                c_loss_max = c_metrics["critic_loss"]
                q_min = c_metrics["q_val_min"]
                q_max = c_metrics["q_val_max"]
            else:
                c_loss_min = min(c_loss_min, c_metrics["critic_loss"])
                c_loss_max = max(c_loss_max, c_metrics["critic_loss"])
                q_min = min(q_min, c_metrics["q_val_min"])
                q_max = max(q_max, c_metrics["q_val_max"])

            if self.iters >= consts.CRITIC_ONLY:
                a_metrics = self.game.player.policy.update_actor(
                    batch, ac_mask=self.ac_mask
                )

                a_losses.append(a_metrics["actor_loss"])
                alphs.append(a_metrics["alpha"])
                entropies.append(a_metrics["entropy"])
                alph_losses.append(a_metrics["alpha_loss"])
                pos_means.append(a_metrics["means"])
                pos_stds.append(a_metrics["stds"])

                if not a_loss_min:
                    a_loss_min = a_metrics["actor_loss"]
                    a_loss_max = a_metrics["actor_loss"]
                    entropy_min = a_metrics["entropy"]
                    entropy_max = a_metrics["entropy"]
                else:
                    a_loss_min = min(a_loss_min, a_metrics["actor_loss"])
                    a_loss_max = max(a_loss_max, a_metrics["actor_loss"])
                    entropy_min = min(entropy_min, a_metrics["entropy"])

        self.logger.log_scalar(
            "Critic/Mean Loss",
            sum(c_losses) / num_train_steps,
            self.train_step
        )
        self.logger.log_scalar(
            "Critic/Highest Loss",
            c_loss_max,
            self.train_step
        )
        self.logger.log_scalar(
            "Critic/Lowest Loss",
            c_loss_min,
            self.train_step
        )

        self.logger.log_scalar(
            "Q/Mean Value",
            sum(q_vals) / num_train_steps,
            self.train_step
        )
        self.logger.log_scalar(
            "Q/Highest Value",
            q_max,
            self.train_step
        )
        self.logger.log_scalar(
            "Q/Lowest Value",
            q_min,
            self.train_step
        )

        if self.iters >= consts.CRITIC_ONLY:
            self.logger.log_scalar(
                "Actor/Mean Loss",
                sum(a_losses) / num_train_steps,
                self.train_step
            )
            self.logger.log_scalar(
                "Actor/Highest Loss",
                a_loss_max,
                self.train_step
            )
            self.logger.log_scalar(
                "Actor/Lowest Loss",
                a_loss_min,
                self.train_step
            )

            self.logger.log_scalar(
                "Entropy/Mean",
                sum(entropies) / num_train_steps,
                self.train_step
            )
            self.logger.log_scalar(
                "Entropy/Highest",
                entropy_max,
                self.train_step
            )
            self.logger.log_scalar(
                "Entropy/Lowest",
                entropy_min,
                self.train_step
            )

            self.logger.log_scalar(
                "Alpha/Mean Value",
                sum(alphs) / num_train_steps,
                self.train_step
            )
            self.logger.log_scalar(
                "Alpha/Mean Loss",
                sum(alph_losses) / num_train_steps,
                self.train_step
            )

            self.logger.log_scalar(
                "Position/Means",
                pos_means,
                self.train_step
            )
            self.logger.log_scalar(
                "Position/Stds",
                pos_stds,
                self.train_step
            )

        print(f"Critic/Mean Loss: {sum(c_losses) / num_train_steps}")
        print(f"Critic/Highest Loss: {c_loss_max}")
        print(f"Critic/Lowest Loss: {c_loss_min}")

        print(f"Q/Mean Value: {sum(q_vals) / num_train_steps}")
        print(f"Q/Highest Value: {q_max}")
        print(f"Q/Lowest Value: {q_min}")

        if self.iters >= consts.CRITIC_ONLY:
            print(f"Actor/Mean Loss: {sum(a_losses) / num_train_steps}")
            print(f"Actor/Highest Loss: {a_loss_max}")
            print(f"Actor/Lowest Loss: {a_loss_min}")

            print(f"Entropy/Mean: {sum(entropies) / num_train_steps}")
            print(f"Entropy/Highest: {entropy_max}")
            print(f"Entropy/Lowest: {entropy_min}")

            print(f"Alpha/Mean Value: {sum(alphs) / num_train_steps}")
            print(f"Alpha/Mean Loss: {sum(alph_losses) / num_train_steps}")

        print()

        self.train_step += 1

    def run_training(
            self, num_iters, num_train_steps, batch_min_steps, max_ep_len
    ):
        self.tot_steps = 0

        for i in range(num_iters):
            print(f"Iteration {i}")
            print(f"Total Steps: {self.tot_steps}")
            print("Collecting Rollouts...")

            # simulate rollouts until minimum batch number of steps
            rolls, iter_steps = self.sim_rolls(batch_min_steps, max_ep_len)
            self.tot_steps += iter_steps

            self.rebuff.add_rolls(rolls)

            print("Running updates...")
            self.train(num_train_steps)

            if i % consts.ITERS_PER_EVAL == 0:
                print("Evaluating...")
                self.sim_n_rolls(5, max_ep_len, render=True)

            if i % consts.SAVE_EVERY == 0:
                print("Saving...")
                self.logger.save_models(self.game.player, self.game.enemy, i)
                print()

            if i % consts.ENEMY_UPDATE == 0 and i != 0:
                print("Updating Enemy...")
                self.game.enemy.policy.update_policy(self.game.player.policy)
                print()

            self.iters += 1


trainer = Trainer()
trainer.run_training(1000, consts.TRAIN_STEPS, 10000, 1800)
# game = Game()
# game.reset()
# game.run()
