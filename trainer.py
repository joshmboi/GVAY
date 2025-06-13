from collections import deque

import consts as consts

from game import Game
from policy import Policy
from pt_pol import PTPolicy
from replaybuffer import ReplayBuffer
from logger import Logger


class Trainer:
    def __init__(self):
        self.tot_steps = 0
        self.max_steps = consts.TOTAL_ITERS
        self.critic_updates = 0

        self.iters = 0
        self.train_step = 0
        self.eval_step = 0

    def train(self, max_ep_len, use_entropy, timestamp=None, last_iter=0):
        # init ac_mask and game
        p_ac_mask = [0, 0, 1, 0, 0]
        e_ac_mask = [1, 0, 0, 0, 0]
        logger = Logger(name="sac", timestamp=timestamp)

        game = Game(training=True)
        p_ptpol = PTPolicy(player=True, training=True, use_entropy=False)
        e_ptpol = PTPolicy(player=False, training=False, use_entropy=False)
        e_ptpol.update_policy(p_ptpol, use_entropy)

        rebuff = ReplayBuffer(cap=consts.REBUFF_SIZE, pretrain=True)

        if timestamp is not None:
            print(f"Loading from {timestamp} at {last_iter}...")
            print()
            logger.load_models(
                p_ptpol, e_ptpol, iter=last_iter,
                pretrain=True, use_entropy=use_entropy
            )
            self.tot_steps = last_iter
            self.iters = last_iter
            rebuff = logger.load_rebuff(iter=last_iter, pretrain=True)

        # done and eval flags
        done = True
        eval = True
        next_eval = False

        # reward accumulation
        accum_p_rew = None
        while self.tot_steps < self.max_steps:
            if done:
                # init steps and frames
                steps = 1
                frames = []

                # init game ob
                ob = game.reset()

                # init reward accumulation
                tot_p_rew, tot_e_rew = 0.0, 0.0

                # add episode return
                if accum_p_rew is not None:
                    logger.log_scalar(
                        "Eval Return",
                        accum_p_rew,
                        self.tot_steps
                    )
                    print(
                        f"Eval Return: {accum_p_rew}"
                    )
                # reset tot rewards
                accum_p_rew = 0.0

                # first action flag
                first_ac = True

                # reset done and first
                done = False

                # initialize sequence capturing lists and state
                p_obs = deque(maxlen=consts.WINDOW)
                e_obs = deque(maxlen=consts.WINDOW)
                state = None

            # add obs
            p_obs.append(ob["player"])
            e_obs.append(ob["enemy"])

            if eval:
                frames.append(game.player_screen())

            p_ac, e_ac = None, None

            # get action and take it if acting frame
            if steps % consts.FPA == 0 and steps >= consts.WINDOW:
                # poll policy
                p_ac = p_ptpol.get_action(state["player"], p_ac_mask)
                e_ac = e_ptpol.get_action(state["enemy"], e_ac_mask)

            else:
                p_ptpol.update_cnnlstm_hidden(p_obs[-1])

            # move forward step
            ob, state, rew, done, info = game.step(
                state_train=True, playing=False, p_ac=p_ac, e_ac=e_ac
            )

            # only when action taking turn
            if p_ac is not None and e_ac is not None:
                # add to replay buffer
                if first_ac:
                    rebuff.add(
                        list(p_obs), state["player"],
                        p_ac[0], p_ac[1], tot_p_rew, done
                    )
                    first_ac = False
                else:
                    rebuff.add(
                        list(p_obs)[-consts.FPA:], state["player"],
                        p_ac[0], p_ac[1], tot_p_rew, done
                    )

                # TODO: enemy transitions too??

                # rewards assigning to actions
                if steps != consts.WINDOW:
                    # last rewards and reset
                    rebuff.reassign_rew(tot_p_rew)

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
                rebuff.reassign_done()

                # add rewards for last action
                rebuff.reassign_rew(tot_p_rew)

                # set done, eval, and next_eval
                done = True
                if eval:
                    logger.log_video(frames, f"{self.tot_steps + 1}.mp4")
                    eval = False

                if next_eval:
                    print("Evaluating...")
                    eval = True
                    next_eval = False

            if (
                steps % consts.FPA == 0 and
                len(rebuff) >= consts.BATCH_SIZE * 10
            ):
                batch = rebuff.sample(consts.BATCH_SIZE)

                # log cnnlstm metrics
                cnnlstm_metrics = p_ptpol.update_cnnlstm(batch)
                logger.log_cnnlstm_metrics(cnnlstm_metrics, self.tot_steps)

                # log critic metrics
                c_metrics = p_ptpol.update_critic(batch, p_ac_mask)
                logger.log_critic_metrics(c_metrics, self.tot_steps)

                self.critic_updates += 1

                if (
                    self.critic_updates % 2 == 0 and
                    self.tot_steps > consts.CRITIC_ONLY
                ):
                    # log actor metrics
                    a_metrics = p_ptpol.update_actor(batch, p_ac_mask)
                    logger.log_actor_metrics(
                        a_metrics, self.tot_steps, use_entropy
                    )

            if self.tot_steps % consts.ITERS_PER_EVAL == 0:
                next_eval = True

            if (
                self.tot_steps % consts.SAVE_EVERY == 0 and
                self.tot_steps != last_iter
            ):
                print("Saving...")
                logger.save_models(
                    p_ptpol, e_ptpol, self.tot_steps,
                    pretrain=True, use_entropy=use_entropy
                )
                logger.save_rebuff(rebuff, self.tot_steps, pretrain=True)
                print()

            if (
                self.tot_steps % consts.ENEMY_UPDATE == 0 and
                self.tot_steps != 0
            ):
                print("Updating Enemy...")
                e_ptpol.update_policy(p_ptpol, use_entropy)
                print()

            # increment steps
            steps += 1
            self.tot_steps += 1

    def sim(self):
        game = Game(training=False)
        ob = game.reset()

        p_pol = Policy(player=False, training=False)
        # p_pol.load_policy("./runs/pol/models/model_40000.pth")

        done = False
        while not done:
            # if game.frame % consts.FPA == 0:
                # p_ac = p_pol.get_action(ob["player"], [0, 0, 1, 0, 0])
            ob, _, _, done, info = game.step(
                playing=True,  # state_train=True, p_ac=p_ac, e_ac=None
            )
            if done:
                game.game_over()
            game.clock.tick(consts.FPS)

        game.quit()


trainer = Trainer()
trainer.train(1080, False)
# trainer.sim()
