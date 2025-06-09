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

    def pretrain(self, max_ep_len=1080, timestamp=None, last_iter=0):
        # init ac_mask and game
        ac_mask = [0, 0, 1, 0, 0]
        e_ac_mask = [1, 0, 0, 0, 0]
        logger = Logger(name="sac", timestamp=timestamp)

        game = Game(training=True)
        p_ptpol = PTPolicy(player=True, training=True)
        e_ptpol = PTPolicy(player=False, training=False)
        e_ptpol.update_policy(p_ptpol)

        rebuff = ReplayBuffer(cap=consts.REBUFF_SIZE, pretrain=True)

        if timestamp is not None:
            print(f"Loading from {timestamp} at {last_iter}...")
            print()
            logger.load_models(p_ptpol, e_ptpol, iter=last_iter, pretrain=True)
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
                p_ac = p_ptpol.get_action(state["player"], ac_mask)
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

                # get cnnlstm metrics
                cnnlstm_metrics = p_ptpol.update_cnnlstm(batch)

                logger.log_scalar(
                    "CNNLSTM/Loss",
                    cnnlstm_metrics["cnnlstm_loss"],
                    self.tot_steps
                )

                c_metrics = p_ptpol.update_critic(batch, ac_mask)

                logger.log_scalar(
                    "Critic/Loss 1",
                    c_metrics["c1_loss"],
                    self.tot_steps
                )
                logger.log_scalar(
                    "Critic/Loss 2",
                    c_metrics["c2_loss"],
                    self.tot_steps
                )
                logger.log_scalar(
                    "Critic/Q Value 1",
                    c_metrics["q1_vals"],
                    self.tot_steps
                )
                logger.log_scalar(
                    "Critic/Q Value 2",
                    c_metrics["q2_vals"],
                    self.tot_steps
                )

                self.critic_updates += 1

                if (
                    self.critic_updates % 2 == 0 and
                    self.tot_steps > consts.CRITIC_ONLY
                ):
                    a_metrics = p_ptpol.update_actor(batch, ac_mask)

                    logger.log_scalar(
                        "Actor/Loss",
                        a_metrics["actor_loss"],
                        self.tot_steps
                    )

                    logger.log_scalar(
                        "Actor/Type Entropy",
                        a_metrics["type_entropy"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Actor/Pos Entropy",
                        a_metrics["pos_entropy"],
                        self.tot_steps
                    )

                    logger.log_scalar(
                        "Actor/Type Alpha Value",
                        a_metrics["type_alpha"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Actor/Pos Alpha Value",
                        a_metrics["pos_alpha"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Actor/Type Alpha Loss",
                        a_metrics["type_alpha_loss"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Actor/Pos Alpha Loss",
                        a_metrics["pos_alpha_loss"],
                        self.tot_steps
                    )

                    logger.log_scalar(
                        "Position/X Mean",
                        a_metrics["means_x"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Position/X Std",
                        a_metrics["stds_x"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Position/Y Mean",
                        a_metrics["means_y"],
                        self.tot_steps
                    )
                    logger.log_scalar(
                        "Position/Y Std",
                        a_metrics["stds_y"],
                        self.tot_steps
                    )

            if self.tot_steps % consts.ITERS_PER_EVAL == 0:
                next_eval = True

            if (
                self.tot_steps % consts.SAVE_EVERY == 0 and
                self.tot_steps != last_iter
            ):
                print("Saving...")
                logger.save_models(
                    p_ptpol, e_ptpol, self.tot_steps, pretrain=True
                )
                logger.save_rebuff(rebuff, self.tot_steps, pretrain=True)
                print()

            if (
                self.tot_steps % consts.ENEMY_UPDATE == 0 and
                self.tot_steps != 0
            ):
                print("Updating Enemy...")
                e_ptpol.update_policy(p_ptpol)
                print()

            # increment steps
            steps += 1
            self.tot_steps += 1

    def train(self, max_ep_len=1080):
        # init game
        self.ac_mask = [0, 0, 1, 0, 0]

        # init game and ac_mask ([no-move, move, q, w, e])
        ac_mask = [0, 0, 1, 0, 0, ]
        game = Game(training=True, ac_mask=ac_mask)
        rebuff = ReplayBuffer(cap=consts.REBUFF_SIZE, pretrain=False)

        self.p_pol = Policy(player=True, training=True)
        self.e_pol = Policy(player=False, training=False)
        self.e_pol.update_policy(self.p_pol)

        # done flag for completion of episode
        done = True

        # reward over episode
        accum_p_rew = None

        # rendering flags
        render = True
        next_render = True

        # whether to swap player and enemy position
        swap = False
        while self.tot_steps < self.max_steps:
            if done:
                # initialize steps and frames
                steps = 1

                # get initial game state
                ob = game.reset(swap=(eval or swap))

                # initialize reward accumulation
                tot_p_rew, tot_e_rew = 0.0, 0.0

                # add episode return
                if accum_p_rew is not None:
                    self.logger.log_scalar(
                        "Eval Return",
                        accum_p_rew,
                        self.tot_steps
                    )
                    print(
                        f"Eval Return: {accum_p_rew}"
                    )
                # reset total rewards
                accum_p_rew = 0.0

                # frames for rendering
                frames = []

                # first action flag
                first_ac = True

                # reset done and first
                done = False

                # initialize sequence capturing lists
                p_obs = deque(maxlen=consts.WINDOW)
                e_obs = deque(maxlen=consts.WINDOW)

            # add observations
            p_obs.append(ob["player"])
            e_obs.append(ob["enemy"])

            if render:
                frames.append(game.player_screen())

            p_ac, e_ac = None, None

            # get action and take it if acting frame
            if steps % consts.FPA == 0 and steps >= consts.WINDOW:
                # poll policy
                p_ac = self.p_pol.get_action(p_obs[-1], self.ac_mask)
                e_ac = self.e_pol.get_action(e_obs[-1], [1, 0, 0, 0, 0])

            else:
                self.p_pol.update_actor_hidden(p_obs[-1])
                self.e_pol.update_actor_hidden(e_obs[-1])

            # move forward step
            ob, rew, done, info = game.step(
                playing=False, p_ac=p_ac, e_ac=e_ac
            )

            # only when action taking turn
            if p_ac is not None and e_ac is not None:
                # add to replay buffer
                if first_ac:
                    rebuff.add(
                        list(p_obs), p_ac[0], p_ac[1], tot_p_rew, done
                    )
                    first_ac = False
                else:
                    rebuff.add(
                        list(p_obs)[-consts.FPA:], p_ac[0],
                        p_ac[1], tot_p_rew, done
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

                # set done
                done = True

                # set render
                if render:
                    self.logger.log_video(frames, f"{self.tot_steps}.mp4")
                    render = False

                if next_render:
                    render = True
                    next_render = False

                # swap positions
                swap = not swap

            if (
                steps % consts.FPA == 0 and
                len(rebuff) >= consts.BATCH_SIZE * 10
            ):
                batch = rebuff.sample(consts.BATCH_SIZE)
                c_metrics = self.p_pol.update_critic(batch, self.ac_mask)

                self.logger.log_scalar(
                    "Critic/Loss",
                    c_metrics["critic_loss"],
                    self.tot_steps
                )
                self.logger.log_scalar(
                    "Critic/Q Value",
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
                        "Actor/Loss",
                        a_metrics["actor_loss"],
                        self.tot_steps
                    )

                    self.logger.log_scalar(
                        "Actor/Entropy",
                        a_metrics["entropy"],
                        self.tot_steps
                    )

                    self.logger.log_scalar(
                        "Actor/Type Alpha Value",
                        a_metrics["type_alpha"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Actor/Type Alpha Value",
                        a_metrics["pos_alpha"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Actor/Type Alpha Loss",
                        a_metrics["type_alpha_loss"],
                        self.tot_steps
                    )
                    self.logger.log_scalar(
                        "Actor/Pos Alpha Loss",
                        a_metrics["pos_alpha_loss"],
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
                next_render = True

            if self.tot_steps % consts.SAVE_EVERY == 0:
                print("Saving...")
                self.logger.save_models(self.p_pol, self.e_pol, self.tot_steps)
                print()

            if (
                self.tot_steps % consts.ENEMY_UPDATE == 0 and
                self.tot_steps != 0
            ):
                print("Updating Enemy...")
                self.e_pol.update_policy(self.p_pol)
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
trainer.pretrain(1080)
