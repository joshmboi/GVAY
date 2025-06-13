import numpy as np
import cv2
import math
import os
import pygame

import consts as consts
from agent import Agent


class Game:
    def __init__(
        self, training=False, disp_w=consts.DISP_W, disp_h=consts.DISP_H
    ):
        # set dummy sound output
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        # set whether training
        self.training = training

        # set display size
        self.disp_w = disp_w
        self.disp_h = disp_h

        if self.training:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # init players
        self.player = Agent(self.disp_w // 4, self.disp_h // 2)
        self.enemy = Agent((self.disp_w * 3) // 4, self.disp_h // 2)

        # init player and enemy screens
        self.p_screen = pygame.Surface(
            (self.disp_w, self.disp_h), pygame.SRCALPHA
        )
        self.e_screen = pygame.Surface(
            (self.disp_w, self.disp_h), pygame.SRCALPHA
        )

        # init game
        pygame.init()
        self.disp = pygame.display.set_mode((self.disp_w, self.disp_h))

    def reset(self, swap=False):
        # init clock
        self.clock = pygame.time.Clock()

        # set x and y of player and enemy
        if not swap:
            p_x, p_y = self.disp_w // 4, self.disp_h // 2
            e_x, e_y = (3 * self.disp_w) // 4, self.disp_h // 2
        else:
            p_x, p_y = (3 * self.disp_w) // 4, self.disp_h // 2
            e_x, e_y = self.disp_w // 4, self.disp_h // 2

        # reset players
        self.player.reset(p_x, p_y)
        self.enemy.reset(e_x, e_y)

        # reset frame
        self.frame = 0

        # draw initial screen
        self.draw()

        # get observations (screen pixels)
        self.p_ob = self.agent_ob(self.p_screen)
        self.e_ob = self.agent_ob(self.e_screen)

        ob = {
            "player": self.p_ob,
            "enemy": self.e_ob
        }
        return ob

    def check_damage(self, damager, damagee):
        # determine damage taken
        damage_taken = 0
        # take damage if hit by projectile
        for proj in damager.projs:
            if math.hypot(
                damagee.x - proj.x, damagee.y - proj.y
            ) <= damagee.rad + proj.rad:
                damage_taken += damagee.take_damage(proj.damage())
                damager.projs.remove(proj)

        # take damage from damage zone
        sc = damager.scorch
        if (sc and
            math.hypot(damagee.x - sc.x, damagee.y - sc.y) <=
                damagee.rad + sc.rad):
            damage_taken += damagee.take_damage(sc.damage(self.frame))

        return damage_taken

    def draw_screen(self, screen, player, enemy):
        # clear screen
        screen.fill(consts.BACKGROUND_COLOR)

        # draw player and enemy
        player.draw(screen, consts.PLAYER_PALETTE, self.frame)
        enemy.draw(screen, consts.ENEMY_PALETTE, self.frame)

        # draw ui
        player.draw_ui(screen, self.frame)

    def draw(self):
        # draw player and enemy screen
        self.draw_screen(self.p_screen, self.player, self.enemy)
        self.draw_screen(self.e_screen, self.enemy, self.player)

        if not self.training:
            self.disp.blit(self.p_screen, (0, 0))

            pygame.display.flip()

    def player_screen(self):
        return np.transpose(
            pygame.surfarray.array3d(self.p_screen),
            (1, 0, 2),
        ).copy()

    def agent_ob(self, screen):
        return np.transpose(
            cv2.resize(
                pygame.surfarray.array3d(screen),
                (self.disp_h // 4, self.disp_w // 4),
                interpolation=cv2.INTER_AREA
            ),
            (2, 0, 1),
        ).copy().astype(np.uint8)

    def sq_dist(self, p, agent):
        return (p[0] - agent[0])**2 + (p[1] - agent[1])**2

    def agent_state(self, player, enemy):
        if player.speed == 0:
            p_dx, p_dy = 0, 0
        else:
            p_dx, p_dy = math.cos(player.ang), math.sin(player.ang)

        # init player state
        p_state = [
            player.x / self.disp_w, player.y / self.disp_h,
            p_dx, p_dx,
            player.health / consts.MAX_HEALTH, player.stam / consts.MAX_STAM,
            max(
                0, (1 - (self.frame - player.q_last_cast) / consts.PROJ_COOL)
            ),
            max(
                0, (1 - (self.frame - player.w_last_cast) / consts.SCORCH_COOL)
            ),
            max(
                0, (1 - (self.frame - player.e_last_cast) / consts.SHIELD_COOL)
            ),
        ]

        # add player projectiles
        p_projs = sorted(
                player.projs, key=lambda p: self.sq_dist(
                    (p.x, p.y), (player.x, player.y)
                )
        )
        for i in range(min(1, len(p_projs))):
            p_state.append(p_projs[i].x / self.disp_w)
            p_state.append(p_projs[i].y / self.disp_h)
            p_state.append(math.cos(p_projs[i].ang))
            p_state.append(math.sin(p_projs[i].ang))
        while len(p_state) < 17:
            p_state.append(0)

        # add scorch
        if player.scorch is not None:
            p_state.append(player.scorch.x / self.disp_w)
            p_state.append(player.scorch.y / self.disp_h)
        else:
            p_state.append(0)
            p_state.append(0)

        # add shield
        if player.shield is not None:
            p_state.append(1)
        else:
            p_state.append(0)

        if enemy.speed == 0:
            e_dx, e_dy = 0, 0
        else:
            e_dx, e_dy = math.cos(enemy.ang), math.sin(enemy.ang)

        e_state = [
            enemy.x / self.disp_w, enemy.y / self.disp_h,
            e_dx, e_dy,
            enemy.health / consts.MAX_HEALTH, enemy.stam / consts.MAX_STAM,
            max(
                0, (1 - (self.frame - enemy.q_last_cast) / consts.PROJ_COOL)
            ),
            max(
                0, (1 - (self.frame - enemy.w_last_cast) / consts.SCORCH_COOL)
            ),
            max(
                0, (1 - (self.frame - enemy.e_last_cast) / consts.SHIELD_COOL)
            ),
        ]

        # add enemy projectiles
        e_projs = sorted(
                enemy.projs, key=lambda p: self.sq_dist(
                    (p.x, p.y), (player.x, player.y)
                )
        )
        for i in range(min(1, len(e_projs))):
            e_state.append(e_projs[i].x / self.disp_w)
            e_state.append(e_projs[i].y / self.disp_h)
            e_state.append(math.cos(e_projs[i].ang))
            e_state.append(math.sin(e_projs[i].ang))
        while len(e_state) < 17:
            e_state.append(0)

        # add scorch
        if enemy.scorch is not None:
            e_state.append(enemy.scorch.x / self.disp_w)
            e_state.append(enemy.scorch.y / self.disp_h)
        else:
            e_state.append(0)
            e_state.append(0)

        # add shield
        if enemy.shield is not None:
            e_state.append(1)
        else:
            e_state.append(0)

        return np.concatenate((
            np.asarray(p_state), np.asarray(e_state)
        )).astype(np.float32)

    def agent_rew(self, player, enemy, p_damage_taken, e_damage_taken):
        rew = 0
        if enemy.health <= 0:
            rew += 100
        if player.health <= 0:
            rew -= 100

        rew += e_damage_taken * 1
        rew -= p_damage_taken * 0.5

        rew -= player.took_unavailable

        # encourage playing in the center
        # rew -= math.hypot(
        #     player.x - self.disp_w // 2, player.y - self.disp_h // 2
        # )**2 * 0.000005

        # rew -= (player.max_stam - player.stam) * 0.0001

        # rew -= (self.frame) * 0.0001

        return rew

    def step(self, state_train=False, playing=False, p_ac=None, e_ac=None):
        # check to see if window was closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, True, None

        # keystrokes if playing
        if playing:
            # check keys pressed
            keys = pygame.key.get_pressed()
            x, y = pygame.mouse.get_pos()
            x = min(max(x, consts.MIN_X), consts.MAX_X)
            y = min(max(y, consts.MIN_Y), consts.MAX_Y)
            if keys[pygame.K_SPACE]:
                self.player.update_des(x, y)
            if keys[pygame.K_q]:
                self.player.play_q(x, y, self.frame)
            if keys[pygame.K_w]:
                self.player.play_w(x, y, self.frame)
            if keys[pygame.K_f]:
                self.player.play_e(self.frame)

        if not playing and p_ac is not None:
            # take player action
            p_ac_idx, p_ac_pos = p_ac
            self.player.take_action(p_ac_idx, p_ac_pos, self.frame)

        if e_ac is not None:
            # take enemy action
            e_ac_idx, e_ac_pos = e_ac
            self.enemy.take_action(e_ac_idx, e_ac_pos, self.frame)

        # set new observations
        self.p_ob = self.agent_ob(self.p_screen)
        self.e_ob = self.agent_ob(self.e_screen)

        # update game state and check damages
        self.player.update_state(self.frame)
        self.enemy.update_state(self.frame)

        p_damage_taken = self.check_damage(self.enemy, self.player)
        e_damage_taken = self.check_damage(self.player, self.enemy)

        self.draw()

        # get observations (screen pixels)
        ob = {
            "player": self.p_ob,
            "enemy": self.e_ob
        }

        state = None
        if state_train:
            # get state (for training)
            state = {
                "player": self.agent_state(self.player, self.enemy),
                "enemy": self.agent_state(self.enemy, self.player)
            }

        # get rewards
        p_rew = self.agent_rew(
            self.player, self.enemy, p_damage_taken, e_damage_taken
        )
        e_rew = self.agent_rew(
            self.enemy, self.player, e_damage_taken, p_damage_taken
        )

        rew = {
            "player": p_rew,
            "enemy": e_rew
        }

        # increase frame
        self.frame += 1

        # check if done
        done = False
        if self.player.health <= 0:
            done = True

        if self.enemy.health <= 0:
            done = True

        info = {
            "player": 0,
            "enemy": 0
        }

        return ob, state, rew, done, info

    def game_over(self):
        font = pygame.font.SysFont("Arial", 72)
        if self.enemy.health <= 0:
            message = "YOU WON"
        elif self.player.health <= 0:
            message = "YOU LOST"
        else:
            message = "GAME OVER"

        text = font.render(message, True, consts.WHITE)
        text_rect = text.get_rect(center=(self.disp_w // 2, self.disp_h // 2))

        self.disp.fill(consts.BACKGROUND_COLOR)
        self.disp.blit(text, text_rect)
        pygame.display.flip()

        # Wait for 2 seconds before quitting or restarting
        pygame.time.wait(2000)

    def quit(self):
        # quit when done
        pygame.display.quit()
        pygame.quit()
