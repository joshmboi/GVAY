import numpy as np
import cv2
import math
import os
import pygame

import consts as consts
from agent import Agent


class Game:
    def __init__(
        self, training=False, ac_mask=None,
        disp_w=consts.DISP_W, disp_h=consts.DISP_H
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

        # action mask for curriculum training
        self.ac_mask = ac_mask

        # init players
        self.player = Agent(
            self.disp_w // 4, self.disp_h // 2,
            palette=consts.PLAYER_PALETTE, rad=28, speed=6, player=True
        )
        self.enemy = Agent(
            (self.disp_w * 3) // 4, self.disp_h // 2,
            palette=consts.ENEMY_PALETTE, rad=28, speed=6, player=False
        )

        # copy over policy to enemy
        self.enemy.policy.update_policy(self.player.policy)

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

    def reset(self):
        # init clock
        self.clock = pygame.time.Clock()

        # reset players
        self.player.reset()
        self.enemy.reset()

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
            damage_taken += damagee.take_damage(sc.damage())

        return damage_taken

    def draw_screen(self, screen, player, enemy):
        # clear screen
        screen.fill(consts.BACKGROUND_COLOR)

        # draw player and enemy
        player.draw(screen, consts.PLAYER_PALETTE)
        enemy.draw(screen, consts.ENEMY_PALETTE)

        # draw ui
        player.draw_ui(screen)

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

    def agent_rew(self, player, enemy, p_damage_taken, e_damage_taken):
        rew = 0
        if enemy.health <= 0:
            rew += 100
        if player.health <= 0:
            rew -= 100

        rew += e_damage_taken * 1
        rew -= p_damage_taken * 0.5

        # encourage playing in the center
        # rew -= math.hypot(
        #     player.x - self.disp_w // 2, player.y - self.disp_h // 2
        # )**2 * 0.00005

        # rew -= (player.max_stam - player.stam) * 0.0001

        # rew -= (self.frame) * 0.0001

        return rew

    def step(self, playing=False):
        # check to see if window was closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, None, True, None

        # keystrokes if playing
        if playing:
            # check keys pressed
            keys = pygame.key.get_pressed()
            x, y = pygame.mouse.get_pos()
            x = min(max(x, 60), self.disp_w - 60)
            y = min(
                max(y, 28 + self.player.rad),
                self.disp_h - self.player.rad - 104
            )
            if keys[pygame.K_SPACE]:
                self.player.update_des(x, y)
            if keys[pygame.K_q]:
                self.player.play_q(x, y)
            if keys[pygame.K_w]:
                self.player.play_w(x, y)
            if keys[pygame.K_f]:
                self.player.play_e()

        ac_idx, ac_pos = None, None

        # get action and take it if acting frame
        if self.frame % consts.FPA == 0 and self.frame >= consts.WINDOW:
            # poll policy
            p_ac_idx, p_ac_pos = self.player.policy.get_action(
                self.p_ob, self.ac_mask
            )
            e_ac_idx, e_ac_pos = self.enemy.policy.get_action(
                self.e_ob, self.ac_mask
            )

            # take action
            if not playing:
                self.player.take_action(p_ac_idx, p_ac_pos)
            self.enemy.take_action(e_ac_idx, e_ac_pos)

            ac_idx = {
                "player": p_ac_idx,
                "enemy": e_ac_idx
            }

            ac_pos = {
                "player": p_ac_pos,
                "enemy": e_ac_pos
            }
        else:
            if not playing:
                self.player.policy.update_actor_hidden(self.p_ob)
            self.enemy.policy.update_actor_hidden(self.e_ob)

        self.p_ob = self.agent_ob(self.p_screen)
        self.e_ob = self.agent_ob(self.e_screen)

        # update game state and check damages
        self.player.update_state()
        self.enemy.update_state()

        p_damage_taken = self.check_damage(self.enemy, self.player)
        e_damage_taken = self.check_damage(self.player, self.enemy)

        self.draw()

        # get observations (screen pixels)
        ob = {
            "player": self.p_ob,
            "enemy": self.e_ob
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

        return ob, ac_idx, ac_pos, rew, done, info

    def game_over(self):
        font = pygame.font.SysFont("Arial", 72)
        if self.enemy.health <= 0:
            message = "YOU WON"
        elif self.player.health <= 0:
            message = "YOU LOST"

        text = font.render(message, True, consts.WHITE)
        text_rect = text.get_rect(center=(self.disp_w // 2, self.disp_h // 2))

        self.disp.fill(consts.BACKGROUND_COLOR)
        self.disp.blit(text, text_rect)
        pygame.display.flip()

        # Wait for 2 seconds before quitting or restarting
        pygame.time.wait(2000)

    def run(self):
        # reset
        self.reset()

        done = False
        while not done:
            _, _, _, _, done, info = self.step(playing=True)
            if done:
                self.game_over()
            self.clock.tick(consts.FPS)

        self.quit()

    def quit(self):
        # quit when done
        pygame.display.quit()
        pygame.quit()
