import pygame
import math
import numpy as np

import consts as consts
from abilities.proj import Projectile
from abilities.scorch import Scorch
from abilities.shield import Shield


class Agent:
    def __init__(
        self, x, y, palette=consts.AGENT_PALETTE, speed=6, rad=28
    ):
        # position
        self.init_x = x
        self.init_y = y

        # radius
        self.rad = rad

        # speed and angle
        self.base_speed = speed
        self.speed = 0
        self.ang = 0

        # color
        self.palette = palette

        # ability trackers
        self.projs = []
        self.scorch = None
        self.shield = None

        # abilities (times in milliseconds)
        self.q_cool = 500
        self.q_last_cast = -self.q_cool
        self.q_rad = 8
        self.q_dist = 400
        self.q_stam = 10

        self.w_cool = 2500
        self.w_last_cast = -self.w_cool
        self.w_cast_time = 300
        self.w_cast_dist = 200
        self.w_rad = 60
        self.w_duration = 2000
        self.w_stam = 20

        self.e_cool = 5000
        self.e_last_cast = -self.e_cool
        self.e_duration = 1000
        self.e_stam = 35
        self.shield_val = 0

        # stamina and health
        self.max_stam = 100
        self.stam = self.max_stam

        # set to max to refill each frame so no mana limits
        self.stam_reg = self.max_stam  # 2

        self.max_health = 200
        self.health = self.max_health
        self.health_reg = 1

    def reset(self):
        # position and desired position
        self.x = self.init_x
        self.y = self.init_y
        self.des_x = self.init_x
        self.des_y = self.init_y

        # speed and angle
        self.speed = 0
        self.ang = 0

        # ability trackers
        self.projs = []
        self.scorch = None
        self.shield = None

        # ability cooldowns
        self.q_last_cast = -self.q_cool
        self.w_last_cast = -self.w_cool
        self.e_last_cast = -self.e_cool

        # stamina and health
        self.stam = self.max_stam
        self.health = self.max_health

    def update_des(self, des_x, des_y):
        self.des_x = des_x
        self.des_y = des_y

    def play_q(self, qx, qy):
        if (
            pygame.time.get_ticks() <= self.q_last_cast + self.q_cool or
            self.stam < self.q_stam
        ):
            return None

        # get projectile position and direction
        proj_ang = math.atan2(qy - self.y, qx - self.x)
        offset = self.rad + self.q_rad

        # add projectile
        proj = Projectile(
            x=self.x + offset * math.cos(proj_ang),
            y=self.y + offset * math.sin(proj_ang),
            ang=proj_ang,
            rad=self.q_rad,
            max_dist=self.q_dist,
            color=self.palette["q"],
            speed=14
        )
        self.projs.append(proj)

        self.q_last_cast = pygame.time.get_ticks()
        self.stam -= self.q_stam

    def play_w(self, wx, wy):
        if (
            pygame.time.get_ticks() <= self.w_last_cast + self.w_cool or
            self.stam < self.w_stam
        ):
            return None

        # check to see if in range of player
        if math.hypot(self.x - wx, self.y - wy) > self.w_cast_dist:
            return None

        self.scorch = Scorch(
            x=wx, y=wy, rad=self.w_rad, birth=pygame.time.get_ticks(),
            cast_time=self.w_cast_time, duration=self.w_duration,
            color=self.palette["w"]
        )

        self.w_last_cast = pygame.time.get_ticks()
        self.stam -= self.w_stam

    def play_e(self):
        if (
            pygame.time.get_ticks() <= self.e_last_cast + self.e_cool or
            self.stam < self.e_stam
        ):
            return None

        shield_rad = self.rad + 4
        self.shield_val = consts.SHIELD_DAMAGE

        # add shield
        self.shield = Shield(
            x=self.x, y=self.y, rad=shield_rad,
            birth=pygame.time.get_ticks(), duration=self.e_duration,
            color=self.palette["e"]
        )

        self.e_last_cast = pygame.time.get_ticks()
        self.stam -= self.e_stam

    def take_damage(self, dmg):
        res_dmg = dmg - self.shield_val

        if res_dmg > 0:
            self.health = max(self.health - res_dmg, 0)

        self.shield_val = max(self.shield_val - dmg, 0)

        return max(res_dmg, 0)

    def update_state(self):
        # check to see if should move
        if math.hypot(self.des_x - self.x, self.des_y - self.y) < 5:
            self.speed = 0
        else:
            self.ang = math.atan2(self.des_y - self.y, self.des_x - self.x)
            self.speed = self.base_speed

        # move player
        self.x += self.speed * math.cos(self.ang)
        self.y += self.speed * math.sin(self.ang)

        # update stamina and health regen
        self.health = min(
            self.health + self.health_reg / consts.FPS, self.max_health
        )
        self.stam = min(self.stam + self.stam_reg / consts.FPS, self.max_stam)

        # update projectiles
        for p in self.projs:
            p.update()

        self.projs = [p for p in self.projs if not p.expired()]

        # update scorches
        if self.scorch:
            if self.scorch.expired():
                self.scorch = None
            else:
                self.scorch.update()

        # update shield
        if self.shield:
            if self.shield.expired():
                self.shield = None
                self.shield_val = 0
            else:
                self.shield.update(self.x, self.y)

    def draw(self, screen, palette=None):
        # draw scorch
        if self.scorch:
            self.scorch.draw(screen, palette["w"])

        pygame.draw.circle(
            screen, palette["agent"], (int(self.x), int(self.y)), self.rad
        )

        # draw projectiles
        for p in self.projs:
            p.draw(screen, palette["q"])

        # draw shield
        if self.shield:
            self.shield.draw(screen, palette["e"])

        bar_w = 120
        health_h = 12
        stam_h = 4
        x_pos = int(self.x - bar_w // 2)
        y_pos = int(self.y - self.rad - health_h - stam_h - 12)

        # health
        pygame.draw.rect(
            screen, consts.HEALTH_DARK, (x_pos, y_pos, bar_w, health_h)
        )
        pygame.draw.rect(
            screen, consts.HEALTH_LIGHT,
            (
                x_pos, y_pos,
                int(bar_w * self.health / self.max_health), health_h
            )
        )

        # Stamina bar (below health)
        y_pos += health_h
        pygame.draw.rect(
            screen, consts.STAM_DARK, (x_pos, y_pos, bar_w, stam_h)
        )
        pygame.draw.rect(
            screen, consts.STAM_LIGHT,
            (x_pos, y_pos, int(bar_w * self.stam / self.max_stam), stam_h)
        )

    def draw_ui(self, screen):
        # fonts
        ab_font = pygame.font.SysFont("Arial", 36)
        bar_font = pygame.font.SysFont("Arial", 24)

        # padding
        pad = 16
        bar_pad = 8

        # display
        screen_w, screen_h = screen.get_width(), screen.get_height()

        # bar sizes
        bar_w = 520
        bar_h = 32
        bar_x = screen_w - bar_w - pad
        bar_y = screen_h - 2 * (bar_h) - bar_pad - pad

        # blank grey background
        blank_y = bar_y - pad
        pygame.draw.rect(
            screen, consts.UI_BACKGROUND_COLOR, (
                0, blank_y, consts.DISP_W, consts.DISP_H - blank_y
            )
        )

        # health bar
        pygame.draw.rect(
            screen, consts.HEALTH_DARK, (bar_x, bar_y, bar_w, bar_h)
        )
        pygame.draw.rect(
            screen, consts.HEALTH_LIGHT,
            (bar_x, bar_y, int(bar_w * (self.health / self.max_health)), bar_h)
        )

        # health number
        h_text = bar_font.render(
            f"{round(self.health, 1)}/{self.max_health}", True, (0, 0, 0)
        )
        screen.blit(
            h_text,
            (
                bar_x + bar_w // 2 - h_text.get_width() // 2,
                bar_y + bar_h // 2 - h_text.get_height() // 2
            )
        )

        # stamina bar
        bar_y += bar_h + bar_pad
        pygame.draw.rect(
            screen, consts.STAM_DARK, (bar_x, bar_y, bar_w, bar_h)
        )
        pygame.draw.rect(
            screen, consts.STAM_LIGHT,
            (bar_x, bar_y, int(bar_w * (self.stam / self.max_stam)), bar_h)
        )

        # stamina number
        s_text = bar_font.render(
            f"{round(self.stam, 1)}/{self.max_stam}", True, (0, 0, 0)
        )
        screen.blit(
            s_text,
            (
                bar_x + bar_w // 2 - s_text.get_width() // 2,
                bar_y + bar_h // 2 - s_text.get_height() // 2
            )
        )

        # abilities
        abs = [
            ("Q", self.q_last_cast, self.q_cool),
            ("W", self.w_last_cast, self.w_cool),
            ("E", self.e_last_cast, self.e_cool)
        ]

        # ability box size and spacing
        box_size = 48
        box_pad = (
                (bar_x - len(abs) * box_size) //
                (len(abs) + 1)
        )

        # position
        y_pos = screen_h - bar_h - pad - bar_pad // 2 - box_size // 2
        x_start = box_pad

        for i, (key, last_cast, cool) in enumerate(abs):
            x_pos = x_start + i * (box_size + box_pad)
            rect = pygame.Rect(x_pos, y_pos, box_size, box_size)

            # background box
            pygame.draw.rect(screen, consts.AB_BACKGROUND, rect)
            pygame.draw.rect(screen, consts.AB_FOREGROUND, rect, 4)

            elapsed = pygame.time.get_ticks() - last_cast
            if elapsed < cool:
                # cooldown overlay
                percent = elapsed / cool
                over_height = int(box_size * percent)
                over = pygame.Rect(
                    x_pos, y_pos + box_size - over_height,
                    box_size, over_height
                )
                pygame.draw.rect(screen, consts.AB_COOLDOWN, over)

                text = ab_font.render(
                    str((cool - elapsed) // 1000), True, (255, 255, 255)
                )
            else:
                # ability key
                text = ab_font.render(key, True, (255, 255, 255))

            screen.blit(
                text, (
                    x_pos + box_size // 2 - text.get_width() // 2,
                    y_pos + box_size // 2 - text.get_height() // 2
                )
            )

    def take_action(self, ac_idx, ac_pos):
        # grab action type and action value
        ac_val = np.array(
            [consts.DISP_W - 120, consts.DISP_H - 104 - 28 * 3]
        ) * ac_pos + np.array(
                [60, 56]
            )

        # 0 - nothing, 1 - move, 2 - q, 3 - w, 4 - e
        if ac_idx == 0:
            pass
        elif ac_idx == 1:
            self.update_des(ac_val[0], ac_val[1])
        elif ac_idx == 2:
            self.play_q(ac_val[0], ac_val[1])
        elif ac_idx == 3:
            self.play_w(ac_val[0], ac_val[1])
        elif ac_idx == 4:
            self.play_e()
