import pygame
import math
import numpy as np

import consts as consts
from abilities.proj import Projectile
from abilities.scorch import Scorch
from abilities.shield import Shield


class Agent:
    def __init__(self, x, y):
        # position
        self.init_x = x
        self.init_y = y

        # radius
        self.rad = consts.AGENT_RAD

        # speed and angle
        self.base_speed = consts.AGENT_SPEED
        self.speed = 0
        self.ang = 0

        # ability trackers
        self.projs = []
        self.scorch = None
        self.shield = None

        # ability characteristics
        self.q_last_cast = -consts.PROJ_COOL
        self.w_last_cast = -consts.SCORCH_COOL
        self.e_last_cast = -consts.SHIELD_COOL
        self.shield_val = 0

        # flag for taking unavailable move
        self.took_unavailable = 0

        # health n stam
        self.stam = consts.MAX_STAM
        self.health = consts.MAX_HEALTH

    def reset(self, x, y):
        # position and desired position
        if x is None or y is None:
            self.x = self.init_x
            self.y = self.init_y
        else:
            self.x = x
            self.y = y

        self.des_x = self.x
        self.des_y = self.y

        # speed and angle
        self.speed = 0
        self.ang = 0

        # ability trackers
        self.projs = []
        self.scorch = None
        self.shield = None

        # ability cooldowns
        self.q_last_cast = -consts.PROJ_COOL
        self.w_last_cast = -consts.SCORCH_COOL
        self.e_last_cast = -consts.SHIELD_COOL

        # stamina and health
        self.stam = consts.MAX_STAM
        self.health = consts.MAX_HEALTH

    def update_des(self, des_x, des_y):
        self.des_x = des_x
        self.des_y = des_y

    def play_q(self, qx, qy, cur_frame):
        if (
            cur_frame <= self.q_last_cast + consts.PROJ_COOL or
            self.stam < consts.PROJ_STAM
        ):
            self.took_unavailable = 1
            return None

        # get projectile position and direction
        proj_ang = math.atan2(qy - self.y, qx - self.x)
        offset = self.rad + consts.PROJ_RAD

        # add projectile
        proj = Projectile(
            x=self.x + offset * math.cos(proj_ang),
            y=self.y + offset * math.sin(proj_ang),
            ang=proj_ang,
            rad=consts.PROJ_RAD,
            max_dist=consts.PROJ_DIST,
            speed=consts.PROJ_SPEED
        )
        self.projs.append(proj)

        self.q_last_cast = cur_frame
        self.stam -= consts.PROJ_STAM

    def play_w(self, wx, wy, cur_frame):
        if (
            cur_frame <= self.w_last_cast + consts.SCORCH_COOL or
            self.stam < consts.SCORCH_STAM
        ):
            self.took_unavailable = 1
            return None

        # check to see if in range of player
        if math.hypot(self.x - wx, self.y - wy) > consts.SCORCH_CASTDIST:
            return None

        self.scorch = Scorch(
            x=wx, y=wy, rad=consts.SCORCH_RAD, birth=cur_frame,
            cast_time=consts.SCORCH_CASTTIME, duration=consts.SCORCH_DURATION
        )

        self.w_last_cast = cur_frame
        self.stam -= consts.SCORCH_STAM

    def play_e(self, cur_frame):
        if (
            cur_frame <= self.e_last_cast + consts.SHIELD_COOL or
            self.stam < consts.SHIELD_STAM
        ):
            self.took_unavailable = 1
            return None

        shield_rad = self.rad + 4
        self.shield_val = consts.SHIELD_DAMAGE

        # add shield
        self.shield = Shield(
            x=self.x, y=self.y, rad=shield_rad,
            birth=cur_frame, duration=consts.SHIELD_DURATION
        )

        self.e_last_cast = cur_frame
        self.stam -= consts.SHIELD_STAM

    def take_damage(self, dmg):
        res_dmg = dmg - self.shield_val

        if res_dmg > 0:
            self.health = max(self.health - res_dmg, 0)

        self.shield_val = max(self.shield_val - dmg, 0)

        return max(res_dmg, 0)

    def update_state(self, cur_frame):
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
            self.health + consts.HEALTH_REG / consts.FPS, consts.MAX_HEALTH
        )
        self.stam = min(
            self.stam + consts.STAM_REG / consts.FPS, consts.MAX_STAM
        )

        # update projectiles
        for p in self.projs:
            p.update()

        self.projs = [p for p in self.projs if not p.expired()]

        # update scorches
        if self.scorch:
            if self.scorch.expired(cur_frame):
                self.scorch = None
            else:
                self.scorch.update()

        # update shield
        if self.shield:
            if self.shield.expired(cur_frame):
                self.shield = None
                self.shield_val = 0
            else:
                self.shield.update(self.x, self.y)

    def draw(self, screen, palette, cur_frame):
        # draw scorch
        if self.scorch:
            self.scorch.draw(screen, palette["w"], cur_frame)

        pygame.draw.circle(
            screen, palette["agent"], (int(self.x), int(self.y)), self.rad
        )

        # draw projectiles
        for p in self.projs:
            p.draw(screen, palette["q"])

        # draw shield
        if self.shield:
            self.shield.draw(screen, palette["e"])

        x_pos = int(self.x - consts.AGENT_BAR_W // 2)
        y_pos = int(
            self.y - consts.AGENT_RAD - consts.AGENT_PAD -
            consts.HEALTH_H - consts.STAM_H
        )

        # health
        pygame.draw.rect(
            screen, consts.HEALTH_DARK,
            (x_pos, y_pos, consts.AGENT_BAR_W, consts.HEALTH_H)
        )
        pygame.draw.rect(
            screen, consts.HEALTH_LIGHT,
            (
                x_pos, y_pos,
                int(consts.AGENT_BAR_W * self.health / consts.MAX_HEALTH),
                consts.HEALTH_H
            )
        )

        # Stamina bar (below health)
        y_pos += consts.HEALTH_H
        pygame.draw.rect(
            screen, consts.STAM_DARK,
            (x_pos, y_pos, consts.AGENT_BAR_W, consts.STAM_H)
        )
        pygame.draw.rect(
            screen, consts.STAM_LIGHT,
            (
                x_pos, y_pos,
                int(consts.AGENT_BAR_W * self.stam / consts.MAX_STAM),
                consts.STAM_H
            )
        )

    def draw_ui(self, screen, cur_frame):
        # fonts
        ab_font = pygame.font.SysFont("Arial", 36)
        bar_font = pygame.font.SysFont("Arial", 24)

        # display
        screen_w, screen_h = screen.get_width(), screen.get_height()

        # bar sizes
        bar_x = screen_w - consts.PLAYER_BAR_W - consts.PAD
        bar_y = (
            screen_h - 2 * (consts.PLAYER_BAR_H) - consts.BAR_PAD - consts.PAD
        )

        # blank grey background
        blank_y = bar_y - consts.PAD
        pygame.draw.rect(
            screen, consts.UI_BACKGROUND_COLOR, (
                0, blank_y, consts.DISP_W, consts.DISP_H - blank_y
            )
        )

        # health bar
        pygame.draw.rect(
            screen, consts.HEALTH_DARK,
            (bar_x, bar_y, consts.PLAYER_BAR_W, consts.PLAYER_BAR_H)
        )
        pygame.draw.rect(
            screen, consts.HEALTH_LIGHT,
            (
                bar_x, bar_y,
                int(consts.PLAYER_BAR_W * (self.health / consts.MAX_HEALTH)),
                consts.PLAYER_BAR_H
            )
        )

        # health number
        h_text = bar_font.render(
            f"{round(self.health, 1)}/{consts.MAX_HEALTH}", True, (0, 0, 0)
        )
        screen.blit(
            h_text,
            (
                bar_x + consts.PLAYER_BAR_W // 2 - h_text.get_width() // 2,
                bar_y + consts.PLAYER_BAR_H // 2 - h_text.get_height() // 2
            )
        )

        # stamina bar
        bar_y += consts.PLAYER_BAR_H + consts.BAR_PAD
        pygame.draw.rect(
            screen, consts.STAM_DARK,
            (bar_x, bar_y, consts.PLAYER_BAR_W, consts.PLAYER_BAR_H)
        )
        pygame.draw.rect(
            screen, consts.STAM_LIGHT,
            (
                bar_x, bar_y,
                int(consts.PLAYER_BAR_W * (self.stam / consts.MAX_STAM)),
                consts.PLAYER_BAR_H
            )
        )

        # stamina number
        s_text = bar_font.render(
            f"{round(self.stam, 1)}/{consts.MAX_STAM}", True, (0, 0, 0)
        )
        screen.blit(
            s_text,
            (
                bar_x + consts.PLAYER_BAR_W // 2 - s_text.get_width() // 2,
                bar_y + consts.PLAYER_BAR_H // 2 - s_text.get_height() // 2
            )
        )

        # abilities
        abs = [
            ("Q", self.q_last_cast, consts.PROJ_COOL),
            ("W", self.w_last_cast, consts.SCORCH_COOL),
            ("E", self.e_last_cast, consts.SHIELD_COOL)
        ]

        # ability box size and spacing
        box_pad = (
                (bar_x - len(abs) * consts.BOX_SIZE) //
                (len(abs) + 1)
        )

        # position
        y_pos = (
            screen_h - consts.PLAYER_BAR_H - consts.PAD -
            consts.BAR_PAD // 2 - consts.BOX_SIZE // 2
        )
        x_start = box_pad

        for i, (key, last_cast, cool) in enumerate(abs):
            x_pos = x_start + i * (consts.BOX_SIZE + box_pad)
            rect = pygame.Rect(x_pos, y_pos, consts.BOX_SIZE, consts.BOX_SIZE)

            # background box
            pygame.draw.rect(screen, consts.AB_BACKGROUND, rect)
            pygame.draw.rect(screen, consts.AB_FOREGROUND, rect, 4)

            elapsed = cur_frame - last_cast
            if elapsed < cool:
                # cooldown overlay
                percent = elapsed / cool
                over_height = int(consts.BOX_SIZE * percent)
                over = pygame.Rect(
                    x_pos, y_pos + consts.BOX_SIZE - over_height,
                    consts.BOX_SIZE, over_height
                )
                pygame.draw.rect(screen, consts.AB_COOLDOWN, over)

                text = ab_font.render(
                    str(int((cool - elapsed) / consts.FPS)), True,
                    (255, 255, 255)
                )
            else:
                # ability key
                text = ab_font.render(key, True, (255, 255, 255))

            screen.blit(
                text, (
                    x_pos + consts.BOX_SIZE // 2 - text.get_width() // 2,
                    y_pos + consts.BOX_SIZE // 2 - text.get_height() // 2
                )
            )

    def take_action(self, ac_idx, ac_pos, cur_frame):
        self.took_unavailable = 0
        # grab action type and action value
        ac_val = np.array(
            [consts.MAX_X - consts.MIN_X, consts.MAX_Y - consts.MIN_Y]
        ) * ac_pos + np.array(
                [consts.MIN_X, consts.MIN_Y]
            )

        # 0 - nothing, 1 - move, 2 - q, 3 - w, 4 - e
        if ac_idx == 0:
            pass
        elif ac_idx == 1:
            self.update_des(ac_val[0], ac_val[1])
        elif ac_idx == 2:
            self.play_q(ac_val[0], ac_val[1], cur_frame)
        elif ac_idx == 3:
            self.play_w(ac_val[0], ac_val[1], cur_frame)
        elif ac_idx == 4:
            self.play_e(cur_frame)
