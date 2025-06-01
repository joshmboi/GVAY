import pygame
import math

import consts as consts


class Projectile:
    def __init__(
            self, x=400, y=300, rad=5, ang=0,
            color=consts.AGENT_PALETTE["q"], speed=10, max_dist=200,
            damage=consts.PROJ_DAMAGE
    ):
        # position and max distance
        self.x = x
        self.y = y
        self.dist = 0
        self.max_dist = max_dist

        # radius
        self.rad = rad

        # speed and angle
        self.speed = speed
        self.ang = ang

        # color
        self.color = color

        # damage
        self.dmg = damage

    def damage(self):
        return self.dmg

    def update(self):
        # update position
        dx = self.speed * math.cos(self.ang)
        dy = self.speed * math.sin(self.ang)

        self.x += dx
        self.y += dy

        # update current distance
        self.dist += math.sqrt(dx**2 + dy**2)

    def draw(self, screen, color=None):
        # set if color not equal to None
        if color:
            self.color = color

        pygame.draw.circle(
            screen, self.color, (int(self.x), int(self.y)), self.rad
        )

    def expired(self):
        return self.dist >= self.max_dist
