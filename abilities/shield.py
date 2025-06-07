import pygame

import consts as consts


class Shield:
    def __init__(
        self, x=400, y=300, rad=25, birth=0, duration=500,
        damage=consts.SHIELD_DAMAGE
    ):
        # position
        self.x = x
        self.y = y

        # radius
        self.rad = rad

        # duration
        self.birth = birth
        self.duration = duration

        # damage proection
        self.dmg = damage

    def damage(self):
        return self.dmg

    def update(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen, color):
        s = pygame.Surface((self.rad * 2, self.rad * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            s, color, (self.rad, self.rad), self.rad
        )

        screen.blit(s, (int(self.x - self.rad), int(self.y - self.rad)))

    def expired(self, cur_frame):
        return cur_frame >= self.birth + self.duration
