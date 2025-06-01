import pygame

import consts as consts


class Shield:
    def __init__(
            self, x=400, y=300, rad=25, color=consts.AGENT_PALETTE["e"],
            birth=0, duration=500, damage=consts.SHIELD_DAMAGE
    ):
        # position
        self.x = x
        self.y = y

        # radius
        self.rad = rad

        # color
        self.color = color

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

    def draw(self, screen, color=None):
        # change color if set
        if color:
            self.color = color

        s = pygame.Surface((self.rad * 2, self.rad * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            s, self.color, (self.rad, self.rad), self.rad
        )

        screen.blit(s, (int(self.x - self.rad), int(self.y - self.rad)))

    def expired(self):
        return pygame.time.get_ticks() >= self.birth + self.duration
