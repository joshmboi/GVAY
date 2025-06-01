import pygame

import consts as consts


class Scorch:
    def __init__(
            self, x=400, y=300, rad=25, color=consts.AGENT_PALETTE["w"],
            cast_time=1000, birth=0, duration=2000,
            damage=consts.SCORCH_DAMAGE
    ):
        # position and max distance
        self.x = x
        self.y = y

        # radius
        self.rad = rad

        # color
        self.color = color

        # duration
        self.cast_time = cast_time
        self.birth = birth
        self.duration = duration

        # damage
        self.dmg = damage

    def damage(self):
        # only do damage if currently active
        if self.active():
            return self.dmg / (consts.FPS * self.duration // 1000)
        return 0

    def update(self):
        return None

    def draw(self, screen, color=None):
        # fraction for fading
        frac = (
            self.duration + self.cast_time -
            (pygame.time.get_ticks() - self.birth)
        ) / self.duration

        # check if color argument
        if color:
            self.color = color

        # update circle color
        color_list = list(self.color)
        color_list[-1] = int(255 * min(1.0, frac))
        self.color = tuple(color_list)

        # cast outline of circle while casting
        width = 0
        if pygame.time.get_ticks() - self.birth <= self.cast_time:
            width = 2

        s = pygame.Surface((self.rad * 2, self.rad * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            s, self.color, (self.rad, self.rad), self.rad, width
        )

        screen.blit(s, (int(self.x - self.rad), int(self.y - self.rad)))

    def active(self):
        return (
            pygame.time.get_ticks() >= self.birth + self.cast_time and
            pygame.time.get_ticks() <
            self.birth + self.duration + self.cast_time
        )

    def expired(self):
        return (
            pygame.time.get_ticks() >=
            self.birth + self.duration + self.cast_time
        )
