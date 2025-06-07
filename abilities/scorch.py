import pygame

import consts as consts


class Scorch:
    def __init__(
        self, x, y, rad, cast_time, birth, duration,
        damage=consts.SCORCH_DAMAGE
    ):
        # position and max distance
        self.x = x
        self.y = y

        # radius
        self.rad = rad

        # duration
        self.cast_time = cast_time
        self.birth = birth
        self.duration = duration

        # damage
        self.dmg = damage

    def damage(self, cur_frame):
        # only do damage if currently active
        if self.active(cur_frame):
            return self.dmg / self.duration
        return 0

    def update(self):
        return None

    def draw(self, screen, color, cur_frame):
        # fraction for fading
        frac = (
            self.duration + self.cast_time -
            (cur_frame - self.birth)
        ) / self.duration

        # update circle color
        color_list = list(color)
        color_list[-1] = int(255 * min(1.0, frac))
        color = tuple(color_list)

        # cast outline of circle while casting
        width = 0
        if cur_frame - self.birth <= self.cast_time:
            width = 2

        s = pygame.Surface((self.rad * 2, self.rad * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            s, color, (self.rad, self.rad), self.rad, width
        )

        screen.blit(s, (int(self.x - self.rad), int(self.y - self.rad)))

    def active(self, cur_frame):
        return (
            cur_frame >= self.birth + self.cast_time and
            cur_frame <
            self.birth + self.duration + self.cast_time
        )

    def expired(self, cur_frame):
        return (
            cur_frame >=
            self.birth + self.duration + self.cast_time
        )
