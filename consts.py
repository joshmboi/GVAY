import torch

# display
DISP_W = 800
DISP_H = 600
FPS = 30
FPA = 6
WINDOW = 2 * FPA

# training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REBUFF_SIZE = 40000
BATCH_SIZE = 500

# number of iterations for things
ITERS_PER_EVAL = 20
SAVE_EVERY = 50
ENEMY_UPDATE = 100
TRAIN_STEPS = 10
CRITIC_ONLY = 20

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (33, 33, 33)
UI_BACKGROUND_COLOR = (100, 100, 100)

# abilities
AB_BACKGROUND = (127, 127, 127)
AB_FOREGROUND = (216, 216, 216)
AB_COOLDOWN = (127, 127, 127, 127)

# agent colors
AGENT_PALETTE = {
        "agent": (61, 84, 158),
        "q": (255, 197, 211),
        "w": (255, 192, 103, 255),
        "e": (201, 160, 220, 180)
}

# player colors
PLAYER_PALETTE = {
        "agent": (179, 235, 242),
        "q": (179, 204, 242),
        "w": (179, 242, 209, 255),
        "e": (107, 151, 156, 180)
}

# enemy colors
ENEMY_PALETTE = {
        "agent": (255, 105, 97),
        "q": (255, 97, 143),
        "w": (255, 177, 97, 255),
        "e": (153, 42, 39, 180)
}

HEALTH_LIGHT = (194, 233, 191)
HEALTH_DARK = (117, 153, 111)

STAM_LIGHT = (255, 238, 140)
STAM_DARK = (204, 178, 51)

# damages
PROJ_DAMAGE = 10
SCORCH_DAMAGE = 60
SHIELD_DAMAGE = 30
