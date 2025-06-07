from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import pickle
import torch
import time

import consts as consts


class Logger:
    def __init__(self, timestamp=None, log_dir="runs", name="default"):
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

        print(timestamp)
        self.log_path = os.path.join(log_dir, f"{name}_{timestamp}")
        self.writer = SummaryWriter(self.log_path)

        self.vid_path = os.path.join(self.log_path, "vids")
        self.model_path = os.path.join(self.log_path, "models")

        os.makedirs(self.vid_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, scalars: dict, step, prefix=""):
        for key, val in scalars.items():
            if val is not None:
                self.writer.add_scalar(
                    f"{prefix}/{key}" if prefix else key, val, step
                )

    def log_video(self, frames, filename="vid.mp4", fps=30):
        filepath = os.path.join(self.vid_path, filename)
        H, W, _ = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filepath, fourcc, fps, (W, H))

        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()

    def save_rebuff(self, rebuff, iter, pretrain=False):
        filepath = os.path.join(
            self.model_path,
            f"rebuff_{iter}_{'pretrain' if pretrain else 'non'}.pkl"
        )

        # save file
        with open(filepath, "wb") as f:
            pickle.dump(rebuff, f)

    def load_rebuff(self, iter, pretrain=False):
        filepath = os.path.join(
            self.model_path,
            f"rebuff_{iter}_{'pretrain' if pretrain else 'non'}.pkl"
        )

        # load file
        with open(filepath, "rb") as f:
            rebuff = pickle.load(f)

        return rebuff

    def save_models(self, p_pol, e_pol, iter, pretrain=False):
        if pretrain:
            torch.save({
                "p_actor": p_pol.actor.state_dict(),
                "p_ac_embed": p_pol.ac_embed.state_dict(),
                "cnnlstm": p_pol.cnnlstm.state_dict(),
                "c1": p_pol.c1.state_dict(),
                "c2": p_pol.c2.state_dict(),
                "actor_opt": p_pol.actor_opt.state_dict(),
                "c1_opt": p_pol.c1_opt.state_dict(),
                "c2_opt": p_pol.c2_opt.state_dict(),
                "ac_embed_opt": p_pol.ac_embed_opt.state_dict(),
                "e_actor": e_pol.actor.state_dict(),
                "e_ac_embed": e_pol.ac_embed.state_dict(),
                "iteration": iter
            }, os.path.join(self.model_path, f"model_{iter}.pth"))
        else:
            torch.save({
                "p_actor": p_pol.actor.state_dict(),
                "p_critic": p_pol.critic.state_dict(),
                "p_ac_embed": p_pol.ac_embed.state_dict(),
                "p_actor_optim": p_pol.actor_optimizer.state_dict(),
                "p_critic_optim": p_pol.critic_optimizer.state_dict(),
                "p_ac_embed_optim": p_pol.ac_embed_optimizer.state_dict(),
                "e_actor": e_pol.actor.state_dict(),
                "e_ac_embed": e_pol.ac_embed.state_dict(),
                "iteration": iter
            }, os.path.join(self.model_path, f"model_{iter}.pth"))

    def load_models(self, p_pol, e_pol, iter, pretrain=False):
        filepath = os.path.join(self.model_path, f"model_{iter}.pth")
        params = torch.load(
            filepath, map_location=consts.DEVICE, weights_only=True
        )
        if pretrain:
            p_pol.actor.load_state_dict(params["p_actor"]),
            p_pol.ac_embed.load_state_dict(params["p_ac_embed"]),
            p_pol.cnnlstm.load_state_dict(params["cnnlstm"]),
            p_pol.c1.load_state_dict(params["c1"]),
            p_pol.c2.load_state_dict(params["c2"]),
            p_pol.actor_opt.load_state_dict(params["actor_opt"]),
            p_pol.c1_opt.load_state_dict(params["c1_opt"]),
            p_pol.c2_opt.load_state_dict(params["c2_opt"]),
            p_pol.ac_embed_opt.load_state_dict(params["ac_embed_opt"]),
            e_pol.actor.load_state_dict(params["e_actor"]),
            e_pol.ac_embed.load_state_dict(params["e_ac_embed"]),
        else:
            p_pol.actor.load_state_dict(params["p_actor"]),
            p_pol.critic.load_state_dict(params["p_critic"]),
            p_pol.ac_embed.load_state_dict(params["p_ac_embed"]),
            p_pol.actor_optimizer.load_state_dict(params["p_actor_optim"]),
            p_pol.critic_optimizer.load_state_dict(params["p_critic_optim"]),
            p_pol.ac_embed_optimizer.load_state_dict(
                params["p_ac_embed_optim"]
            ),
            e_pol.actor.load_state_dict(params["e_actor"]),
            e_pol.ac_embed.load_state_dict(params["e_ac_embed"]),

    def close(self):
        self.writer.close()
