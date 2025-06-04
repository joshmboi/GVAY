from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import torch
import time


class Logger:
    def __init__(self, log_dir="runs", name="default"):
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

    def save_models(self, p_pol, e_pol, iter):
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

    def close(self):
        self.writer.close()
