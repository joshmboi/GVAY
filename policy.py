import copy
import torch
import torch.optim as optim
import torch.nn.functional as F

import consts as consts
from models import Actor, Critic, AcEmbed


class Policy:
    def __init__(self, lr=1e-3, player=True, training=False):
        # device
        self.device = consts.DEVICE

        # whether player and whether training
        self.player = player
        self.training = training

        # actor and hidden state
        self.actor = Actor().to(self.device)
        self.actor_hidden = None
        if training:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        if self.training:
            # critic
            self.critic = Critic().to(self.device)
            self.critic_hidden = None
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # action embedding
        self.ac_embed = AcEmbed().to(self.device)
        if self.training:
            self.ac_embed_optimizer = optim.Adam(
                self.ac_embed.parameters(), lr=lr
            )

        # alpha for entropy control
        self.log_alpha_type = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )

        self.log_alpha_pos = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )
        if self.training:
            self.type_alpha_optimizer = optim.Adam([self.log_alpha_type], lr=lr)
            self.pos_alpha_optimizer = optim.Adam([self.log_alpha_pos], lr=lr)
            self.type_target_entropy = -0.5 * 1
            self.pos_target_entropy = -2 * 0.5

    def make_device_tensor(self, arr):
        return torch.tensor(arr).to(self.device)

    def update_policy(self, other_policy):
        self.actor.load_state_dict(other_policy.actor.state_dict())
        self.ac_embed.load_state_dict(other_policy.ac_embed.state_dict())

    def update_actor_hidden(self, ob):
        ob = self.make_device_tensor(ob).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            _, _, _, self.actor_hidden = self.actor(ob, self.actor_hidden)

    def get_action(self, ob, ac_mask):
        # so no gradients are calculated
        with torch.no_grad():
            ob = self.make_device_tensor(ob).unsqueeze(0).unsqueeze(0) / 255.0

            # get embed and position
            p_embed, ac_pos_raw, _, self.actor_hidden = self.actor(
                ob, self.actor_hidden
            )

            # get logits and probabilities
            sim_logits = self.ac_embed(p_embed)

            # mask out any actions
            if ac_mask is not None:
                ac_mask_tensor = torch.as_tensor(
                    ac_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                sim_logits = sim_logits.masked_fill(~ac_mask_tensor, -1e9)

            probs = F.softmax(sim_logits, dim=-1)

            # sample from policy and get embed
            ac_idx = probs.multinomial(1).item()

        return ac_idx, torch.sigmoid(ac_pos_raw).squeeze(0).cpu().numpy()

    def update_critic(self, batch, ac_mask, gamma=0.99, tau=0.005):
        # break open batch
        obs, ac_idxs, ac_poses, rews, nobs, dones = batch

        obs = self.make_device_tensor(obs) / 255.0
        ac_idxs = self.make_device_tensor(ac_idxs)
        ac_poses = self.make_device_tensor(ac_poses)
        rews = self.make_device_tensor(rews)
        nobs = self.make_device_tensor(nobs) / 255.0
        dones = self.make_device_tensor(dones).float()

        with torch.no_grad():
            # get next actions, observation embeddings, and q values
            p_embed_nexts, pos_next_raws, _, _ = self.actor(nobs)

            # get similarities and associated probabilities
            sim_logits_nexts = self.ac_embed(p_embed_nexts)

            # mask out invalid actions
            if ac_mask is not None:
                ac_mask_tensor = torch.as_tensor(
                    ac_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                sim_logits_nexts = sim_logits_nexts.masked_fill(
                    ~ac_mask_tensor,
                    -1e9
                )

            prob_nexts = F.softmax(sim_logits_nexts, dim=-1)

            ac_embed_nexts = (
                prob_nexts.unsqueeze(-1) * self.ac_embed.embedding.weight
            ).sum(dim=1)

            ac_nexts = torch.cat(
                [ac_embed_nexts, torch.sigmoid(pos_next_raws)], dim=-1
            )

            # calculate q values and target values
            q_nexts, _ = self.target_critic(nobs, ac_nexts)
            targets = rews + gamma * (1 - dones) * q_nexts

        # get action embeddings
        ac_embed = self.ac_embed.embedding(ac_idxs)
        acs = torch.cat([ac_embed, ac_poses], dim=-1)

        # q estimates given taken actions
        qs, _ = self.critic(obs, acs)

        critic_loss = F.mse_loss(qs, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for tc_param, c_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            tc_param.data.copy_(
                (1.0 - tau) * tc_param.data + tau * c_param.data
            )

        metric = {
            "critic_loss": critic_loss.item(),
            "q_values": qs.mean().item(),
            "q_val_min": qs.min().item(),
            "q_val_max": qs.max().item()
        }
        return metric

    def update_actor(self, batch, ac_mask):
        # break open batch
        obs, _, _, _, _, _ = batch

        obs = self.make_device_tensor(obs) / 255.0

        # get actions
        ac_embeds, ac_pos_raws, (
            ac_pos_means, ac_pos_stds
        ), _ = self.actor(obs)

        # ac_embeds entropy
        sim_logits = self.ac_embed(ac_embeds)

        # mask out actions cannot take
        if ac_mask is not None:
            ac_mask_tensor = torch.as_tensor(
                ac_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            sim_logits = sim_logits.masked_fill(~ac_mask_tensor, -1e9)

        probs = F.softmax(sim_logits, dim=-1)
        log_probs = F.log_softmax(sim_logits, dim=-1)

        ac_embeds = (
            probs.unsqueeze(-1) * self.ac_embed.embedding.weight
        ).sum(dim=1)

        acs = torch.cat(
            [ac_embeds, torch.sigmoid(ac_pos_raws)], dim=-1
        )

        # get q value using current critic
        qs, _ = self.critic(obs, acs)

        # ac_type entropy (Softmax)
        ac_type_entropy = -(probs * log_probs).sum(dim=-1)

        # ac_pos entropy (Gaussian)
        dist = torch.distributions.Normal(ac_pos_means, ac_pos_stds)
        ac_pos_entropy = dist.entropy().sum(dim=-1)

        # penalize for large means
        pos_center = 0.5
        pos_penalty = 1e-3 * ((ac_pos_means - pos_center) ** 2).mean()

        # total entropy
        entropy = (
                self.log_alpha_type.exp() * ac_type_entropy +
                self.log_alpha_pos.exp() * ac_pos_entropy
        ).detach()

        # alpha loss
        type_alpha_loss = -(
                self.log_alpha_type.exp() * (
                    ac_type_entropy - self.type_target_entropy
                ).detach()
        ).mean()
        pos_alpha_loss = -(
                self.log_alpha_pos.exp() * (
                    ac_pos_entropy - self.pos_target_entropy
                ).detach()
        ).mean()

        actor_loss = (-qs + entropy).mean() + pos_penalty

        self.actor_optimizer.zero_grad()
        self.ac_embed_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.ac_embed_optimizer.step()

        self.type_alpha_optimizer.zero_grad()
        type_alpha_loss.backward()
        self.type_alpha_optimizer.step()

        self.pos_alpha_optimizer.zero_grad()
        pos_alpha_loss.backward()
        self.pos_alpha_optimizer.step()

        means = ac_pos_means.mean(dim=-1)
        stds = ac_pos_stds.mean(dim=-1)

        metric = {
            "actor_loss": actor_loss.item(),
            "type_alpha": self.log_alpha_type.exp().mean().item(),
            "pos_alpha": self.log_alpha_pos.exp().mean().item(),
            "type_alpha_loss": type_alpha_loss.item(),
            "pos_alpha_loss": pos_alpha_loss.item(),
            "entropy": entropy.mean().item(),
            "means_x": means[0],
            "means_y": means[1],
            "stds_x": stds[0],
            "stds_y": stds[1]
        }
        return metric

    def load_policy(self, filepath):
        params = torch.load(
            filepath, map_location=consts.DEVICE, weights_only=True
        )
        if self.player:
            self.actor.load_state_dict(params["p_actor"])
            self.ac_embed.load_state_dict(params["p_ac_embed"])

            if self.training:
                self.critic.load_state_dict(params["p_critic"])
                self.actor_optimizer.load_state_dict(["p_actor_optim"])
                self.critic_optimizer.load_state_dict(["p_critic_optim"])
                self.ac_embed_optimizer.load_state_dict(["p_ac_embed_optim"])
        else:
            self.actor.load_state_dict(params["e_actor"])
            self.ac_embed.load_state_dict(params["e_ac_embed"])
