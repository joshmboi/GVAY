import copy
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import consts as consts
from sac import Actor, Critic, AcEmbed


class Policy:
    def __init__(
            self, actor=Actor(), critic=Critic(), ac_embed=AcEmbed(),
            lr=1e-3, device=consts.DEVICE, player=True
    ):
        # device
        self.device = device

        # actor and hidden state
        self.actor = actor.to(self.device)
        self.actor_hidden = None
        if player:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        if player:
            # critic
            self.critic = critic.to(self.device)
            self.critic_hidden = None
            self.target_critic = copy.deepcopy(critic).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # action embedding
        self.ac_embed = ac_embed.to(self.device)
        if player:
            self.ac_embed_optimizer = optim.Adam(
                self.ac_embed.parameters(), lr=lr
            )

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        if player:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-2)
            self.target_entropy = -7.0 / 2

    def make_device_tensor(self, arr):
        return torch.tensor(arr).to(self.device)

    def update_policy(self, other_policy):
        self.actor.load_state_dict(other_policy.actor.state_dict())
        self.ac_embed.load_state_dict(other_policy.ac_embed.state_dict())

    def update_actor_hidden(self, ob):
        ob = self.make_device_tensor(ob).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            _, _, _, self.actor_hidden = self.actor(ob, self.actor_hidden)

    def get_action(self, ob, ac_mask, eps=0.0):
        if np.random.rand() < eps:
            return (
                    np.random.randint(0, 5),
                    np.random.rand(2).astype(np.float32)
            )

        # so no gradients are calculated
        with torch.no_grad():
            ob = self.make_device_tensor(ob).unsqueeze(0).unsqueeze(0) / 255.0

            # get embed and position
            p_embed, ac_pos, (
                ac_pos_means, ac_pos_stds
            ), self.actor_hidden = self.actor(ob, self.actor_hidden)

            # get logits and probabilities
            sim_logits = self.ac_embed(p_embed)

            # mask out any actions
            if ac_mask:
                ac_mask_tensor = torch.as_tensor(
                    ac_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                sim_logits = sim_logits.masked_fill(ac_mask_tensor, -1e9)

            probs = F.softmax(sim_logits, dim=-1)

            # sample from policy and get embed
            ac_idx = probs.multinomial(1).item()

            # sample position and clamp
            dist = torch.distributions.Normal(ac_pos_means, ac_pos_stds)
            ac_pos = torch.clamp(dist.sample(), 0.0, 1.0)

        return ac_idx, ac_pos.squeeze(0).cpu().numpy()

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
            p_embed_nexts, pos_nexts, _, _ = self.actor(nobs)

            # get similarities and associated probabilities
            sim_logits_nexts = self.ac_embed(p_embed_nexts)

            # mask out invalid actions
            if ac_mask:
                ac_mask_tensor = torch.as_tensor(
                    ac_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                sim_logits_nexts = sim_logits_nexts.masked_fill(
                    ac_mask_tensor,
                    -1e9
                )

            prob_nexts = F.softmax(sim_logits_nexts, dim=-1)

            ac_embed_nexts = (
                prob_nexts.unsqueeze(-1) * self.ac_embed.embedding.weight
            ).sum(dim=1)

            ac_nexts = torch.cat([ac_embed_nexts, pos_nexts], dim=-1)

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

    def update_actor(self, batch, alpha=0.2):
        # break open batch
        obs, _, _, _, _, _ = batch

        obs = self.make_device_tensor(obs) / 255.0

        # get actions
        ac_embeds, ac_poses, (
            ac_pos_means, ac_pos_stds
        ), _ = self.actor(obs)
        acs = torch.cat([ac_embeds, ac_poses], dim=-1)

        # get q value using current critic
        qs, _ = self.critic(obs, acs)

        # ac_embeds entropy
        sim_logits = self.ac_embed(ac_embeds)
        log_probs_embed = -sim_logits.logsumexp(dim=-1)

        # ac_pos entropy (Gaussian)
        dist = torch.distributions.Normal(ac_pos_means, ac_pos_stds)
        log_probs_pos = dist.log_prob(ac_poses).sum(dim=-1)

        # total entropy
        entropy = log_probs_embed + log_probs_pos
        entropy = entropy

        # alpha loss
        alpha_loss = -(
            self.log_alpha.exp() * (entropy + self.target_entropy).detach()
        ).mean()

        actor_loss = (-qs + self.log_alpha.exp() * entropy).mean()

        self.actor_optimizer.zero_grad()
        self.ac_embed_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.ac_embed_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        metric = {
            "actor_loss": actor_loss.item(),
            "alpha": self.log_alpha.exp().mean().item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": entropy.mean().item()
        }
        return metric
