import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import consts as consts
from models import CNNLSTM, ActorModule, CriticModule, AcEmbed


class PTPolicy:
    def __init__(self, lr=1e-3, player=True, training=False):
        # device
        self.device = consts.DEVICE

        # whether player and whether training
        self.player = player
        self.training = training

        # init actor and hidden state
        self.actor = ActorModule(pretrain=True).to(self.device)
        if self.training:
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)

        # action embedding
        self.ac_embed = AcEmbed().to(self.device)
        if self.training:
            self.ac_embed_opt = optim.Adam(self.ac_embed.parameters(), lr=lr)

        if self.training:
            # init cnnlstm
            self.cnnlstm_hidden = None
            self.cnnlstm = CNNLSTM(pretrain=True).to(self.device)
            self.cnnlstm_opt = optim.Adam(self.cnnlstm.parameters(), lr=lr)

            # init critic
            self.c1 = CriticModule(pretrain=True).to(self.device)
            self.c2 = CriticModule(pretrain=True).to(self.device)
            self.c1_opt = optim.Adam(self.c1.parameters(), lr=lr)
            self.c2_opt = optim.Adam(self.c2.parameters(), lr=lr)

            # target critics
            self.tc1 = copy.deepcopy(self.c1).to(self.device)
            self.tc2 = copy.deepcopy(self.c2).to(self.device)

        # alphas for entropy control
        self.log_type_alph = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )
        self.log_pos_alph = torch.tensor(
            0.25, requires_grad=True, device=self.device
        )

        if self.training:
            self.type_alph_opt = optim.Adam([self.log_type_alph], lr=lr)
            self.pos_alph_opt = optim.Adam([self.log_pos_alph], lr=lr)
            self.type_target_entropy = 0
            self.pos_target_entropy = -1.0

    def make_device_tensor(self, arr, dtype=None):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def get_logits(self, p_embed, ac_mask):
        # get logits
        ac_logits = self.ac_embed(p_embed)

        # mask out any actions
        if ac_mask is not None:
            ac_mask_tensor = torch.as_tensor(
                ac_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            ac_logits = ac_logits.masked_fill(~ac_mask_tensor, -1e9)

        return ac_logits

    def update_policy(self, other_policy):
        self.actor.load_state_dict(other_policy.actor.state_dict())
        self.ac_embed.load_state_dict(other_policy.ac_embed.state_dict())
        self.log_type_alph = other_policy.log_type_alph
        self.log_pos_alph = other_policy.log_type_alph

    def update_cnnlstm_hidden(self, ob):
        ob = self.make_device_tensor(ob).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            _, self.cnnlstm_hidden = self.cnnlstm(ob, self.cnnlstm_hidden)

    def get_action(self, state, ac_mask):
        # so no gradients are calculated
        with torch.no_grad():
            state = self.make_device_tensor(state).unsqueeze(0)

            # get embed and position dist stats
            p_embed, (ac_pos_mean, ac_pos_std) = self.actor(state)

            # get prob distribution and sample
            logits = self.get_logits(p_embed, ac_mask)
            type_dist = torch.distributions.Categorical(logits=logits)
            ac_idx = type_dist.sample().squeeze(0).cpu().numpy()

            # get position
            pos_dist = torch.distributions.Normal(ac_pos_mean, ac_pos_std)
            ac_pos = torch.sigmoid(pos_dist.sample()).squeeze(0).cpu().numpy()

        return ac_idx, ac_pos

    def update_cnnlstm(self, batch):
        # break open batch
        obs = self.make_device_tensor(batch.obs) / 255.0
        states = self.make_device_tensor(batch.states)

        # predict states and get loss
        pred_states, _ = self.cnnlstm(obs)
        cnnlstm_loss = F.mse_loss(pred_states, states)

        self.cnnlstm_opt.zero_grad()
        cnnlstm_loss.backward()
        self.cnnlstm_opt.step()

        metrics = {
            "cnnlstm_loss": cnnlstm_loss.item()
        }
        return metrics

    def update_critic(self, batch, ac_mask, gamma=0.99, tau=0.005):
        # break open batch
        states = self.make_device_tensor(batch.states)
        ac_idxs = self.make_device_tensor(batch.ac_idxs)
        ac_poses = self.make_device_tensor(batch.ac_poses)
        rews = self.make_device_tensor(batch.rews)
        n_states = self.make_device_tensor(batch.n_states)
        dones = self.make_device_tensor(batch.dones, dtype=torch.float32)

        with torch.no_grad():
            # get next actions, observation embeddings, and q values
            p_embed_nexts, (
                pos_next_means, pos_next_stds
            ) = self.actor(n_states)

            # get dist and use to determine in embedding space
            next_logits = self.get_logits(p_embed_nexts, ac_mask)
            type_dist = torch.distributions.Categorical(logits=next_logits)
            ac_embed_weight = self.ac_embed.embedding.weight.detach()
            ac_embed_nexts = type_dist.probs @ ac_embed_weight

            # get dist and use for positions
            pos_dist = torch.distributions.Normal(
                pos_next_means, pos_next_stds
            )
            ac_pos_nexts = torch.sigmoid(pos_dist.sample())

            ac_nexts = torch.cat(
                [ac_embed_nexts, torch.sigmoid(ac_pos_nexts)], dim=-1
            )

            # calculate q values and target values
            q_nexts = torch.minimum(
                self.tc1(n_states, ac_nexts), self.tc2(n_states, ac_nexts)
            )
            targets = rews + gamma * (1 - dones) * q_nexts

            # get action embeddings
            ac_embed = self.ac_embed.embedding(ac_idxs)
            acs = torch.cat([ac_embed, ac_poses], dim=-1)

        # q estimates given taken actions
        q1s = self.c1(states, acs)
        q2s = self.c2(states, acs)

        c1_loss = F.mse_loss(q1s, targets)
        c2_loss = F.mse_loss(q2s, targets)

        # backprop losses
        self.c1_opt.zero_grad()
        c1_loss.backward()
        self.c1_opt.step()

        self.c2_opt.zero_grad()
        c2_loss.backward()
        self.c2_opt.step()

        with torch.no_grad():
            # update targets
            for tc_param, c_param in zip(
                self.tc1.parameters(), self.c1.parameters()
            ):
                tc_param.copy_((1.0 - tau) * tc_param + tau * c_param)
            for tc_param, c_param in zip(
                self.tc2.parameters(), self.c2.parameters()
            ):
                tc_param.copy_((1.0 - tau) * tc_param + tau * c_param)

            q1_mean = q1s.mean()
            q2_mean = q2s.mean()

        metrics = {
            "c1_loss": c1_loss.item(),
            "c2_loss": c2_loss.item(),
            "q1_vals": q1_mean,
            "q2_vals": q2_mean
        }
        return metrics

    def update_actor(self, batch, ac_mask):
        # break open batch
        states = self.make_device_tensor(batch.states)

        # get actions
        p_embed, (ac_pos_means, ac_pos_stds) = self.actor(states)

        # get type dist and action embeds
        logits = self.get_logits(p_embed, ac_mask)
        type_probs = F.softmax(logits, dim=-1)
        ac_embeds = type_probs @ self.ac_embed.embedding.weight

        # pos dist and sampling
        pos_dist = torch.distributions.Normal(
            ac_pos_means, ac_pos_stds
        )
        ac_pos_raw = pos_dist.rsample()

        # concat for actions
        acs = torch.cat([ac_embeds, torch.sigmoid(ac_pos_raw)], dim=-1)

        # get q value using current critic
        q1s = self.c1(states, acs)
        q2s = self.c2(states, acs)

        # type log probs
        type_log_probs = F.log_softmax(logits, dim=-1)
        type_log_prob = (type_probs * type_log_probs).sum(dim=-1, keepdim=True)
        type_entropy = -type_log_prob

        # pos log probs
        pos_log_prob = pos_dist.log_prob(ac_pos_raw).sum(dim=-1, keepdim=True)
        pos_entropy = pos_dist.entropy().sum(dim=-1, keepdim=True)

        # penalize for large means
        pos_means_center = 0
        pos_means_penalty = 1e-3 * ((ac_pos_means - pos_means_center) ** 2).mean()
        pos_log_std_penalty = 1e-3 * (ac_pos_stds ** 2).mean()

        # total entropy
        entropy = (
                self.log_type_alph.exp() * type_entropy +
                self.log_pos_alph.exp() * pos_entropy
        )

        # alpha loss
        type_alpha_loss = -self.log_type_alph.exp() * (
            type_entropy.detach().mean() + self.type_target_entropy
        )
        pos_alpha_loss = -self.log_pos_alph.exp() * (
            pos_entropy.detach().mean() + self.pos_target_entropy
        )

        # total log prob

        actor_loss = (
            (
                self.log_type_alph.exp() * type_log_prob +
                self.log_pos_alph.exp() * pos_log_prob
            ) - torch.minimum(q1s, q2s)
        ).mean() + pos_log_std_penalty

        self.actor_opt.zero_grad()
        self.ac_embed_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.ac_embed_opt.step()

        self.type_alph_opt.zero_grad()
        type_alpha_loss.backward()
        self.type_alph_opt.step()

        self.pos_alph_opt.zero_grad()
        pos_alpha_loss.backward()
        self.pos_alph_opt.step()

        with torch.no_grad():
            self.log_pos_alph.clamp_(np.log(1e-4), np.log(0.2))

        with torch.no_grad():
            means = ac_pos_means.mean(dim=-1)
            stds = ac_pos_stds.mean(dim=-1)
            type_alpha_mean = self.log_type_alph.exp().mean()
            pos_alpha_mean = self.log_pos_alph.exp().mean()
            type_entropy_mean = type_entropy.mean()
            pos_entropy_mean = pos_entropy.mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "type_alpha": type_alpha_mean,
            "pos_alpha": pos_alpha_mean,
            "type_alpha_loss": type_alpha_loss.item(),
            "pos_alpha_loss": pos_alpha_loss.item(),
            "type_entropy": type_entropy_mean,
            "pos_entropy": pos_entropy_mean,
            "means_x": means[0],
            "means_y": means[1],
            "stds_x": stds[0],
            "stds_y": stds[1]
        }
        return metrics

    def load_policy(self, filepath):
        params = torch.load(
            filepath, map_location=consts.DEVICE, weights_only=True
        )
        if self.player:
            self.actor.load_state_dict(params["p_actor"])
            self.ac_embed.load_state_dict(params["p_ac_embed"])

            if self.training:
                self.cnnlstm.load_state_dict(params["cnnlstm"])
                self.c1.load_state_dict(params["c1"])
                self.c2.load_state_dict(params["c2"])
                self.actor_opt.load_state_dict(["actor_opt"])
                self.c1_opt.load_state_dict(["c1_opt"])
                self.c1_opt.load_state_dict(["c2_opt"])
                self.ac_embed_opt.load_state_dict(["ac_embed_opt"])
        else:
            self.actor.load_state_dict(params["e_actor"])
            self.ac_embed.load_state_dict(params["e_ac_embed"])
