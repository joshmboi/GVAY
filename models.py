import torch
import torch.nn as nn
import torch.nn.functional as F

import consts as consts


class CNNLSTM(nn.Module):
    def __init__(
        self, feature_dim=consts.FEATURE_DIM, hidden_dim=consts.LSTM_DIM,
        state_dim=consts.STATE_DIM, pretrain=False
    ):
        super().__init__()

        # convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 200, 150)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.shape[1]

        self.fc = nn.Linear(flat_dim, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # projection layer for pretraining
        self.pretrain = pretrain
        self.proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, seq, hidden=None):
        # batch, time, channels, height, width
        B, T, C, H, W = seq.shape
        seq = seq.view(B * T, C, H, W)

        # get features from cnn
        features = self.fc(self.conv(seq))
        features = features.view(B, T, -1)

        # get state info
        lstm_out, hidden = self.lstm(self.norm(features), hidden)
        state = lstm_out[:, -1, :]

        # project to state dimension if pretraining
        if self.pretrain:
            state = self.proj(state)

        return state, hidden


class ActorModule(nn.Module):
    def __init__(
        self, input_dim=consts.LSTM_DIM, num_embed=consts.AC_EMBED_DIM,
        state_dim=consts.STATE_DIM, pretrain=False
    ):
        super().__init__()

        # if pretraining
        self.pretrain = pretrain
        self.proj = nn.Linear(state_dim, input_dim)

        # choose action embedding
        self.ac_embed = nn.Linear(input_dim, num_embed)

        # choose action position
        self.ac_pos_mean_x = nn.Linear(input_dim, 1)
        self.ac_pos_mean_y = nn.Linear(input_dim, 1)
        self.ac_pos_logstd_x = nn.Linear(input_dim, 1)
        self.ac_pos_logstd_y = nn.Linear(input_dim, 1)

    def forward(self, state):
        # project up from state during pretraining
        if self.pretrain:
            state = self.proj(state)

        # action embedding
        ac_embed = self.ac_embed(state)

        # sample position of action
        ac_pos_mean_x = self.ac_pos_mean_x(state)
        ac_pos_mean_y = self.ac_pos_mean_y(state)
        ac_pos_mean = torch.cat((ac_pos_mean_x, ac_pos_mean_y), dim=-1)

        ac_pos_std_x = torch.exp(
            torch.clamp(self.ac_pos_logstd_x(state), -4, 0.25)
        )
        ac_pos_std_y = torch.exp(
            torch.clamp(self.ac_pos_logstd_x(state), -4, 0.25)
        )
        ac_pos_std = torch.cat((ac_pos_std_x, ac_pos_std_y), dim=-1)

        # noise
        ac_pos_eps = torch.randn_like(ac_pos_std)
        ac_pos_raw = ac_pos_mean + ac_pos_std * ac_pos_eps

        return ac_embed, ac_pos_raw, (ac_pos_mean, ac_pos_std)


class AcEmbed(nn.Module):
    def __init__(
        self, num_actions=consts.NUM_ACTIONS, embed_dim=consts.AC_EMBED_DIM
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)

    def forward(self, pred_embed):
        pred_norm = F.normalize(pred_embed, dim=-1)
        embed_norm = F.normalize(self.embedding.weight, dim=-1)
        logits = torch.matmul(pred_norm, embed_norm.T)
        return logits


class Actor(nn.Module):
    def __init__(
        self, hidden_dim=consts.LSTM_DIM, num_embed=consts.AC_EMBED_DIM
    ):
        super().__init__()

        # init cnn, layer norm, and lstm
        self.cnnlstm = CNNLSTM(hidden_dim)

        # actor module
        self.ac_module = ActorModule(hidden_dim, num_embed)

    def forward(self, seq, hidden=None):
        lstm_out, features, hidden = self.cnnlstm(seq, hidden)
        last_out = lstm_out[:, -1, :]

        ac_embed, ac_pos_raw, ac_stats = self.ac_module(last_out)

        return ac_embed, ac_pos_raw, ac_stats, hidden


class CriticModule(nn.Module):
    def __init__(
        self, ac_dim=consts.AC_DIM, hidden_dim=consts.LSTM_DIM,
        state_dim=consts.STATE_DIM, pretrain=False
    ):
        super().__init__()

        # if pretraining
        self.pretrain = pretrain
        self.proj = nn.Linear(state_dim, hidden_dim)

        # critic value calculation
        self.q_calc = nn.Sequential(
            nn.Linear(hidden_dim + ac_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, ac):
        if self.pretrain:
            state = self.proj(state)

        state_ac = torch.cat([state, ac], dim=-1)

        return self.q_calc(state_ac).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, ac_dim=consts.AC_DIM, hidden_dim=consts.LSTM_DIM):
        super().__init__()

        # init cnn, layer norm, and lstm
        self.cnnlstm = CNNLSTM(hidden_dim)

        # init critic module
        self.critic_module = CriticModule(ac_dim, hidden_dim)

    def forward(self, seq, ac, hidden=None):
        lstm_out, features, hidden = self.cnnlstm(seq, hidden)
        last_out = lstm_out[:, -1, :]

        q_val = self.critic_module(last_out, ac)

        return q_val, hidden
