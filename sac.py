import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 200, 150)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.shape[1]

        self.feature_dim = 128

        self.fc = nn.Linear(flat_dim, self.feature_dim)

        self.norm = nn.LayerNorm(self.feature_dim)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, seq, hidden=None):
        # batch, time, channels, height, width
        B, T, C, H, W = seq.shape
        seq = seq.view(B * T, C, H, W)

        # get features from cnn
        features = self.fc(self.conv(seq))
        features = features.view(B, T, -1)

        lstm_out, hidden = self.lstm(self.norm(features), hidden)
        return lstm_out, features, hidden


class Actor(nn.Module):
    def __init__(self, hidden_dim=256, num_embed=4, num_vals=2):
        super().__init__()

        # init cnn, layer norm, and lstm
        self.cnnlstm = CNNLSTM(hidden_dim)

        # choose action embedding
        self.ac_embed = nn.Linear(hidden_dim, num_embed)

        # choose action position
        self.ac_pos_mean = nn.Linear(hidden_dim, num_vals)
        self.ac_pos_logstd = nn.Linear(hidden_dim, num_vals)

    def forward(self, seq, hidden=None):
        lstm_out, features, hidden = self.cnnlstm(seq, hidden)
        last_out = lstm_out[:, -1, :]

        # action embedding
        ac_embed = self.ac_embed(last_out)

        # sample position of action
        ac_pos_mean = torch.sigmoid(self.ac_pos_mean(last_out))
        ac_pos_std = torch.exp(
            torch.clamp(self.ac_pos_logstd(last_out), -4, 0.5)
        )

        # noise
        ac_pos_eps = torch.randn_like(ac_pos_std)
        ac_pos = torch.clamp(ac_pos_mean + ac_pos_std * ac_pos_eps, 0.0, 1.0)

        return ac_embed, ac_pos, (ac_pos_mean, ac_pos_std), hidden


class Critic(nn.Module):
    def __init__(self, ob_dim=256, ac_dim=6, hidden_dim=256):
        super().__init__()

        # init cnn, layer norm, and lstm
        self.cnnlstm = CNNLSTM(ob_dim)

        # critic value calculation
        self.q_calc = nn.Sequential(
            nn.Linear(ob_dim + ac_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, seq, ac, hidden=None):
        lstm_out, features, hidden = self.cnnlstm(seq, hidden)
        last_out = lstm_out[:, -1, :]

        x = torch.cat([last_out, ac], dim=-1)
        return self.q_calc(x).squeeze(-1), hidden


class AcEmbed(nn.Module):
    def __init__(self, num_actions=5, embed_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)

    def forward(self, pred_embed):
        pred_norm = F.normalize(pred_embed, dim=-1)
        embed_norm = F.normalize(self.embedding.weight, dim=-1)
        sims = torch.matmul(pred_norm, embed_norm.T)
        return sims

    def select_action(self, pred_embed):
        sims = self.forward(pred_embed)
        probs = F.softmax(sims, dim=-1)
        action = probs.multinomial(1).squeeze(-1)
        return action
