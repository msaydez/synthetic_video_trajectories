import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import generalized_box_iou
import torch
import torch.nn as nn
#from tracker.Social_lstm import SocialModel


# ========== Model ==========
class EncoderLSTM(nn.Module):
    def __init__(self, in_dim=4, hidden=256, num_layers=1, bidirectional=True):
        super().__init__()
        self.inlstm = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, lengths):
        #packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_seq, hidden_state = self.lstm(x)
        #print('Output Sequence',output_seq)
        return hidden_state  # torch.cat((h[-2], h[-1]), dim=1)  h[-1]


class DecoderLSTM(nn.Module):
    def __init__(self, hidden=64, out_dim=4):
        super().__init__()
        self.bilstm = nn.LSTM(out_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(hidden, hidden//2),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden//2, hidden//4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, lengths, h):
        # h: [B, hidden], m: [B]
        h_n, c_n = h
        # x = x[torch.arange(x.shape[0]), torch.tensor(lengths)-1]
        #print('Input Shape',x.shape, h_n.shape, c_n.shape)
        output, hidden_state = self.bilstm(x.unsqueeze(1),(h_n, c_n))
        return output, hidden_state


class GaussianHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(hidden, hidden//2),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden//2, hidden//4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features, 8))
        # self.fc = nn.Linear(in_features, 8)

    def forward(self, x, last_bbox_norm):
        out = self.fc(x.squeeze(1))
        """mu = out[:, :4]
        sigma_raw = out[:, 4:6]
        rho_raw = out[:, 6]"""

        mu = out[:, :4]
        sigma_raw = out[:, 4:8]
        rho_raw = out[:, 6]

        sigma = torch.exp(sigma_raw)  # enforce positive std
        rho = torch.tanh(rho_raw)  # clamp to [-1, 1]

        return mu, sigma, rho


class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 4)

    def forward(self, in_features):
        out = self.fc(in_features.squeeze(1))
        return out


class LSTMPositionPredictor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.enc = EncoderLSTM(hidden=hidden)
        self.dec = DecoderLSTM(hidden=hidden)
        self.gauss_head = GaussianHead(in_features=hidden*2)
        self.head = LinearHead(in_features=hidden)

    def forward(self, x, lengths, target_len):
        predictions = torch.zeros(x.shape[0], target_len, x.shape[2])
        # print(predictions.shape)
        sigma_l = torch.zeros(x.shape[0], target_len, x.shape[2])
        rho = torch.zeros(x.shape[0], target_len, x.shape[2])
        #print(x.shape ,x[:-1].shape)
        hidden_state = self.enc(x, lengths)

        x = x[torch.arange(x.shape[0]), torch.tensor(lengths) - 1]
        #print(x)
        for t in range(target_len):

            output, hidden_state = self.dec(x, lengths, hidden_state)
            pred, sigma, rho = self.gauss_head(output, x)

            predictions[:, t, :] = pred
            sigma_l[:, t, :] = sigma

            x = pred

        return predictions, sigma_l, rho

    """def forward(self, x, lengths, target_len, warmup_steps=5):
        batch_size, seq_len, feature_dim = x.shape

        predictions = torch.zeros(batch_size, target_len+warmup_steps, feature_dim, device=x.device)
        sigma_l = torch.zeros(batch_size, target_len+warmup_steps, feature_dim, device=x.device)
        rho = torch.zeros(batch_size, target_len+warmup_steps, feature_dim, device=x.device)

        # Encode past sequence
        hidden_state = self.enc(x, lengths)

        # Start warm-up from last few GT positions
        # Use min(warmup_steps, seq_len) to avoid indexing errors
        warmup_range = min(warmup_steps, seq_len)
        warmup_inputs = x[:, -warmup_range:, :]  # shape: [B, warmup_steps, 4]

        # Step through warm-up
        for t in range(warmup_range):
            decoder_input = warmup_inputs[:, t, :]
            output, hidden_state = self.dec(decoder_input, lengths, hidden_state)
            output, sigma, rho_val = self.gauss_head(output, decoder_input)
            print('Decoder input', decoder_input, 'Output', output)

            predictions[:, t, :] = output
            sigma_l[:, t, :] = sigma
            rho[:, t, :] = rho_val

        # Autoregressive prediction from warmup_steps → target_len
        for t in range(warmup_range, target_len):
            decoder_input = predictions[:, t - 1, :].detach()  # last predicted step
            output, hidden_state = self.dec(decoder_input, lengths, hidden_state)
            output, sigma, rho_val = self.gauss_head(output, decoder_input)

            predictions[:, t, :] = output
            sigma_l[:, t, :] = sigma
            rho[:, t, :] = rho_val

        return predictions, sigma_l, rho"""


"""class LSTMPositionPredictor(nn.Module):
    def __init__(self, args):
        super(LSTMPositionPredictor, self).__init__()
        self.args = args
        self.enc = SocialModel(args, infer=False)
        self.dec = DecoderLSTM(hidden=args.rnn_size)
        self.gauss_head = GaussianHead(in_features=args.rnn_size)
        self.head = LinearHead(in_features=args.rnn_size)

    def forward(self, input_seq, grid_seq, hidden_states, cell_states):
        
        input_seq: [batch_size, seq_len, num_peds, input_size]
        grid_seq: [batch_size, seq_len, num_peds, grid_size*grid_size]
        hidden_states: [batch_size, num_peds, rnn_size]
        cell_states: [batch_size, num_peds, rnn_size] or None
        
        batch_size, seq_len, num_peds, _ = input_seq.shape

        # Encoder: Social LSTM
        encoder_outputs, hidden_states, cell_states = self.enc(input_seq, grid_seq, hidden_states, cell_states)

        # Aggregate hidden states to initialize decoder (e.g., mean over agents)
        h_agg = hidden_states.mean(dim=1).unsqueeze(0)  # [1, batch_size, rnn_size]
        c_agg = None
        if not self.args.gru:
            c_agg = cell_states.mean(dim=1).unsqueeze(0)  # [1, batch_size, rnn_size]

        x_last = input_seq[:, -1, 0, :]  # last position of first pedestrian [batch_size, input_size]
        x = x_last.unsqueeze(1)  # [batch_size, 1, input_size]

        predictions = torch.zeros(batch_size, self.args.target_len, self.args.output_size, device=input_seq.device)
        sigma_l = torch.zeros_like(predictions)
        rho = torch.zeros_like(predictions)

        hidden = (h_agg, c_agg) if not self.args.gru else h_agg

        for t in range(self.args.target_len):
            output, hidden = self.dec(x.squeeze(1), [1] * batch_size, hidden)
            if self.args.nll:
                output, sigma, rho_val = self.gauss_head(output)
            else:
                output = self.head(output)

            predictions[:, t, :] = output
            if self.args.nll:
                sigma_l[:, t, :] = sigma
                rho[:, t, :] = rho_val.unsqueeze(-1).expand_as(output)

            x = output.unsqueeze(1)

        if self.args.nll:
            return predictions, sigma_l, rho
        else:
            return predictions"""

