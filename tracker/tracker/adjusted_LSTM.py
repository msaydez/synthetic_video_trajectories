import torch.nn.utils.rnn as rnn_utils
import torch
import torch.nn as nn

# ========== Model ==========

class EncoderLSTM(nn.Module):
    def __init__(self, in_dim=4, hidden=256, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden = hidden
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_seq, (h_n, c_n) = self.lstm(packed)
        output_seq, _ = rnn_utils.pad_packed_sequence(output_seq, batch_first=True)
        return (h_n, c_n)



class DecoderLSTM(nn.Module):
    def __init__(self, bbox_input_size=4, hidden=256, output_size=4, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden

        # Embed bbox input to hidden size
        self.bbox_embedding = nn.Linear(bbox_input_size, hidden)

        # Decoder RNN (LSTM)
        self.lstm = nn.LSTM(hidden, hidden, num_layers)


        self.relu = nn.ReLU()

        # Output layer to predict next bbox
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, last_bbox_norm, hidden):
        """
        last_bbox_norm: (batch_size, bbox_input_size=4)
        hidden: tuple (h_0, c_0) from encoder, each (num_layers, batch, hidden_size)

        Returns:
          output_bbox_norm: (batch_size, output_size=4)
          hidden: updated hidden state tuple
        """
        # Embed bbox input
        embedded = self.bbox_embedding(last_bbox_norm)  # (batch_size, hidden_size)


        # Add seq dimension = 1 for LSTM input: (1, batch_size, hidden_size)
        embedded = embedded.unsqueeze(0)

        # Run LSTM for 1 step
        output, hidden = self.lstm(embedded, hidden)

        # output: (1, batch_size, hidden_size) -> squeeze seq dim
        output = output.squeeze(0)

        # Predict next normalized bbox
        output_bbox_norm = self.fc(output)
        #print('Output', output_bbox_norm)
        output_bbox_norm = output_bbox_norm #+ last_bbox_norm   # (batch_size, 4)



        return output_bbox_norm, hidden



class LSTMPositionPredictor(nn.Module):
    def __init__(self, hidden=1024, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden
        self.enc = EncoderLSTM(hidden=hidden, num_layers=num_layers)
        self.dec = DecoderLSTM(hidden=hidden, num_layers=num_layers)

    def forward(self, x, lengths, target_len):
        predictions = torch.zeros(x.shape[0], target_len, x.shape[2])
        # print(predictions.shape)

        hidden_state = self.enc(x, lengths)


        """ """
        for t in range(target_len):
            if t == 0:
                #x=x
                x = x[torch.arange(x.shape[0]), torch.tensor(lengths) - 1]
            else:
                x = output
            #print(x.shape, hidden_state.shape)
            #print('Hidden States',x, hidden_state)
            output, hidden_state = self.dec(x, hidden_state)

            predictions[:, t, :] = output



        return predictions, output, hidden_state
