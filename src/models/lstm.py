from torch import nn
import torch

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        num_classes,
        device,
        num_channels=3,
        feature_extractor=None,
        use_all_hidden=False,
        sequence_length=1,
        bidireactional=False,
        num_layers=1,
        dropout=0.0,
        proj_size=0,
    ):
        """

        Args:
            input_size:
            hidden_dim:
            num_classes:
            device:
            feature_extractor: If given, must have output size with length <input_size>
            use_all_hidden:
            sequence_length:
            bidireactional:
            num_layers:
        """
        super().__init__()
        self.input_size = input_size
        self.h_cell = hidden_dim
        self.h_out = hidden_dim if proj_size == 0 else proj_size
        self.num_layers = num_layers
        self.device = device
        self.bidireactional = bidireactional
        self.d_param = 2 if bidireactional else 1
        self.sequence_length = sequence_length
        self.use_all_hidden = use_all_hidden
        self.feature_extractor = feature_extractor
        self.num_channels = num_channels
        if self.feature_extractor is not None:
            self.feature_extractor = self.feature_extractor.to(self.device)

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidireactional,
            batch_first=True,
            dropout=dropout,
        )
        output_size = (
            hidden_dim * self.d_param
            if not use_all_hidden
            else hidden_dim * self.d_param * sequence_length
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        if self.feature_extractor is not None:
            # First pass through the feature extractor
            self.feature_extractor = self.feature_extractor.to(self.device)
            # x should be of size (batch_size, window_size) + <feature_extractor input size>
            x_out = torch.zeros(
                [batch_size, self.sequence_length, self.input_size], dtype=torch.float32
            )
            x_out = x_out.to(self.device)
            for i in range(self.sequence_length):
                x_out[:, i, :] = self.feature_extractor(
                    x[:, i : (i + self.num_channels), :]
                )
            x = x_out
            x = x.to(self.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        # x should be of size (batch_size, window_size, input_size)
        out, hidden = self.lstm(x, (hidden, cell))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # Output has format (batch_size, self.windows_size, self.d_param*self.hidden_dim)
        out = out.reshape(batch_size, self.sequence_length, self.d_param * self.h_out)
        h_n = out[:, -1, :]

        if not self.use_all_hidden:
            h_n = h_n.reshape(batch_size, -1)
            output = self.linear_classifier(h_n)
        else:
            out = out.reshape(batch_size, -1)
            output = self.linear_classifier(out)
        return output

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.d_param * self.num_layers, batch_size, self.h_out)
        hidden = hidden.to(self.device)
        return hidden

    def init_cell(self, batch_size):
        cell = torch.zeros(self.d_param * self.num_layers, batch_size, self.h_cell)
        cell = cell.to(self.device)
        return cell