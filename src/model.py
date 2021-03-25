from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = torch.nn.GRU(
            input_size=embeddings.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0),
            bidirectional=bidirectional,
        )
        self.batchnorm = torch.nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = torch.nn.Linear(hidden_size * 2, 256)
        self.fc2 = torch.nn.Linear(256, num_class)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeds = self.embed(batch)
        x, hidden = self.gru(embeds.transpose(0, 1))
        x = self.batchnorm(x.transpose(0, 1)[:, -1, :])
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return {"labels": x}
