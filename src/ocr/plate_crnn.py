"""lightweight crnn model for license plate text recognition.

architecture:
- cnn feature extractor (small convnet)
- bidirectional lstm sequence model
- ctc output layer
"""

import torch
import torch.nn as nn


# character set used for encoding/decoding
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0  # ctc blank
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for ctc blank


def encode_text(text: str) -> list[int]:
    """encode plate text to index sequence."""
    return [CHAR_TO_IDX[c] for c in text.upper() if c in CHAR_TO_IDX]


def decode_output(indices: list[int]) -> str:
    """decode ctc output indices to text (greedy, collapse repeats + remove blanks)."""
    result = []
    prev = BLANK_IDX
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            if idx in IDX_TO_CHAR:
                result.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(result)


class PlateRecCRNN(nn.Module):
    """small crnn for plate recognition.

    input: grayscale image resized to (1, img_h, img_w)
    output: (seq_len, batch, num_classes) log probabilities
    """

    def __init__(self, img_h: int = 32, num_classes: int = NUM_CLASSES, hidden_size: int = 128) -> None:
        super().__init__()
        self.img_h = img_h

        # cnn feature extractor
        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # h/2
            # block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # h/4
            # block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # h/8, w unchanged
            # block 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # after cnn: (batch, 128, h/8, w)
        # collapse height dimension to get (batch, 128 * h/8, w)
        cnn_out_h = img_h // 8
        rnn_input_size = 128 * cnn_out_h

        # bidirectional lstm
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )

        # output projection
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, h, w)
        features = self.cnn(x)  # (batch, 128, h/8, w')

        b, c, h, w = features.size()
        # reshape: collapse c and h -> (batch, c*h, w)
        features = features.view(b, c * h, w)
        # permute to (w, batch, c*h) for rnn (seq_len, batch, features)
        features = features.permute(2, 0, 1)

        rnn_out, _ = self.rnn(features)  # (w, batch, hidden*2)
        output = self.fc(rnn_out)  # (w, batch, num_classes)

        return output.log_softmax(dim=2)
