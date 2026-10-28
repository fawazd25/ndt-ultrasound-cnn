"""Le réseau de neurones : un CNN 1D (convolutions sur un signal)."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Entrée  : (lot, 1, 512)  un signal par ligne
    Sortie  : (lot,)         un "logit" par signal (grand = défaut probable)

    4 blocs [convolution -> ReLU -> max-pooling] qui détectent des motifs de plus
    en plus larges (une oscillation, un écho, un écho + son contexte...), puis une
    moyenne globale et une couche linéaire qui décide.
    """

    def __init__(self, canaux=(8, 16, 32, 32, 32), noyau=15):
        super().__init__()
        couches, c_in = [], 1
        for c_out in canaux:
            couches += [nn.Conv1d(c_in, c_out, noyau, padding=noyau // 2),
                        nn.ReLU(),
                        nn.MaxPool1d(2)]
            c_in = c_out
        self.extracteur = nn.Sequential(*couches)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x):
        x = self.extracteur(x)          # (lot, C, L')
        x = x.mean(dim=-1)              # moyenne globale -> (lot, C)
        return self.fc(x).squeeze(-1)   # (lot,)
