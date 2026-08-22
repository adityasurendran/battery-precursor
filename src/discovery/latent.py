"""Latent state discovery: learn representation, find transition."""

from __future__ import annotations

import numpy as np
from typing import Optional


class LatentDiscovery:
    """Learn latent representation and detect state transitions."""

    def __init__(self, n_components: int = 5, window: int = 50):
        self.n_components = n_components
        self.window = window

    def fit_pca(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return transformed data."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(self.n_components, X.shape[1]))
        Z = pca.fit_transform(X)
        return Z, pca.explained_variance_ratio_

    def fit_autoencoder(self, X: np.ndarray, epochs: int = 50) -> tuple:
        """Fit simple autoencoder and return latent representation."""
        try:
            import torch
            import torch.nn as nn

            input_dim = X.shape[1]
            latent_dim = min(self.n_components, input_dim // 2)

            class Autoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, latent_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, input_dim),
                    )

                def forward(self, x):
                    z = self.encoder(x)
                    return self.decoder(z), z

            model = Autoencoder()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            X_tensor = torch.tensor(X, dtype=torch.float32)
            for _ in range(epochs):
                recon, z = model(X_tensor)
                loss = loss_fn(recon, X_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, Z = model(X_tensor)
            return Z.numpy(), model

        except ImportError:
            # Fallback to PCA
            return self.fit_pca(X)

    def detect_transition(self, Z: np.ndarray, window: int = 50) -> dict:
        """Detect state transition in latent space using CUSUM."""
        if Z.ndim > 1:
            # Use first principal component
            signal = Z[:, 0]
        else:
            signal = Z

        # CUSUM on mean of latent dimensions
        if Z.ndim > 1:
            signal = np.mean(Z, axis=1)

        mean = np.mean(signal[:min(window, len(signal))])
        cusum_pos = 0.0
        cusum_neg = 0.0
        threshold = 3.0

        transitions = []
        for i in range(len(signal)):
            diff = signal[i] - mean
            cusum_pos = max(0, cusum_pos + diff)
            cusum_neg = max(0, cusum_neg - diff)
            if cusum_pos > threshold or cusum_neg > threshold:
                transitions.append(i)
                cusum_pos = 0
                cusum_neg = 0

        return {
            "transitions": transitions,
            "first_transition": transitions[0] if transitions else -1,
            "n_transitions": len(transitions),
            "latent_signal": signal.tolist() if len(signal) < 1000 else signal[::10].tolist(),
        }
