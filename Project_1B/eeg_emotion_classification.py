#!/usr/bin/env python
# coding: utf-8

# Emotion recognition using EEG and computer games (+micro:bit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time

# Constants
SAMPLE_RATE = 32  # (Hz)
GAMES = ["boring", "calm", "horror", "funny"]

def main():

    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------
    # Read the data
    # --------------------
    data = []
    for game_id, game in enumerate(GAMES):
        game_data = pd.read_csv(os.path.join("data", f"S01G{game_id + 1}AllChannels.csv"))
        game_data["game"] = game
        data.append(game_data)

    data = pd.concat(data, axis=0, ignore_index=True)

    # --------------------
    # Choose electrode
    # --------------------
    electrode = "T7"

    # Plot and save instead of showing (for remote environments)
    fig, ax = plt.subplots(1, 1)
    for game in GAMES:
        ax.plot(data[data["game"] == game][electrode], label=game)

    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("mV")
    ax.legend()
    plt.savefig("signal_plot.png")
    plt.close()

    # --------------------
    # Create dataset
    # --------------------
    data = data[[electrode, "game"]]

    clip_length = 2  # seconds

    clipped_data = []
    y = []
    for game_id, game in enumerate(GAMES):
        game_signal = data[data['game'] == game][electrode].to_numpy()

        clips = np.array_split(
            game_signal,
            len(game_signal) // (clip_length * SAMPLE_RATE)
        )

        clipped_data.extend(clips)
        y.extend([game_id] * len(clips))

    # Remove edge effects
    min_length = np.min([len(arr) for arr in clipped_data])
    X = []
    for array in clipped_data:
        X.append(array[:min_length])

    X = np.vstack(X, dtype=float)
    y = np.array(y, dtype=int)

    print("Dataset shape:", X.shape, y.shape)

    # --------------------
    # Train/Test Split
    # --------------------
    np.random.seed(123)

    X = np.expand_dims(X, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    # --------------------
    # Dataset class
    # --------------------
    class LFPDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y.long()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        LFPDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        LFPDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------
    # Model
    # --------------------
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 1, kernel_size=4, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv1d(1, 1, kernel_size=4, padding="same"),
        torch.nn.Flatten(),
        torch.nn.Linear(min_length, 4),
        torch.nn.LogSoftmax(dim=1)
    )

    model = model.to(device)

    # --------------------
    # Training function
    # --------------------
    def train_model(n_epoch, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for e in range(n_epoch):
            model.train()

            train_loss = []
            train_acc = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = torch.nn.functional.nll_loss(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                prediction = torch.argmax(logits, dim=1)
                train_acc.append(
                    accuracy_score(
                        y_batch.cpu().numpy(),
                        prediction.cpu().numpy()
                    )
                )

            # Evaluation
            model.eval()
            test_loss = []
            test_acc = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    logits = model(X_batch)
                    loss = torch.nn.functional.nll_loss(logits, y_batch)

                    test_loss.append(loss.item())

                    prediction = torch.argmax(logits, dim=1)
                    test_acc.append(
                        accuracy_score(
                            y_batch.cpu().numpy(),
                            prediction.cpu().numpy()
                        )
                    )

            print(
                f"Epoch {e} | "
                f"train_loss={np.mean(train_loss):.4f}, "
                f"train_acc={np.mean(train_acc):.4f}, "
                f"test_loss={np.mean(test_loss):.4f}, "
                f"test_acc={np.mean(test_acc):.4f}"
            )

        return model

    # --------------------
    # Training with timing
    # --------------------
    start_time = time.time()

    train_model(n_epoch=100, model=model)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")

    # --------------------
    # Example prediction
    # --------------------
    a_clip = torch.tensor(X[0]).unsqueeze(0).to(device).float()

    model.eval()
    with torch.no_grad():
        prediction = model(a_clip)
        prediction = torch.argmax(prediction, dim=1).item()

    print("Predicted emotion:", GAMES[prediction])


if __name__ == "__main__":
    main()
