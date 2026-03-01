#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


GAMES = ["boring", "calm", "horror", "funny"]


def load_dataset():

    root_dir = os.path.expanduser("~/GAMEEMO")

    X = []
    y = []

    for subject_folder in os.listdir(root_dir):
        if not subject_folder.startswith("("):
            continue

        csv_path = os.path.join(
            root_dir,
            subject_folder,
            "Preprocessed EEG Data",
            ".csv format"
        )

        if not os.path.exists(csv_path):
            continue

        for filename in os.listdir(csv_path):
            if not filename.endswith(".csv"):
                continue

            filepath = os.path.join(csv_path, filename)
            df = pd.read_csv(filepath)

            df = df.select_dtypes(include=[np.number])
            df = df.fillna(df.mean())
            df = df.fillna(0)

            means = df.mean().values
            stds = df.std().values
            stds = np.where(stds == 0, 1e-6, stds)

            features = np.concatenate([means, stds])
            features = np.nan_to_num(features)

            game_number = int(filename.split("G")[1][0])
            label = game_number - 1

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Global normalization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1e-6, X_std)
    X = (X - X_mean) / X_std

    return X, y


def build_model(input_size, architecture):

    layers = []
    prev_size = input_size

    for hidden_size in architecture:
        layers.append(torch.nn.Linear(prev_size, hidden_size))
        layers.append(torch.nn.ReLU())
        prev_size = hidden_size

    layers.append(torch.nn.Linear(prev_size, 4))
    layers.append(torch.nn.LogSoftmax(dim=1))

    return torch.nn.Sequential(*layers)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epoch = 100

    start_train = time.time()

    for epoch in range(n_epoch):

        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = torch.nn.functional.nll_loss(logits, y_train)

        loss.backward()
        optimizer.step()

    end_train = time.time()
    train_time = end_train - start_train

    # Final train accuracy
    model.eval()
    with torch.no_grad():
        logits_train = model(X_train)
        preds_train = torch.argmax(logits_train, dim=1)
        train_acc = accuracy_score(
            y_train.cpu().numpy(),
            preds_train.cpu().numpy()
        )

    # Test
    start_test = time.time()
    with torch.no_grad():
        logits_test = model(X_test)
        preds_test = torch.argmax(logits_test, dim=1)
        test_acc = accuracy_score(
            y_test.cpu().numpy(),
            preds_test.cpu().numpy()
        )
    end_test = time.time()
    test_time = end_test - start_test

    return train_acc, test_acc, train_time, test_time


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X, y = load_dataset()
    print("Dataset shape:", X.shape)

    np.random.seed(123)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    input_size = X.shape[1]

    architectures = [
        [64],
        [128],
        [128, 64],
        [256, 128],
        [256, 128, 64],
        [512, 256, 128],
    ]

    results = []

    for i, arch in enumerate(architectures):

        print("\n====================================")
        print(f"Case {i+1} Architecture: {arch}")
        print("====================================")

        model = build_model(input_size, arch).to(device)

        train_acc, test_acc, train_time, test_time = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, device
        )

        results.append([str(arch), train_acc, test_acc, train_time, test_time])

    # Convert to DataFrame
    results_df = pd.DataFrame(
        results,
        columns=["Architecture", "Train_Accuracy", "Test_Accuracy", "Train_Time", "Test_Time"]
    )

    print("\n\nFINAL COMPARISON TABLE")
    print(results_df)

    # Save table to CSV
    results_df.to_csv("model_comparison_results.csv", index=False)

    # --------------------
    # Generate Plots
    # --------------------

    model_indices = range(1, len(results_df) + 1)

    # Plot 1: Train Accuracy
    plt.figure()
    plt.plot(model_indices, results_df["Train_Accuracy"], marker="o")
    plt.title("Train Accuracy vs Model Size")
    plt.xlabel("Model Case")
    plt.ylabel("Train Accuracy")
    plt.xticks(model_indices)
    plt.savefig("train_accuracy_plot.png")
    plt.close()

    # Plot 2: Test Accuracy
    plt.figure()
    plt.plot(model_indices, results_df["Test_Accuracy"], marker="o")
    plt.title("Test Accuracy vs Model Size")
    plt.xlabel("Model Case")
    plt.ylabel("Test Accuracy")
    plt.xticks(model_indices)
    plt.savefig("test_accuracy_plot.png")
    plt.close()

    # Plot 3: Train Time
    plt.figure()
    plt.plot(model_indices, results_df["Train_Time"], marker="o")
    plt.title("Train Time vs Model Size")
    plt.xlabel("Model Case")
    plt.ylabel("Train Time (seconds)")
    plt.xticks(model_indices)
    plt.savefig("train_time_plot.png")
    plt.close()

    # Plot 4: Test Time
    plt.figure()
    plt.plot(model_indices, results_df["Test_Time"], marker="o")
    plt.title("Test Time vs Model Size")
    plt.xlabel("Model Case")
    plt.ylabel("Test Time (seconds)")
    plt.xticks(model_indices)
    plt.savefig("test_time_plot.png")
    plt.close()

    print("\nAll plots saved as PNG files.")


if __name__ == "__main__":
    main()
