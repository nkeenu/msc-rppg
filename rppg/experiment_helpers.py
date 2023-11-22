import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set()


def train_model(model, train_loader, val_loader, criterion, optimiser, num_epochs, device="cpu"):
    """
    Train the model and return the best model, training and validation losses.
    :param model: Model to train
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param criterion: Loss function
    :param optimiser: Optimiser
    :param num_epochs: Number of epochs to train for
    :param device: Device to train on
    :return: Best model, training and validation losses
    """
    best_model = None
    best_val_loss = 1e10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {train_loss} - Validation loss: {val_loss}")

    return best_model, train_losses, val_losses


def test_model(model, test_loader, criterion, device="cpu"):

    model.eval()
    test_loss = 0.0
    predictions = []
    labels = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(inputs)
            loss = criterion(predictions, labels)

        test_loss += loss.item() * inputs.size(0)
        predictions.append(predictions.cpu().numpy())
        labels.append(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    return test_loss, predictions, labels


def plot_losses(train_losses, val_losses):
    
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss (bpm)")
    plt.legend()
    plt.show()

    # Print the minimum validation loss
    print(f"Minimum validation loss: {min(val_losses)}")


def plot_predictions(labels, predictions, clip=False):
    if clip:
        predictions = [hr for clip in predictions for hr in clip]
        labels = [hr for clip in labels for hr in clip]
    else:
        predictions = predictions.flatten()
        labels = labels.flatten()

    r = pearsonr(labels, predictions)[0]
    # Diagonal line representing perfect correlation
    line_x = np.linspace(min(labels), max(labels), 100)
    line_y = line_x
    
    plt.figure(figsize=(5, 4))
    plt.scatter(labels, predictions, s=1)
    plt.plot(line_x, line_y, color=sns.color_palette()[3], linestyle='--')
    plt.xlabel("True heart rate (bpm)")
    plt.ylabel("Predicted heart rate (bpm)")

    bbox_style = dict(boxstyle="round, pad=0.3", facecolor="#FFFF99", alpha=0.5)
    plt.text(0.95, 0.05, f'r = {r:.3f}', transform=plt.gca().transAxes, va="bottom", ha="right", bbox=bbox_style)

    plt.show()

