"""
Training utilities for the hybrid quantum-classical classifier.
"""

import copy

import torch

from src.training.history import TrainingHistory
from src.evaluation.metrics import compute_metrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device="cpu",
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.history = TrainingHistory()

        self.best_model = None

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch using mini-batches.

        Parameters
        ----------
        train_loader : DataLoader
            Mini-batch training data.

        Returns
        -------
        float
            Average training loss.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Clear gradients from previous batch
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_batch)

            # Calculate loss
            loss = self.criterion(outputs, y_batch)

            # Backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Accumulate loss
            batch_size = X_batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        average_loss = total_loss / total_samples

        return average_loss

    def evaluate(self, data_loader):
        """
        Evaluate the model without updating its parameters.

        Parameters
        ----------
        data_loader : DataLoader
            Validation or test data.

        Returns
        -------
        float
            Average loss.

        dict
            Classification metrics.
        """

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():

            for X_batch, y_batch in data_loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)

                # Calculate loss
                loss = self.criterion(outputs, y_batch)

                batch_size = X_batch.size(0)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Convert logits to binary predictions
                predictions = (
                    torch.sigmoid(outputs) >= 0.5
                ).float()

                all_labels.extend(
                    y_batch.cpu().numpy().flatten()
                )

                all_predictions.extend(
                    predictions.cpu().numpy().flatten()
                )

        average_loss = total_loss / total_samples

        metrics = compute_metrics(
            all_labels,
            all_predictions
        )

        return average_loss, metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=30,
    ):
        """
        Train the model using mini-batches.

        The model with the lowest validation loss is saved
        as the best model.

        Parameters
        ----------
        train_loader : DataLoader
            Mini-batch training data.

        val_loader : DataLoader
            Validation data.

        epochs : int
            Number of training epochs.

        Returns
        -------
        TrainingHistory
            Recorded training history.
        """

        best_val_loss = float("inf")

        for epoch in range(epochs):

            # -------------------------
            # Training
            # -------------------------

            train_loss = self.train_epoch(
                train_loader
            )

            # -------------------------
            # Validation
            # -------------------------

            val_loss, val_metrics = self.evaluate(
                val_loader
            )

            # -------------------------
            # Training accuracy
            # -------------------------

            _, train_metrics = self.evaluate(
                train_loader
            )

            train_accuracy = train_metrics["accuracy"]
            val_accuracy = val_metrics["accuracy"]

            # -------------------------
            # Save history
            # -------------------------

            self.history.add_epoch(
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_accuracy,
                val_acc=val_accuracy,
            )

            # -------------------------
            # Save best model
            # -------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                self.best_model = copy.deepcopy(
                    self.model.state_dict()
                )

            # -------------------------
            # Progress output
            # -------------------------

            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        # Restore best model after training
        if self.best_model is not None:

            self.model.load_state_dict(
                self.best_model
            )

        return self.history
