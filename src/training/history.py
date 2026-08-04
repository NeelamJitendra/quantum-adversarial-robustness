"""
Training history container.
"""


class TrainingHistory:

    def __init__(self):

        self.train_loss = []
        self.val_loss = []

        self.train_accuracy = []
        self.val_accuracy = []

    def add_epoch(
        self,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self.train_accuracy.append(train_acc)
        self.val_accuracy.append(val_acc)