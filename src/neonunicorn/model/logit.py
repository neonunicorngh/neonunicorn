"""
logit.py
----------
Custom implementation of Logistic Regression using PyTorch
for binary classification tasks (Bonus: Question 6).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LogisticRegressionModel:
    """
    A simple logistic regression implementation optimized
    with stochastic gradient descent (SGD) using PyTorch tensors.

    Parameters
    ----------
    n_features : int
        Number of input features.
    lr : float, optional (default=0.01)
        Learning rate for SGD.
    epochs : int, optional (default=1000)
        Number of training epochs.
    pos_weight : float, optional (default=1.0)
        Weight for positive samples (to handle imbalance).
    """

    def __init__(self, n_features, lr=0.01, epochs=1000, pos_weight=1.0):
        self.lr = lr
        self.epochs = epochs
        self.model = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss(weight=torch.tensor([pos_weight]))
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_history = []

    def fit(self, X_train, y_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            y_pred = self.sigmoid(self.model(X_train))
            loss = self.loss_fn(y_pred, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

        return loss.item()

    def predict_proba(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.sigmoid(self.model(X_test))
        return y_pred.numpy()

    def predict(self, X_test, threshold=0.5):
        probs = self.predict_proba(X_test)
        return (probs >= threshold).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
