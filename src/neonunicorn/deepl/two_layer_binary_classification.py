import torch
import torch.nn.functional as F


def _init_weight(shape, fan_in, device):
    # std = sqrt(2/fan_in)
    std = (2.0 / fan_in) ** 0.5
    return torch.nn.Parameter(
        torch.randn(*shape, device=device, dtype=torch.float32) * std
    )


def binary_classification(d, n, epochs=10000, h=0.001):
    """
    Train a binary classifier using gradient descent + PyTorch autograd.

    Args:
        d (int): number of features
        n (int): number of samples
        epochs (int): default 10000
        h (float): learning rate, default 0.001

    Returns:
        W1, W2, W3, W4 (trained weight tensors on CPU),
        loss_history (list of loss values for each epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Generate random feature matrix X: (n, d), Float32
    X = torch.randn(n, d, dtype=torch.float32, device=device)

    # 2) Generate labels Y: (n, 1), rule: sum(features) > 2 => 1 else 0
    Y = (X.sum(dim=1, keepdim=True) > 2).float()

    # 3) Initialize weights with std sqrt(2/fan_in)
    W1 = _init_weight((d, 48),  fan_in=d,  device=device)
    W2 = _init_weight((48, 16), fan_in=48, device=device)
    W3 = _init_weight((16, 32), fan_in=16, device=device)
    W4 = _init_weight((32, 1),  fan_in=32, device=device)

    optimizer = torch.optim.SGD([W1, W2, W3, W4], lr=h)

    loss_history = []

    for _ in range(epochs):
        optimizer.zero_grad()

        Z1 = X @ W1                 # (n,48)
        A1 = torch.sigmoid(Z1 @ W2) # (n,16)
        Z2 = A1 @ W3                # (n,32)
        Yhat = torch.sigmoid(Z2 @ W4)  # (n,1)

        loss = F.binary_cross_entropy(Yhat, Y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return W1.detach().cpu(), W2.detach().cpu(), W3.detach().cpu(), W4.detach().cpu(), loss_history
