import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    """Compute cross-entropy loss."""
    probs = softmax(logits)
    return -np.log(probs[target] + 1e-9)


def sgd_update(param: np.ndarray, grad: np.ndarray, lr: float) -> None:
    """Perform SGD parameter update."""
    param -= lr * grad


def perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return np.exp(loss)
