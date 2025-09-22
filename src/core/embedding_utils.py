import numpy as np
from typing import Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_embeddings_2d(embeddings: Union[np.ndarray, "torch.Tensor"], method: str = "tsne") -> np.ndarray:
    """Reduce embeddings to 2D for visualisation."""
    # Convert PyTorch tensor to numpy array, detaching from gradient computation
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    elif hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()

    if method == "pca":
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings)
    elif method == "tsne":
        # Set perplexity based on sample size (must be < n_samples)
        n_samples = embeddings.shape[0]
        perplexity = min(30.0, n_samples - 1)  # Default 30, but cap at n_samples-1
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        return tsne.fit_transform(embeddings)
    else:
        raise ValueError(f"Invalid reduction method: {method}. Supported methods: 'pca', 'tsne'")
