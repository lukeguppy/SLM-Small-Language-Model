import unittest
import numpy as np

from ..core.utils import softmax, cross_entropy_loss, sgd_update, perplexity
from ..core.embedding_utils import reduce_embeddings_2d


class TestUtils(unittest.TestCase):
    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
        # Test that it's increasing
        self.assertTrue(np.all(np.diff(result) >= 0))

    def test_cross_entropy_loss(self):
        logits = np.array([1.0, 2.0, 3.0])
        target = 1
        loss = cross_entropy_loss(logits, target)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

        # Test with perfect prediction
        perfect_logits = np.array([0.0, 100.0, 0.0])
        perfect_loss = cross_entropy_loss(perfect_logits, 1)
        self.assertLess(perfect_loss, 0.01)

    def test_sgd_update(self):
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        lr = 0.01
        original_param = param.copy()
        sgd_update(param, grad, lr)
        expected = original_param - lr * grad
        np.testing.assert_array_equal(param, expected)

    def test_perplexity(self):
        loss = 1.0
        ppl = perplexity(loss)
        self.assertAlmostEqual(ppl, np.exp(1.0))

        loss = 0.0
        ppl = perplexity(loss)
        self.assertAlmostEqual(ppl, 1.0)

    def test_reduce_embeddings_2d_pca(self):
        """Test PCA-based embedding reduction"""
        # Create sample embeddings
        embeddings = np.random.rand(10, 50)  # 10 samples, 50 dimensions

        reduced = reduce_embeddings_2d(embeddings, method="pca")
        self.assertEqual(reduced.shape, (10, 2))

    def test_reduce_embeddings_2d_tsne(self):
        """Test t-SNE-based embedding reduction"""
        # Create larger sample for t-SNE (needs more samples for stable results)
        embeddings = np.random.rand(20, 10)  # 20 samples, 10 dimensions

        reduced = reduce_embeddings_2d(embeddings, method="tsne")
        self.assertEqual(reduced.shape, (20, 2))

    def test_reduce_embeddings_2d_invalid_method(self):
        """Test invalid reduction method"""
        embeddings = np.random.rand(5, 10)

        with self.assertRaises(ValueError):
            reduce_embeddings_2d(embeddings, method="invalid")


if __name__ == "__main__":
    unittest.main()
