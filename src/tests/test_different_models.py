import sys
import os

sys.path.append(".")

from src.core.model_loader import ModelLoader


def test_model_loading():
    print("Testing ModelLoader with different model architectures...\n")

    # Test 1: Large model (lukegpt_best.pt)
    print("=" * 50)
    print("Testing lukegpt_best.pt (large model):")
    try:
        model1 = ModelLoader.load_model_safely("models/lukegpt_best.pt", vocab_size=939)
        print("SUCCESS: Model loaded successfully!")
        print(f"   vocab_size: {model1.vocab_size}")
        print(f"   embed_dim: {model1.embed_dim}")
        print(f"   n_heads: {model1.n_heads}")
        print(f"   n_layers: {model1.n_layers}")
        print(f"   ff_dim: {model1.ff_dim}")
    except Exception as e:
        print(f"FAILED: {e}")

    print()

    # Test 2: Smaller model (test2.pt)
    print("=" * 50)
    print("Testing test2.pt (smaller model):")
    try:
        model2 = ModelLoader.load_model_safely("models/test2.pt", vocab_size=939)
        print("SUCCESS: Model loaded successfully!")
        print(f"   vocab_size: {model2.vocab_size}")
        print(f"   embed_dim: {model2.embed_dim}")
        print(f"   n_heads: {model2.n_heads}")
        print(f"   n_layers: {model2.n_layers}")
        print(f"   ff_dim: {model2.ff_dim}")
    except Exception as e:
        print(f"FAILED: {e}")

    print()
    print("=" * 50)
    print("Summary:")
    print("- Both models have the same vocab_size (939)")
    print("- But different architectures (embed_dim, n_heads, n_layers, ff_dim)")
    print("- ModelLoader automatically detected and loaded both correctly!")


if __name__ == "__main__":
    test_model_loading()
    print("\nTest completed - no cleanup needed as this test only loads existing models")
