import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGUIStructure(unittest.TestCase):
    """Test the new GUI structure components"""

    def test_vocab_manager_import(self):
        """Test that VocabManager can be imported"""
        try:
            from src.gui.managers.vocab_manager import VocabManager

            self.assertTrue(hasattr(VocabManager, "load_vocabulary"))
            self.assertTrue(hasattr(VocabManager, "get_vocab"))
        except ImportError as e:
            self.fail(f"Failed to import VocabManager: {e}")

    def test_model_manager_import(self):
        """Test that ModelManager can be imported"""
        try:
            from src.gui.managers.model_manager import ModelManager

            self.assertTrue(hasattr(ModelManager, "load_model"))
            self.assertTrue(hasattr(ModelManager, "predict_next_tokens"))
        except ImportError as e:
            self.fail(f"Failed to import ModelManager: {e}")

    def test_training_manager_import(self):
        """Test that TrainingManager can be imported"""
        try:
            from src.gui.managers.training_manager import TrainingManager

            self.assertTrue(hasattr(TrainingManager, "start_training"))
            self.assertTrue(hasattr(TrainingManager, "stop_training"))
        except ImportError as e:
            self.fail(f"Failed to import TrainingManager: {e}")

    def test_widget_imports(self):
        """Test that new widgets can be imported"""
        widgets_to_test = [
            ("src.gui.widgets.autocomplete", "AutocompleteTextEdit"),
            ("src.gui.widgets.training_controls", "TrainingControls"),
            ("src.gui.widgets.progress_tracker", "ProgressTracker"),
            ("src.gui.widgets.custom_progress_bar", "CustomProgressBar"),
            ("src.gui.widgets.model_selector", "ModelSelector"),
            ("src.gui.widgets.attention_matrix", "AttentionMatrixWidget"),
            ("src.gui.widgets.embedding_plot", "EmbeddingPlotWidget"),
            ("src.gui.widgets.loss_plot_widget", "LossPlotWidget"),
        ]

        for module_name, class_name in widgets_to_test:
            with self.subTest(module=module_name, class_name=class_name):
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    self.assertTrue(hasattr(cls, "__init__"))
                except (ImportError, AttributeError) as e:
                    self.fail(f"Failed to import {class_name} from {module_name}: {e}")

    def test_tab_imports(self):
        """Test that new tabs can be imported"""
        try:
            from src.gui.tabs.input_tab import InputTab

            self.assertTrue(hasattr(InputTab, "setup_ui"))
        except ImportError as e:
            self.fail(f"Failed to import InputTab: {e}")

    def test_vocab_manager_basic_functionality(self):
        """Test basic VocabManager functionality"""
        from src.gui.managers.vocab_manager import VocabManager
        from src.core.logger import get_testing_logger

        logger = get_testing_logger()
        vocab_manager = VocabManager(logger)

        # Test with a simple vocab
        vocab_text = "hello\nworld\ntest\n"
        vocab_file = os.path.join(os.path.dirname(__file__), "test_vocab.txt")

        try:
            with open(vocab_file, "w") as f:
                f.write(vocab_text)

            vocab_manager.load_vocabulary(vocab_file)

            self.assertEqual(
                vocab_manager.get_vocab_size(), 4
            )  # 3 tokens + 1 for <PAD>
            self.assertIn("hello", vocab_manager.get_vocab())
            self.assertIn("<PAD>", vocab_manager.get_vocab())

            # Test token conversion
            tokens = vocab_manager.text_to_tokens("hello world")
            self.assertEqual(len(tokens), 2)
            self.assertEqual(tokens[0], 1)  # hello should be token 1

        finally:
            if os.path.exists(vocab_file):
                os.remove(vocab_file)


if __name__ == "__main__":
    unittest.main()
