import os


class ProjectPaths:
    """Handles project structure paths."""

    @staticmethod
    def get_project_root():
        """Get the absolute path to the project root directory."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @staticmethod
    def get_data_dir():
        """Get the path to the data directory."""
        return os.path.join(ProjectPaths.get_project_root(), "data")

    @staticmethod
    def get_logs_dir():
        """Get the path to the logs directory."""
        return os.path.join(ProjectPaths.get_project_root(), "logs")


class ModelPaths:
    """Handles model-related paths and file operations."""

    @staticmethod
    def get_models_dir():
        """Get the path to the models directory."""
        return os.path.join(ProjectPaths.get_project_root(), "models")

    @staticmethod
    def _normalise_model_name(model_name):
        """Normalise model name by removing extension if present."""
        if model_name.endswith(".pt"):
            return model_name[:-3]
        return model_name

    @staticmethod
    def get_model_dir(model_name):
        """Get the path to a specific model's directory."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_models_dir(), base_name)

    @staticmethod
    def get_model_path(model_name, extension="pt"):
        """Get the full path to a model file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}.{extension}")

    @staticmethod
    def get_meta_path(model_name):
        """Get the full path to a model's meta file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}.meta")

    @staticmethod
    def get_best_model_path(model_name, extension="pt"):
        """Get the path to the best model file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}_best.{extension}")

    @staticmethod
    def get_final_model_path(model_name, extension="pt"):
        """Get the path to the final model file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}_final.{extension}")

    @staticmethod
    def get_best_meta_path(model_name):
        """Get the path to the best model's meta file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}_best.meta")

    @staticmethod
    def get_final_meta_path(model_name):
        """Get the path to the final model's meta file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}_final.meta")

    @staticmethod
    def ensure_model_dir(model_name):
        """Ensure the model-specific directory exists."""
        base_name = ModelPaths._normalise_model_name(model_name)
        model_dir = ModelPaths.get_model_dir(base_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    @staticmethod
    def ensure_models_dir():
        """Ensure the models directory exists."""
        os.makedirs(ModelPaths.get_models_dir(), exist_ok=True)


class DataPaths:
    """Handles data-related paths."""

    @staticmethod
    def get_vocab_path(vocab_filename="tokens.txt"):
        """Get the full path to the vocabulary file."""
        return os.path.join(ProjectPaths.get_data_dir(), vocab_filename)

    @staticmethod
    def get_sentences_path(sentences_filename="sentences.txt"):
        """Get the full path to the sentences file."""
        return os.path.join(ProjectPaths.get_data_dir(), sentences_filename)

    @staticmethod
    def get_training_data_dir():
        """Get the path to the training data directory."""
        return os.path.join(ModelPaths.get_models_dir(), "training_data")

    @staticmethod
    def get_training_data_path(model_name):
        """Get the full path to a model's training data CSV file."""
        base_name = ModelPaths._normalise_model_name(model_name)
        return os.path.join(ModelPaths.get_model_dir(base_name), f"{base_name}_training_data.csv")

    @staticmethod
    def ensure_training_data_dir():
        """Ensure the training data directory exists."""
        os.makedirs(DataPaths.get_training_data_dir(), exist_ok=True)
