import os


class ModelDiscovery:
    """Handles model discovery and listing operations."""

    @staticmethod
    def get_model_files():
        """Get list of all model files (.pt) in the models directory and subdirectories."""
        from .paths import ModelPaths

        models_dir = ModelPaths.get_models_dir()
        if not os.path.exists(models_dir):
            return []

        model_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".pt"):
                    # Get the relative path from models directory
                    rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                    model_files.append(rel_path)
        return model_files

    @staticmethod
    def get_model_names():
        """Get list of all unique model names from model files."""
        from .paths import ModelPaths

        models_dir = ModelPaths.get_models_dir()
        if not os.path.exists(models_dir):
            return []

        model_names = set()
        for file in os.listdir(models_dir):
            if file.endswith(".pt"):
                # Extract base name by removing .pt extension and any _best/_final suffix
                base_name = file[:-3]  # Remove .pt
                if base_name.endswith("_best"):
                    base_name = base_name[:-5]
                elif base_name.endswith("_final"):
                    base_name = base_name[:-6]
                model_names.add(base_name)
        return list(model_names)

    @staticmethod
    def get_available_models():
        """Get all available models with their variants in a structured format."""
        from .paths import ModelPaths

        models_dir = ModelPaths.get_models_dir()
        if not os.path.exists(models_dir):
            return []

        available_models = []

        # Find all model files in subdirectories
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".pt"):
                    # Get the relative path from models directory
                    rel_path = os.path.relpath(os.path.join(root, file), models_dir)

                    # Extract the filename without .pt
                    filename = os.path.basename(rel_path)
                    base_name = filename[:-3]

                    # The display name is the filename without .pt
                    display_name = base_name

                    # The model identifier is also the base_name (includes variant suffix)
                    model_identifier = base_name

                    available_models.append((display_name, model_identifier))

        return sorted(available_models)
