import os


class FileOperations:
    """Handles file operations for models."""

    @staticmethod
    def delete_model_files(model_name, logger):
        """Delete all files associated with a model (model, meta, training data)."""
        from .paths import ModelPaths, DataPaths

        if model_name.endswith("_best"):
            base_name = ModelPaths._normalise_model_name(model_name[:-5])
            files_to_delete = [
                ModelPaths.get_best_model_path(base_name),
                ModelPaths.get_best_meta_path(base_name),
                DataPaths.get_training_data_path(base_name),
            ]
        elif model_name.endswith("_final"):
            base_name = ModelPaths._normalise_model_name(model_name[:-6])
            files_to_delete = [
                ModelPaths.get_final_model_path(base_name),
                ModelPaths.get_final_meta_path(base_name),
                DataPaths.get_training_data_path(base_name),
            ]
        else:
            # Regular model files
            base_name = ModelPaths._normalise_model_name(model_name)
            files_to_delete = [
                ModelPaths.get_model_path(base_name),
                ModelPaths.get_meta_path(base_name),
                DataPaths.get_training_data_path(base_name),
            ]

        deleted_files = []
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    logger.info(f"Deleted: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

        return deleted_files
