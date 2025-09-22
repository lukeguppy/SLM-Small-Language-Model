import os
import json
from datetime import datetime


def test_meta_format():
    """Test that meta files are saved in the correct key-value format"""

    # Test the format of existing meta files
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        print("Models directory not found")
        return

    meta_files = [f for f in os.listdir(models_dir) if f.endswith(".meta")]

    print(f"Found {len(meta_files)} meta files: {meta_files}")

    for meta_file in meta_files:
        meta_path = os.path.join(models_dir, meta_file)
        print(f"\n--- Testing {meta_file} ---")

        try:
            with open(meta_path, "r") as f:
                content = f.read().strip()

            # Check if it's JSON format (old format)
            if content.startswith("{"):
                print("ERROR: Meta file is in JSON format, should be key-value format")
                try:
                    data = json.loads(content)
                    print(f"   JSON content: {data}")
                except:
                    print("   Could not parse JSON")
            else:
                # Check key-value format
                lines = content.split("\n")
                parsed_data = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        parsed_data[key] = value

                print("SUCCESS: Meta file is in key-value format")

                # Check for required fields
                required_fields = [
                    "vocab_size",
                    "embed_dim",
                    "n_heads",
                    "n_layers",
                    "ff_dim",
                    "dropout",
                    "lr",
                    "epochs",
                    "weight_decay",
                    "batch_size",
                    "train_size",
                    "val_size",
                    "test_size",
                    "vocab_path",
                    "saved_date",
                    "saved_time",
                ]

                missing_fields = []
                for field in required_fields:
                    if field not in parsed_data:
                        missing_fields.append(field)

                if missing_fields:
                    print(f"WARNING: Missing fields: {missing_fields}")
                else:
                    print("SUCCESS: All required fields present")

                # Print some key values
                print(
                    f"   Sample fields: vocab_size={parsed_data.get('vocab_size')}, "
                    f"embed_dim={parsed_data.get('embed_dim')}, "
                    f"saved_date={parsed_data.get('saved_date')}"
                )

        except Exception as e:
            print(f"ERROR reading {meta_file}: {str(e)}")


if __name__ == "__main__":
    test_meta_format()
