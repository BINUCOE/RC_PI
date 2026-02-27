import yaml
import os


def load_yaml_config(config_path="config.yaml", dataset_name=None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration {config_path} not found!")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    active_dataset = dataset_name if dataset_name else config.get("active_dataset")

    if "dataset_config" not in config or active_dataset not in config["dataset_config"]:
        raise ValueError(f"Dataset '{active_dataset}' not found in configuration!")

    dataset_config = config["dataset_config"][active_dataset]

    merged_config = {
        "ACTIVE_DATASET": active_dataset,
        **dataset_config,
        **config.get("process_config", {}),
        **config.get("output_config", {})
    }

    def list_to_slice(list_range):
        if isinstance(list_range, list) and len(list_range) == 2:
            return slice(list_range[0], list_range[1])
        return list_range

    slice_keys = [
        "VT_IMG_CROP_X_RANGE", "VT_IMG_CROP_Y_RANGE",
        "ODO_IMG_TRANS_Y_RANGE", "ODO_IMG_TRANS_X_RANGE",
        "ODO_IMG_YAW_ROT_Y_RANGE", "ODO_IMG_YAW_ROT_X_RANGE",
        "ODO_IMG_HEIGHT_V_Y_RANGE", "ODO_IMG_HEIGHT_V_X_RANGE"
    ]
    for key in slice_keys:
        if key in merged_config:
            merged_config[key] = list_to_slice(merged_config[key])

    return merged_config


CONFIG = load_yaml_config()

if __name__ == "__main__":
    default_config = load_yaml_config("config.yaml")
    kitti_config = load_yaml_config("config.yaml", dataset_name="kitti_00")
