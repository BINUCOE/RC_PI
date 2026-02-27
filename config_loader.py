import yaml
import os


def load_yaml_config(config_path="config.yaml", dataset_name=None):
    """
    加载YAML配置文件，并进行必要的格式转换
    :param config_path: YAML配置文件路径
    :param dataset_name: 外部指定的数据集名称 (若传入，将覆盖 yaml 中的 active_dataset)
    :return: 解析后的配置字典
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")

    # 读取YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 【核心修改点】：优先使用外部传入的 dataset_name，否则读取 YAML 中的默认配置
    active_dataset = dataset_name if dataset_name else config.get("active_dataset")

    if "dataset_config" not in config or active_dataset not in config["dataset_config"]:
        raise ValueError(f"数据集 '{active_dataset}' 的配置在 yaml 文件中不存在！")

    dataset_config = config["dataset_config"][active_dataset]

    # 合并所有配置
    merged_config = {
        # 基础信息
        "ACTIVE_DATASET": active_dataset,
        # 数据集配置
        **dataset_config,
        # 运行配置
        **config.get("process_config", {}),
        # 输出配置
        **config.get("output_config", {})
    }

    # 将YAML中的列表转换为slice对象（适配原代码的slice参数）
    def list_to_slice(list_range):
        """将 [start, end] 列表转换为 slice(start, end)"""
        if isinstance(list_range, list) and len(list_range) == 2:
            return slice(list_range[0], list_range[1])
        return list_range

    # 转换所有需要的range参数
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


# 提供一个默认的全局配置（默认加载 YAML 中的 active_dataset）
CONFIG = load_yaml_config()

if __name__ == "__main__":
    # 测试默认配置加载
    print("=== 默认加载 ===")
    default_config = load_yaml_config("config.yaml")
    print(f"当前激活数据集: {default_config['ACTIVE_DATASET']}")

    # 测试动态传参加载
    print("\n=== 动态覆盖加载 ===")
    kitti_config = load_yaml_config("config.yaml", dataset_name="kitti_00")
    print(f"当前激活数据集: {kitti_config['ACTIVE_DATASET']}")


# import yaml
# import os
#
# def load_yaml_config(config_path="config.yaml"):
#     """
#     加载YAML配置文件，并进行必要的格式转换
#     :param config_path: YAML配置文件路径
#     :return: 解析后的配置字典
#     """
#     # 检查配置文件是否存在
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
#
#     # 读取YAML文件
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
#
#     # 获取激活的数据集配置
#     active_dataset = config["active_dataset"]
#     dataset_config = config["dataset_config"][active_dataset]
#
#     # 合并所有配置
#     merged_config = {
#         # 基础信息
#         "ACTIVE_DATASET": active_dataset,
#         # 数据集配置
#         **dataset_config,
#         # 运行配置
#         **config["process_config"],
#         # 输出配置
#         **config["output_config"]
#     }
#
#     # 将YAML中的列表转换为slice对象（适配原代码的slice参数）
#     def list_to_slice(list_range):
#         """将 [start, end] 列表转换为 slice(start, end)"""
#         if isinstance(list_range, list) and len(list_range) == 2:
#             return slice(list_range[0], list_range[1])
#         return list_range
#
#     # 转换所有需要的range参数
#     slice_keys = [
#         "VT_IMG_CROP_X_RANGE", "VT_IMG_CROP_Y_RANGE",
#         "ODO_IMG_TRANS_Y_RANGE", "ODO_IMG_TRANS_X_RANGE",
#         "ODO_IMG_YAW_ROT_Y_RANGE", "ODO_IMG_YAW_ROT_X_RANGE",
#         "ODO_IMG_HEIGHT_V_Y_RANGE", "ODO_IMG_HEIGHT_V_X_RANGE"
#     ]
#     for key in slice_keys:
#         if key in merged_config:
#             merged_config[key] = list_to_slice(merged_config[key])
#
#     return merged_config
#
#
# # 加载配置（供其他模块调用）
# CONFIG = load_yaml_config()
#
# if __name__ == "__main__":
#     # 测试配置加载
#     config = load_yaml_config("config.yaml")
#     for key, value in config.items():
#         print(f"{key}: {value}")