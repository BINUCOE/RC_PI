import numpy as np
from scipy.special import i0  # 导入修正贝塞尔函数

def pre_encoder(pos: np.ndarray, dims=36) -> np.ndarray:
    # x = np.linspace(-np.pi, np.pi, dims)  # 假设有36个神经元
    # A = 10.
    # a = 0.5
    # z_range = 2 * np.pi  # 角度范围为-π到π
    # d = x - pos
    # d = np.remainder(d, z_range)
    # d = np.where(d > 0.5 * z_range, d - z_range, d)
    # return A * np.exp(-0.25 * np.square(d / a))
    ################################################################################
    x = np.linspace(-180, 180, dims)  # 假设有36个神经元
    A = 10.
    a = 0.5
    z_range = 2 * 180  # 角度范围为-π到π
    d = x - pos
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return A * np.exp(-0.25 * np.square(d / a))

# def pre_encoder(pos: np.ndarray, dims=36) -> np.ndarray:
#     """改进的高斯角度编码器，使用冯·米塞斯分布"""
#     # 使用弧度制，保持单位一致
#     centers = np.linspace(-np.pi, np.pi, dims)  # 神经元中心位置（弧度）
#     kappa = 8.0  # 集中参数，控制调谐宽度
#
#     # 计算角度差并处理周期性
#     d = pos - centers.reshape(1, -1)
#     d = (d + np.pi) % (2 * np.pi) - np.pi  # 映射到[-π, π]
#
#     # 使用冯·米塞斯分布（圆形正态分布）
#     return np.exp(kappa * np.cos(d)) / (2 * np.pi * i0(kappa))


# ------------------ 3D 外积编码 ------------------
def cann3d_pre_encoder(I: np.ndarray, dims: int = 12, concat: bool = False) -> np.ndarray:
    # 拆分三维分量
    I_x = I[:, 0].reshape(-1, 1)
    I_y = I[:, 1].reshape(-1, 1)
    I_z = I[:, 2].reshape(-1, 1)

    # 逐分量编码
    x_enc = pre_encoder(I_x, dims)
    y_enc = pre_encoder(I_y, dims)
    z_enc = pre_encoder(I_z, dims)

    if concat:
        # 拼接模式：(N, 3*dims)
        return np.concatenate([x_enc, y_enc, z_enc], axis=1)
    else:
        # 外积模式：(N, dims, dims, dims)
        x_enc = x_enc.reshape(-1, dims, 1, 1)
        y_enc = y_enc.reshape(-1, 1, dims, 1)
        z_enc = z_enc.reshape(-1, 1, 1, dims)
        return x_enc * y_enc * z_enc  # 逐元素乘法实现外积
###################

# import numpy as np
#
#
# def gaussian_encoder(pos: np.ndarray, centers: np.ndarray, A=10., a=0.5) -> np.ndarray:
#     """优化后的通用高斯编码器"""
#     z_range = 2 * 180  # 假设角度范围为-π到π
#     d = np.remainder(centers - pos, z_range)
#     d = np.where(d > 0.5 * z_range, d - z_range, d)
#     return A * np.exp(-0.25 * (d / a) ** 2)
#
#
# def pre_encoder(pos: np.ndarray, dims=36) -> np.ndarray:
#     """基于高斯编码器的1D编码"""
#     centers = np.linspace(-180, 180, dims)
#     return gaussian_encoder(pos, centers)


# def cann3d_pre_encoder(I: np.ndarray, dims: int = 12, concat: bool = False) -> np.ndarray:
#     """优化后的基于高斯编码器的3D编码"""
#     centers = np.linspace(-180, 180, dims)
#     I_reshaped = I.reshape(-1, 3, 1)  # 将 I 的每一列扩展为 (N, 3, 1)
#     d = np.remainder(centers - I_reshaped, 2 * 180)
#     d = np.where(d > 180, d - 2 * 180, d)
#     encoded = 10. * np.exp(-0.25 * (d / 0.5) ** 2)  # 直接计算高斯编码
#
#     if concat:
#         return encoded.reshape(-1, 3 * dims)  # 展平并拼接
#     else:
#         return np.prod(encoded.reshape(-1, 3, dims, 1), axis=1)  # 计算外积


# ------------------------------
# 解码函数
# ------------------------------
def decoder_in_math(cann_state: np.ndarray, range: tuple = (-np.pi, np.pi), theta_output: bool = True) -> np.ndarray:
    """NumPy实现的解码器，将神经元激活向量还原为角度或三角分量"""
    neuro_num = cann_state.shape[1]
    theta = np.linspace(range[0], range[1], neuro_num)  # 神经元对应的角度网格

    # 去除最后一个神经元（与第一个重复，处理周期性）
    cann_state = cann_state[:, :-1]
    theta = theta[:-1]

    sin_sum = np.sum(cann_state * np.sin(theta), axis=1)
    cos_sum = np.sum(cann_state * np.cos(theta), axis=1)

    # neuro_num = cann_state.shape[1]
    # theta = np.linspace(range[0], range[1], neuro_num, endpoint=False)  # Avoid redundant slicing
    #
    # # Compute weighted sums directly
    # sin_sum = np.dot(cann_state, np.sin(theta))
    # cos_sum = np.dot(cann_state, np.cos(theta))

    if theta_output:
        # 解码为角度：arctan2(正弦加权和, 余弦加权和)
        output = np.arctan2(sin_sum, cos_sum).reshape(-1, 1)
    else:
        # 解码为正弦和余弦分量
        output = np.stack([sin_sum, cos_sum], axis=1).reshape(-1, 2)

    return output


def cann3d_decoder_in_math(U: np.ndarray, theta_output: bool = True) -> np.ndarray:
    """NumPy实现的3D张量解码器，还原三维角度向量"""
    # 沿不同维度求和，提取x、y、z分量的激活向量
    U_x = U.sum(axis=(2, 3))  # 沿y,z维度求和
    U_y = U.sum(axis=(1, 3))  # 沿x,z维度求和
    U_z = U.sum(axis=(1, 2))  # 沿x,y维度求和

    # 逐分量解码
    out_x = decoder_in_math(U_x, theta_output=theta_output)
    out_y = decoder_in_math(U_y, theta_output=theta_output)
    out_z = decoder_in_math(U_z, theta_output=theta_output)

    # 拼接结果
    return np.concatenate([out_x, out_y, out_z], axis=1)


# ------------------------------
# 向量处理函数
# ------------------------------
def vt_modle_hdc(x: np.ndarray) -> np.ndarray:
    """HDC模型的向量调制函数"""
    m = 0.2 * np.exp(-x ** 2 / 9.1)
    return (m / (1 + m)) * x


def vt_modle_gcn(x: np.ndarray) -> np.ndarray:
    """GCN模型的向量调制函数"""
    m = 0.13 * np.exp(-x ** 2 / 10.98)
    return (m / (1 + m)) * x


def find_short_delta(start, end):
    delta = end - start
    delta = delta if abs(delta) < 36 - abs(delta) else np.sign(delta) * (abs(delta) - 36)
    return delta


def find_short_vector(start: np.ndarray, end: np.ndarray, dims: int = 36) -> np.ndarray:
    """计算两个角度向量之间的最短距离向量"""
    delta = []
    for s, e in zip(start, end):
        delta.append(find_short_delta(s, e))
    return np.array(delta, dtype=np.float32)


def get_delta_hdc(start, end, dims=36) -> np.ndarray:
    """计算HDC模型的偏差向量（2D）"""
    vec = find_short_vector(start, end, dims)
    vector_norm = np.linalg.norm(vec)

    if vector_norm > 9 or vector_norm < 1e-10:
        return np.zeros(2, dtype=np.float32)
    else:
        return vt_modle_hdc(vector_norm) * (vec / vector_norm)


def get_delta_gcn(start: np.ndarray, end: np.ndarray, dims: int = 36) -> np.ndarray:
    """计算GCN模型的偏差向量（3D）"""
    vec = find_short_vector(start, end, dims)
    vector_norm = np.linalg.norm(vec)
    # if vector_norm < 1e-10:
    #     raise ValueError("Zero-length vector detected. start and end points are identical.")
    if vector_norm > 9 or vector_norm < 1e-10:
        return np.zeros(3, dtype=np.float32)
    else:
        return vt_modle_gcn(vector_norm) * (vec / vector_norm)


def get_all_yaw_height_hdc(yawRotV: np.ndarray, heightV: np.ndarray, cur_target_yaw=0, cur_target_height=0):
    """
    获取当前激活的Yaw-Height HDC编码
    :param vt_id: 当前视觉模板ID
    :param yawRotV: 当前Yaw旋转速度
    :param heightV: 当前高度变化速度
    :param VT: 当前视觉模板数据
    :return: 当前激活的Yaw-Height HDC编码
    """
    # 计算当前Yaw和Height的目标编码
    cur_target_yaw += yawRotV
    cur_target_height += heightV

    # 编码为HDC格式
    cur_target_yaw_encoded = pre_encoder(np.array(cur_target_yaw).reshape(1, 1))
    cur_target_height_encoded = pre_encoder(np.array(cur_target_height).reshape(1, 1))

    # 返回编码结果
    return cur_target_yaw_encoded, cur_target_height_encoded


# def vt_modle_hdc(x: np.ndarray) -> np.ndarray:
#     """改进的HDC调制函数"""
#     # 使用更合理的参数
#     return x * np.exp(-0.5 * x ** 2) / (1 + 0.5 * x ** 2)
#
#
# def vt_modle_gcn(x: np.ndarray) -> np.ndarray:
#     """改进的GCN调制函数"""
#     # 基于冯·米塞斯分布的调制
#     return x * np.exp(-0.5 * x ** 2) / (1 + 0.3 * x ** 2)
#
#
# def angular_difference(a: float, b: float, period: float = 2 * np.pi) -> float:
#     """计算最短周期性角度差"""
#     diff = (b - a + period / 2) % period - period / 2
#     return diff
#
#
# def find_short_vector(start: np.ndarray, end: np.ndarray, dims: int = 36) -> np.ndarray:
#     """改进的角度差向量计算"""
#     # 计算每个维度的角度差
#     return np.array([angular_difference(s, e) for s, e in zip(start, end)])
#
#
# def smooth_threshold(norm: float, max_norm: float = np.pi / 2, steepness: float = 5.0) -> float:
#     """平滑阈值函数"""
#     return 1 / (1 + np.exp(steepness * (norm - max_norm)))
#
#
# def get_delta_gcn(start: np.ndarray, end: np.ndarray, dims: int = 36) -> np.ndarray:
#     """改进的GCN偏差向量计算"""
#     vec = find_short_vector(start, end, dims)
#     vector_norm = np.linalg.norm(vec)
#
#     # 使用平滑阈值替代硬阈值
#     weight = smooth_threshold(vector_norm)
#
#     if weight < 0.01:  # 极小响应阈值
#         return np.zeros(3, dtype=np.float32)
#     else:
#         # 应用调制函数
#         modulated = vt_modle_gcn(vector_norm) * (vec / vector_norm)
#         return weight * modulated
#
#
# def get_delta_hdc(start, end, dims=36) -> np.ndarray:
#     """改进的HDC偏差向量计算"""
#     vec = find_short_vector(start, end, dims)
#     vector_norm = np.linalg.norm(vec)
#
#     # 使用平滑阈值
#     weight = smooth_threshold(vector_norm, max_norm=np.pi / 3)
#
#     if weight < 0.01:
#         return np.zeros(2, dtype=np.float32)
#     else:
#         modulated = vt_modle_hdc(vector_norm) * (vec / vector_norm)
#         return weight * modulated
#
#
# # ------------------ 改进的Yaw-Height编码 ------------------
# def get_all_yaw_height_hdc(yawRotV: np.ndarray, heightV: np.ndarray,
#                            cur_target_yaw: float = 0, cur_target_height: float = 0):
#     """改进的Yaw-Height编码"""
#     # 更新目标值
#     cur_target_yaw = (cur_target_yaw + yawRotV) % (2 * np.pi)
#     cur_target_height += heightV
#
#     # 编码
#     yaw_enc = pre_encoder(np.array([cur_target_yaw]))
#     height_enc = pre_encoder(np.array([cur_target_height]))
#
#     return yaw_enc, height_enc

# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 测试编码和解码
    np.random.seed(42)
    test_angles = np.random.uniform(-np.pi, np.pi, size=(5, 1))  # 随机角度
    encoded = pre_encoder(test_angles)
    print("角度编码形状：", encoded.shape)
    decoded = decoder_in_math(encoded)
    print("角度编码解码误差：", np.mean(np.abs(test_angles - decoded)))

    # 测试3D编码
    test_3d = np.random.uniform(-np.pi, np.pi, size=(5, 3))  # 3D角度向量
    encoded_3d = cann3d_pre_encoder(test_3d)
    decoded_3d = cann3d_decoder_in_math(encoded_3d)
    print("3D编码解码误差：", np.mean(np.abs(test_3d - decoded_3d)))

    # 测试最短距离计算
    start = np.array([1, 2, 2])
    end = np.array([35, 35, 34])
    delta = get_delta_gcn(start, end)
    print("GCN偏差向量：", delta)
