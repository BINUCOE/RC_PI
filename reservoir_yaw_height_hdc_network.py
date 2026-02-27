import numpy as np
import pickle
from utils.func import pre_encoder, get_delta_hdc
import torch
import torch.nn as nn


class ReservoirMLP(nn.Module):
    def __init__(self):
        super(ReservoirMLP, self).__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(in_features=36, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=2)
        )

    def forward(self, x):
        return self.decoder_net(x)


class Yaw_Height_HDC_Reservoir:
    def __init__(self, reservoir_path, mpl_path, half_height_range=18):
        try:
            with open(reservoir_path, "rb") as f:
                self.reservoir = pickle.load(f)
        except FileNotFoundError:
            print("Warning: ESN model file not found, reservoir will be None")
            self.reservoir = None

        self.cann_reservoir = ReservoirMLP()
        try:
            state_dict = torch.load(mpl_path)
            self.cann_reservoir.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
        self.cann_reservoir.eval()

        # 确定编码的高度范围
        self.half_height_range = half_height_range
        self.height_th = np.pi / self.half_height_range
        self.theta_th = np.pi / 18
        self.height_buffer = 0
        self.pre_height = 0
        self.pre_yaw = 0
        self.odo_yaw = 0
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY = 0.1
        self.cann_ann_history = []
        self.target_yaw_history = []
        self.yaw_decoded_history = []
        curYawTheta, curHeight = self.get_hdc_initial_value()
        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = [curYawTheta, curHeight]

    def get_hdc_initial_value(self):
        return 0., 0.

    def get_current_yaw_height_value(self, yaw_cann_state, height_cann_state, heightV):
        with torch.no_grad():
            yaw_tensor = torch.tensor(yaw_cann_state).float()
            yaw_decoder_output = self.cann_reservoir.decoder_net(yaw_tensor).cpu().numpy()
            yaw_angle = np.arctan2(yaw_decoder_output[0, 0], yaw_decoder_output[0, 1])

            if heightV == 0:
                height_angle = self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH[1]
            else:
                height_tensor = torch.tensor(height_cann_state).float()
                height_decoder_output = self.cann_reservoir.decoder_net(height_tensor).cpu().numpy()
                height_angle = np.arctan2(height_decoder_output[0, 0], height_decoder_output[0, 1])

        yaw_normalized = yaw_angle % (2 * np.pi)
        height_normalized = height_angle % (2 * np.pi)

        return yaw_normalized * 36 / (2 * np.pi), height_normalized * 36 / (2 * np.pi)

    def yaw_height_hdc_iteration(self, vt_id, yawRotV, heightV, VT):
        cur_target_yaw = self.pre_yaw + yawRotV
        self.pre_yaw = cur_target_yaw

        cur_target_height = self.pre_height + heightV
        self.pre_height = cur_target_height
        cur_target_height *= self.height_th

        cur_target_yaw_encoded = pre_encoder(np.array(cur_target_yaw))
        cur_target_height_encoded = pre_encoder(np.array(cur_target_height))

        yaw_cann_state = self.reservoir.run(cur_target_yaw_encoded)
        height_cann_state = None if heightV == 0 else self.reservoir.run(cur_target_height_encoded)

        self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = self.get_current_yaw_height_value(yaw_cann_state, height_cann_state,
                                                                                heightV)
        if VT[vt_id].first != 1:
            act_yaw = VT[vt_id].hdc_yaw
            act_height = VT[vt_id].hdc_height
            bais = get_delta_hdc(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH, [act_yaw, act_height])

            self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH = np.mod(np.array(self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH) + bais,
                                                         36).tolist()

            self.pre_yaw += bais[0] * self.theta_th
            self.pre_height += bais[1]

        return self.MAX_ACTIVE_YAW_HEIGHT_HIS_PATH
