import numpy as np
import torch
import torch.nn as nn
from utils.func import cann3d_pre_encoder, get_delta_gcn
import pickle


class ReservoirMLP(nn.Module):
    def __init__(self):
        super(ReservoirMLP, self).__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(in_features=36, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=6)
        )

    def forward(self, x):
        return self.decoder_net(x)


class GridCellsNetworkReservoir:
    def __init__(self, reservoir_path, mlp_path):
        self.GC_X_DIM = 36
        self.GC_Y_DIM = 36
        self.GC_Z_DIM = 36
        self.GC_X_TH_SIZE = np.pi / 18
        self.GC_Y_TH_SIZE = np.pi / 18
        self.GC_Z_TH_SIZE = np.pi / 18
        self.GC_TH_SIZE = np.array([self.GC_X_TH_SIZE, self.GC_Y_TH_SIZE, self.GC_Z_TH_SIZE])

        try:
            with open(reservoir_path, "rb") as f:
                self.reservoir = pickle.load(f)
        except FileNotFoundError:
            print("Warning: ESN model file not found, reservoir will be None")
            self.reservoir = None

        self.cann_reservoir = ReservoirMLP()
        try:
            state_dict = torch.load(mlp_path)
            self.cann_reservoir.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
        self.cann_reservoir.eval()

        self.buffer = [0, 0, 0]

        # initailize location and hidden statae
        self.hidden_state = (torch.zeros(2, 111), torch.zeros(2, 111))
        gcX, gcY, gcZ = self.get_gc_initial_pos()
        self.MAX_ACTIVE_XYZ_PATH = [gcX, gcY, gcZ]
        self.pre_location = np.array(self.MAX_ACTIVE_XYZ_PATH).reshape(1, 3)

        # recording list
        self.cann_reservoir_history = []
        self.cur_location_encoded_history = []
        self.delta_x_history = []
        self.transV_history = []

    def get_gc_initial_pos(self):
        gcX = int(np.floor((self.GC_X_DIM - 1) / 2))  # in 1:GC_X_DIM
        gcY = int(np.floor((self.GC_Y_DIM - 1) / 2))  # in 1:GC_Y_DIM
        gcZ = int(np.floor((self.GC_Z_DIM - 1) / 2))  # in 1:GC_Z_DIM
        return gcX, gcY, gcZ

    def get_gc_xyz(self, cann_state):
        with torch.no_grad():
            cann_state_tensor = torch.tensor(cann_state).float()
            pre_decoded_output = np.array(self.cann_reservoir.decoder_net(cann_state_tensor))

        # get the output in radian
        gc_x = np.arctan2(pre_decoded_output[0, 0], pre_decoded_output[0, 1])
        gc_y = np.arctan2(pre_decoded_output[0, 2], pre_decoded_output[0, 3])
        gc_z = np.arctan2(pre_decoded_output[0, 4], pre_decoded_output[0, 5])
        # from radian to idx
        gc_x = np.mod(gc_x, 2 * np.pi) / self.GC_X_TH_SIZE
        gc_y = np.mod(gc_y, 2 * np.pi) / self.GC_Y_TH_SIZE
        gc_z = np.mod(gc_z, 2 * np.pi) / self.GC_Z_TH_SIZE

        return gc_x, gc_y, gc_z

    def gc_iteration(self, vt_id, transV, curYawThetaInRadian, heightV, VT):
        """
        updatae_steps:
            1.get the location which needs to be encoded
            2.ANN forward
            3.decoding state
        """

        delta_x = transV * np.cos(curYawThetaInRadian)
        delta_y = transV * np.sin(curYawThetaInRadian)
        delta_z = heightV
        cur_location = self.pre_location + np.array([delta_x, delta_y, delta_z])
        self.pre_location = cur_location
        cur_location_encoded = cur_location * self.GC_TH_SIZE

        cur_location_encoded = cann3d_pre_encoder(cur_location_encoded, concat=True)

        cann_state = self.reservoir.run(cur_location_encoded)

        # self.cann_reservoir_history.append(cann_state.squeeze(0))
        self.MAX_ACTIVE_XYZ_PATH = self.get_gc_xyz(cann_state)

        if VT[vt_id].first != 1:
            actX = np.mod(VT[vt_id].gc_x, self.GC_X_DIM)
            actY = np.mod(VT[vt_id].gc_y, self.GC_Y_DIM)
            actZ = np.mod(VT[vt_id].gc_z, self.GC_Z_DIM)
            bais = get_delta_gcn(self.MAX_ACTIVE_XYZ_PATH, [actX, actY, actZ])
            self.MAX_ACTIVE_XYZ_PATH = np.mod(np.array(self.MAX_ACTIVE_XYZ_PATH) + bais, 36).tolist()
            self.pre_location += bais
        return self.MAX_ACTIVE_XYZ_PATH
