import os
import time
import inspect
import numpy as np
from PIL import Image

# Import configuration loader
from config_loader import load_yaml_config

# Import network and vision modules
import reservoir_grid_cells_network
import reservoir_yaw_height_hdc_network
from utils import multilayered_experience_map
from utils import visual_odometry_bak, visual_template_bak, multilayered_experience_map_bak
from utils import visual_odometry_kitti, visual_template_kitti


def get_valid_kwargs(func, config_dict):
    """
    Auto adapter: Automatically extract required parameters from config_dict
    """
    sig = inspect.signature(func)
    valid_keys = [k for k in sig.parameters.keys() if k in config_dict]
    return {k: config_dict[k] for k in valid_keys}


class NeuroSLAM:
    def __init__(self, dataset_type="CUSTOM"):
        self.cfg = load_yaml_config(config_path="config.yaml", dataset_name=dataset_type)
        self.dataset_type = dataset_type
        self.DEGREE_TO_RADIAN = np.pi / 180

        if dataset_type == "CUSTOM":
            self.model_dir = "Self"
            self.initial_pose = [np.pi / 2, 0.0, 0.0, 0.0]
            self.vo_class = visual_odometry_bak.VisualOdometry
            self.vt_class = visual_template_bak.VisualTemplateManager
            self.mep_class = multilayered_experience_map_bak.ExperienceMap
        elif dataset_type in ["KITTI_00", "KITTI_07"]:
            self.model_dir = dataset_type
            self.initial_pose = [0.0, 0.0, 0.0, 0.0]
            self.vo_class = visual_odometry_kitti.VisualOdometry
            self.vt_class = visual_template_kitti.VisualTemplateManager
            self.mep_class = multilayered_experience_map.ExperienceMap
        else:
            raise ValueError(f"Unknown Dataset type: {dataset_type}")

        # 3. Read image paths
        img_path = self.cfg.get("image_path", "")
        if not os.path.exists(img_path):
            print(f"Warning: Image path not found: {img_path}")
            self.image_sources = []
        else:
            self.image_sources = sorted([os.path.join(img_path, f) for f in os.listdir(img_path)])

    def run(self):
        # Prioritize reading process configuration from YAML, use default values if not exists
        n_steps = self.cfg.get("n_steps", 1)
        vt_step_config = self.cfg.get("VT_STEP", 1)

        m_dir = self.model_dir
        # Initialize reservoir networks
        hdcn = reservoir_yaw_height_hdc_network.Yaw_Height_HDC_Reservoir(
            f"model/{m_dir}/hdc_esn_model.pkl", f"model/{m_dir}/reservoir_mlp_hdc_model.pth")
        gcn = reservoir_grid_cells_network.GridCellsNetworkReservoir(
            f"model/{m_dir}/gc_esn_model.pkl", f"model/{m_dir}/reservoir_mlp_gc_model.pth")

        # Instantiate core classes using adaptive function to prevent parameter mismatch
        vo_kwargs = get_valid_kwargs(self.vo_class.__init__, self.cfg)
        vt_kwargs = get_valid_kwargs(self.vt_class.__init__, self.cfg)
        mep_kwargs = get_valid_kwargs(self.mep_class.__init__, self.cfg)

        vo = self.vo_class(**vo_kwargs)
        vt = self.vt_class(**vt_kwargs)
        mep = self.mep_class(**mep_kwargs)

        gcX, gcY, gcZ = gcn.get_gc_initial_pos()
        curYawTheta, curHeightValue = hdcn.get_hdc_initial_value()
        temp, odo_x, odo_y, odo_z = self.initial_pose

        curFrame, preImg = 0, 0
        odo_history, gcn_history = [], []
        t_start = time.time()

        for i in range(0, len(self.image_sources), n_steps):
            print(f"\rProcessing progress: {((i // n_steps) + 1)} / {len(self.image_sources) // n_steps} frames...", end="")

            img = Image.open(self.image_sources[i])

            if img.mode != 'L':
                img = img.convert('L')

            curGrayImg = np.asarray(img, dtype=np.float32) / 255.0

            transV, yawRotV, heightV = vo.visual_odometry(np.copy(curGrayImg), 0)
            yawRotV *= self.DEGREE_TO_RADIAN

            curFrame += 1
            if getattr(vt, 'VT_STEP', vt_step_config) == 1:
                vt_img = np.copy(curGrayImg)
            else:
                if np.mod(curFrame, getattr(vt, 'VT_STEP', vt_step_config)) == 1:
                    vt_img = np.copy(curGrayImg)
                    preImg = vt_img
                else:
                    vt_img = preImg

            vt_id, VT = vt.visual_template(vt_img, gcX, gcY, gcZ, curYawTheta, curHeightValue)

            [curYawTheta, curHeightValue] = hdcn.yaw_height_hdc_iteration(vt_id, yawRotV, heightV, VT)
            [gcX, gcY, gcZ] = gcn.gc_iteration(vt_id, transV, curYawTheta * self.DEGREE_TO_RADIAN, heightV, VT)
            mep.exp_map_iteration(vt_id, transV, yawRotV, heightV, gcX, gcY, gcZ, curYawTheta, curHeightValue, VT)

            vt.PREV_VT_ID = mep.PREV_VT_ID = vt_id

            temp += yawRotV
            odo_x, odo_y, odo_z = odo_x + transV * np.cos(temp), odo_y + transV * np.sin(temp), odo_z + heightV
            odo_history.append([odo_x, odo_y, odo_z])
            gcn_history.append([gcX, gcY, gcZ])

        run_time = time.time() - t_start
        print(f"\nElapsed time: {run_time:.4f} seconds")
        exp_history = [[e.x_exp, e.y_exp, e.z_exp] for e in mep.EXPERIENCES]

        return {"odo": odo_history, "exp": exp_history, "gcn": gcn_history}, run_time


if __name__ == "__main__":
    datasets_to_test = ["CUSTOM", "KITTI_00", "KITTI_07"]
    runs_per_dataset = 5

    time_records = {ds: [] for ds in datasets_to_test}

    for dataset in datasets_to_test:
        print(f"\n{'=' * 60}")
        print(f"Start processing dataset: {dataset}")
        print(f"{'=' * 60}")

        for run_idx in range(runs_per_dataset):
            print(f"\n---> {dataset} : Run {run_idx + 1} / {runs_per_dataset} <---")

            slam = NeuroSLAM(dataset)
            if dataset == "CUSTOM":
                n_step = 2
            res, elapsed_time = slam.run()

            time_records[dataset].append(elapsed_time)

    # ---------------- Save time records to file ---------------- #
    with open("rc_runtime_report.txt", "w", encoding='utf-8') as f:
        f.write("Runtime statistics report for each dataset (Unit: seconds)\n")
        f.write("#" * 60 + "\n")

        print("\n\n" + "#" * 60)
        print("Runtime statistics report for each dataset (Unit: seconds)")
        print("#" * 60)

        for dataset, times in time_records.items():
            if not times:
                continue
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            f.write(f"\nDataset: [{dataset}]\n")
            for i, t in enumerate(times):
                f.write(f"  Run {i + 1}: {t:.4f} s\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Average time: {avg_time:.4f} s\n")
            f.write(f"  Maximum time: {max_time:.4f} s\n")
            f.write(f"  Minimum time: {min_time:.4f} s\n")

            print(f"\nDataset: [{dataset}]")
            for i, t in enumerate(times):
                print(f"  Run {i + 1}: {t:.4f} s")
            print("-" * 30)
            print(f"  Average time: {avg_time:.4f} s")
            print(f"  Maximum time: {max_time:.4f} s")
            print(f"  Minimum time: {min_time:.4f} s")

        f.write("\n" + "#" * 60 + "\n")
        print("\n" + "#" * 60)
