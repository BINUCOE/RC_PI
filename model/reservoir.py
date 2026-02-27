from CANN_model import CANN1D, CANN3D
import numpy as np
import brainpy as bp
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare
import matplotlib.pyplot as plt
import pickle
import os


def neuroslam_data(hdc_input_path, hdc_state_save_path, hdc_model_path):
    cann = CANN1D(num=36, k=8.1, z_min=-180, z_max=180, a=10, tau=1)
    Iext_encoded_list = []
    cann_state_list = []

    temp_yaw_height = np.loadtxt(hdc_input_path)
    for i in range(len(temp_yaw_height)):
        yaw = temp_yaw_height[i, 0]
        Iext = yaw.reshape(-1, 1)
        Iext_encoded = cann.get_stimulus_by_pos(Iext)

        runner = bp.DSRunner(cann, inputs=['input', Iext_encoded, 'iter'], monitors=['u'], dyn_vars=cann.vars())
        runner.run(0.1)

        Iext_encoded_list.append(Iext_encoded.reshape(-1))
        cann_state_list.append(runner.mon.u.reshape(-1))

    print(
        f"Encoded input shape: {np.array(Iext_encoded_list).shape}, CANN state shape: {np.array(cann_state_list).shape}")

    split_index = int(len(temp_yaw_height) * 0.8)
    x_train, y_train = np.array(Iext_encoded_list[:split_index]), np.array(cann_state_list[:split_index])
    x_test, y_test = np.array(Iext_encoded_list[split_index:]), np.array(cann_state_list[split_index:])
    np.savetxt(hdc_state_save_path, cann_state_list)
    print(f"CANN states saved to {hdc_state_save_path}")

    reservoir = Reservoir(units=100, sr=0.8, lr=0.2)
    readout = Ridge(ridge=1e-5)
    esn = reservoir >> readout

    esn.fit(x_train, y_train)

    with open(hdc_model_path, 'wb') as f:
        pickle.dump(esn, f)
    print(f"Model saved to {hdc_model_path}")

    predictions = esn.run(x_test)

    print(f"RMSE: {rmse(y_test, predictions)}; R^2 score: {rsquare(y_test, predictions)}")


def get_peak_position(u):
    u = np.array(u)
    idx = np.unravel_index(np.argmax(u), u.shape)
    return np.array(idx)


def neuroslam_3D_data(gc_input_path, gc_state_save_dir):
    os.makedirs(gc_state_save_dir, exist_ok=True)

    save_cann_x = os.path.join(gc_state_save_dir, "cann_state_x.txt")
    save_cann_y = os.path.join(gc_state_save_dir, "cann_state_y.txt")
    save_cann_z = os.path.join(gc_state_save_dir, "cann_state_z.txt")

    save_input_x = os.path.join(gc_state_save_dir, "Iext_encoded_x.txt")
    save_input_y = os.path.join(gc_state_save_dir, "Iext_encoded_y.txt")
    save_input_z = os.path.join(gc_state_save_dir, "Iext_encoded_z.txt")

    if all(os.path.exists(p) for p in
           [save_cann_x, save_cann_y, save_cann_z,
            save_input_x, save_input_y, save_input_z]):
        print("Found cached data, loading...")
        cann_state_x = np.loadtxt(save_cann_x)
        cann_state_y = np.loadtxt(save_cann_y)
        cann_state_z = np.loadtxt(save_cann_z)

        Iext_encoded_x = np.loadtxt(save_input_x)
        Iext_encoded_y = np.loadtxt(save_input_y)
        Iext_encoded_z = np.loadtxt(save_input_z)

        print(
            f"Iext_encoded_x shape: {np.array(Iext_encoded_x).shape}, cann_state_x shape: {np.array(cann_state_x).shape}")

        return (Iext_encoded_x, Iext_encoded_y, Iext_encoded_z,
                cann_state_x, cann_state_y, cann_state_z)

    cann = CANN3D(num=36, k=0.1, z_min=-180, z_max=180, a=10, tau=1)

    state_x_list, state_y_list, state_z_list = [], [], []
    input_x_list, input_y_list, input_z_list = [], [], []

    Iext_encoded_list, cann_state_list = [], []

    gird_x_y_z_history = np.loadtxt(gc_input_path)

    for i in range(len(gird_x_y_z_history)):
        delta_x, delta_y, delta_z = gird_x_y_z_history[i]

        Iext = np.array([delta_x, delta_y, delta_z]).reshape(1, 3)
        Iext_encoded = cann.get_stimulus_by_pos(Iext)  # (num, num, num)

        runner = bp.DSRunner(
            cann,
            inputs=['input', Iext_encoded, 'iter'],
            monitors=['u'],
            dyn_vars=cann.vars()
        )
        runner.run(0.1)

        u = runner.mon.u.squeeze()  # (num, num, num)
        Iext_encoded = Iext_encoded.squeeze()  # (num, num, num)
        Iext_encoded_list.append(Iext_encoded)
        cann_state_list.append(u)

        state_x_list.append(np.sum(u, axis=(1, 2)))  # (,num)
        state_y_list.append(np.sum(u, axis=(0, 2)))  # (,num)
        state_z_list.append(np.sum(u, axis=(0, 1)))  # (,num)

        input_x_list.append(np.sum(Iext_encoded, axis=(1, 2)))
        input_y_list.append(np.sum(Iext_encoded, axis=(0, 2)))
        input_z_list.append(np.sum(Iext_encoded, axis=(0, 1)))

        print(f"Step {i + 1}/{len(gird_x_y_z_history)} processed.", end='\r')

    print("\nSimulation finished.")

    print(
        f"Iext_encoded shape: {np.array(Iext_encoded_list).shape}, 3D_cann_state shape: {np.array(cann_state_list).shape}")

    cann_state_x = np.array(state_x_list)
    cann_state_y = np.array(state_y_list)
    cann_state_z = np.array(state_z_list)

    Iext_encoded_x = np.array(input_x_list)
    Iext_encoded_y = np.array(input_y_list)
    Iext_encoded_z = np.array(input_z_list)

    print(f"Iext_encoded_x shape: {np.array(Iext_encoded_x).shape}, "
          f"cann_state_x shape: {np.array(cann_state_x).shape}",
          f"Iext_encoded_y shape: {np.array(Iext_encoded_y).shape}, "
          f"cann_state_y shape: {np.array(cann_state_y).shape}",
          f"Iext_encoded_z shape: {np.array(Iext_encoded_z).shape}, "
          f"cann_state_z shape: {np.array(cann_state_z).shape}")

    np.savetxt(save_cann_x, cann_state_x)
    np.savetxt(save_cann_y, cann_state_y)
    np.savetxt(save_cann_z, cann_state_z)

    np.savetxt(save_input_x, Iext_encoded_x)
    np.savetxt(save_input_y, Iext_encoded_y)
    np.savetxt(save_input_z, Iext_encoded_z)

    print(f"All states saved to {gc_state_save_dir}")

    return (Iext_encoded_x, Iext_encoded_y, Iext_encoded_z,
            cann_state_x, cann_state_y, cann_state_z)


def train_esn(x_data, y_data, gc_model_save_path):
    split_index = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split_index], y_data[:split_index]
    x_test, y_test = x_data[split_index:], y_data[split_index:]

    if all(os.path.exists(p) for p in
           [gc_model_save_path]):
        print("Found cached models, loading...")

        with open(gc_model_save_path, "rb") as f:
            esn_x = pickle.load(f)

        print("esn_x model", esn_x)

        pred_x = esn_x.run(x_test[:, :, 0])
        pred_y = esn_x.run(x_test[:, :, 1])
        pred_z = esn_x.run(x_test[:, :, 2])
        predictions = np.stack([pred_x, pred_y, pred_z], axis=2)

        print(f"RMSE (x): {rmse(y_test[:, :, 0], pred_x)}")
        print(f"RMSE (y): {rmse(y_test[:, :, 1], pred_y)}")
        print(f"RMSE (z): {rmse(y_test[:, :, 2], pred_z)}")
        print(f"R^2 (x): {rsquare(y_test[:, :, 0], pred_x)}")
        print(f"R^2 (y): {rsquare(y_test[:, :, 1], pred_y)}")
        print(f"R^2 (z): {rsquare(y_test[:, :, 2], pred_z)}")

        print(f"x_test shape : {x_test.shape}, y_test shape: {y_test.shape}, predictions shape: {predictions.shape}")

        return x_test, y_test, predictions

    reservoir = Reservoir(units=300, sr=0.01, lr=0.1)

    readout_x = Ridge(ridge=1e-5)

    esn_x = reservoir >> readout_x

    esn_x.fit(x_train[:, :, 0], y_train[:, :, 0])

    pred_x = esn_x.run(x_test[:, :, 0])
    pred_y = esn_x.run(x_test[:, :, 1])
    pred_z = esn_x.run(x_test[:, :, 2])
    predictions = np.stack([pred_x, pred_y, pred_z], axis=2)

    print(f"RMSE (x): {rmse(y_test[:, :, 0], pred_x)}")
    print(f"RMSE (y): {rmse(y_test[:, :, 1], pred_y)}")
    print(f"RMSE (z): {rmse(y_test[:, :, 2], pred_z)}")
    print(f"R^2 (x): {rsquare(y_test[:, :, 0], pred_x)}")
    print(f"R^2 (y): {rsquare(y_test[:, :, 1], pred_y)}")
    print(f"R^2 (z): {rsquare(y_test[:, :, 2], pred_z)}")

    with open(gc_model_save_path, "wb") as f:
        pickle.dump(esn_x, f)
    print(f"Model saved to {gc_model_save_path}")

    print(f"x_test shape : {x_test.shape}, y_test shape: {y_test.shape}, predictions shape: {predictions.shape}")

    return x_test, y_test, predictions


def train_esn_all(x_data, y_data):
    split_index = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split_index], y_data[:split_index]
    x_test, y_test = x_data[split_index:], y_data[split_index:]

    reservoir = Reservoir(units=100, sr=0.8, lr=0.2)

    readout = Ridge(ridge=1e-5)
    esn = reservoir >> readout

    esn.fit(x_train, y_train)

    pred = esn.run(x_test)

    print(f"RMSE (x): {rmse(y_test, pred)}")
    print(f"R^2 (x): {rsquare(y_test, pred)}")

    with open("model/esn_model_xyz.pkl", "wb") as f:
        pickle.dump(esn, f)
    print("Model saved to model/esn_model_xyz.pkl")

    return x_test, y_test, np.array(pred)


def plot_3d_results(y_true, y_pred):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(y_true[:, :, 0], y_true[:, :, 1], y_true[:, :, 2], label="True trajectory", c='b', s=10)
    ax.scatter(y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2], label="Predicted trajectory", c='r', s=10, marker='^')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def plot_cann_state_and_input(Iext_encoded_x, Iext_encoded_y, Iext_encoded_z, cann_state_x, cann_state_y, cann_state_z):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(cann_state_x[0, :], label='CANN State X', color='b')
    ax[0].plot(Iext_encoded_x[0, :], label='Input X', color='g', linestyle='--')
    ax[0].legend()
    ax[0].set_title('CANN State and Input Projection on X-axis')
    ax[1].plot(cann_state_y[0, :], label='CANN State Y', color='b')
    ax[1].plot(Iext_encoded_y[0, :], label='Input Y', color='g', linestyle='--')
    ax[1].legend()
    ax[1].set_title('CANN State and Input Projection on Y-axis')
    ax[2].plot(cann_state_z[0, :], label='CANN State Z', color='b')
    ax[2].plot(Iext_encoded_z[0, :], label='Input Z', color='g', linestyle='--')
    ax[2].legend()
    ax[2].set_title('CANN State and Input Projection on Z-axis')
    plt.tight_layout()
    plt.show()


def plot_xyz_errors(y_true, y_pred):
    errors = np.abs(y_true - y_pred)

    global_errors = errors.mean(axis=1)

    std_errors = errors.std(axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    x = np.arange(len(global_errors))

    ax[0].plot(x, global_errors[:, 0], label='Mean Error X', color='r')
    ax[0].fill_between(
        x,
        global_errors[:, 0] - std_errors[:, 0],
        global_errors[:, 0] + std_errors[:, 0],
        color='r',
        alpha=0.2,
    )
    ax[0].set_title('Prediction Error on X-axis')
    ax[0].legend()
    ax[0].set_xlabel('Sample Index')
    ax[0].set_ylabel('Absolute Error')

    # 绘制Y轴误差曲线及标准差区间
    ax[1].plot(x, global_errors[:, 1], label='Mean Error Y', color='b')
    ax[1].fill_between(
        x,
        global_errors[:, 1] - std_errors[:, 1],
        global_errors[:, 1] + std_errors[:, 1],
        color='b',
        alpha=0.2,
    )
    ax[1].set_title('Prediction Error on Y-axis')
    ax[1].legend()
    ax[1].set_xlabel('Sample Index')
    ax[1].set_ylabel('Absolute Error')

    ax[2].plot(x, global_errors[:, 2], label='Mean Error Z', color='c')
    ax[2].fill_between(
        x,
        global_errors[:, 2] - std_errors[:, 2],
        global_errors[:, 2] + std_errors[:, 2],
        color='c',
        alpha=0.2,
    )
    ax[2].set_title('Prediction Error on Z-axis')
    ax[2].legend()
    ax[2].set_xlabel('Sample Index')
    ax[2].set_ylabel('Absolute Error')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main(gc_input_path, gc_state_save_dir, model_save_path):
    Iext_encoded_x, Iext_encoded_y, Iext_encoded_z, cann_state_x, cann_state_y, cann_state_z = neuroslam_3D_data(
        gc_input_path, gc_state_save_dir)
    plot_cann_state_and_input(Iext_encoded_x, Iext_encoded_y, Iext_encoded_z, cann_state_x, cann_state_y, cann_state_z)

    x_data = np.stack([Iext_encoded_x, Iext_encoded_y, Iext_encoded_z], axis=2)
    y_data = np.stack([cann_state_x, cann_state_y, cann_state_z], axis=2)
    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")

    x_test, y_test, predictions = train_esn(x_data, y_data, model_save_path)
    print(f"y_test shape: {y_test.shape}, predictions shape: {predictions.shape}")
    plot_3d_results(y_test, predictions)
    plot_xyz_errors(y_test, predictions)


if __name__ == "__main__":
    # dataset = "KITTI_00"
    dataset = "Self"

    # hdc
    hdc_input_path = f"../1D_data/{dataset}/temp_yaw_height.txt"
    hdc_state_save_path = f"../1D_data/{dataset}/hdc_cann_state.txt"
    hdc_model_save_path = f"./{dataset}/hdc_esn_model.pkl"
    neuroslam_data(hdc_input_path, hdc_state_save_path, hdc_model_save_path)

    # # gc
    # gc_input_path = f"../3D_data/{dataset}/gird_x_y_z_history.txt"
    # gc_state_save_dir = f"../3D_data/{dataset}/"
    # model_save_path = f"./{dataset}/gc_esn_model.pkl"
    # main(gc_input_path, gc_state_save_dir, model_save_path)
