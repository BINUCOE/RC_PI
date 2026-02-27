import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import os


class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TASK_TYPE = "regression"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 500
    WARMUP = 10
    N_SAMPLES = 1000
    TIMESTEPS = 1
    INPUT_FEAT_DIM = 36
    RESERVOIR_UNITS = 36
    MLP_HIDDEN1 = 12
    MLP_OUTPUT_DIM = 2
    DATASET = "KITTI_00"
    MODEL_SAVE_PATH = f"{DATASET}/reservoir_mlp_hdc_model.pth"
    DIM = "1D"



class Config_3D:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TASK_TYPE = "regression"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 500
    WARMUP = 10
    N_SAMPLES = 1000
    TIMESTEPS = 1
    INPUT_FEAT_DIM = 36
    RESERVOIR_UNITS = 36
    MLP_HIDDEN1 = 12
    MLP_OUTPUT_DIM = 6
    DATASET = "KITTI_00"
    MODEL_SAVE_PATH = f"{DATASET}/reservoir_mlp_gc_model.pth"
    DIM = "3D"


np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
print(f"[Config] Test Configuration: Device={Config.DEVICE}, Task Type={Config.TASK_TYPE}, Time Steps={Config.TIMESTEPS}")


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, config: Config):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tensor_dict = {
        "X_train": torch.tensor(X_train_scaled, dtype=torch.float32).to(config.DEVICE),
        "y_train": torch.tensor(y_train, dtype=torch.float32).to(config.DEVICE),
        "X_test": torch.tensor(X_test_scaled, dtype=torch.float32).to(config.DEVICE),
        "y_test": torch.tensor(y_test, dtype=torch.float32).to(config.DEVICE)
    }

    train_dataset = TensorDataset(tensor_dict["X_train"], tensor_dict["y_train"])
    test_dataset = TensorDataset(tensor_dict["X_test"], tensor_dict["y_test"])

    dataloader_dict = {
        "train": DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    }

    print(f"[Preprocess] Data Preprocessing Completed: Training Batches={len(dataloader_dict['train'])}")
    return scaler, dataloader_dict, tensor_dict


class ReservoirMLP(nn.Module):
    def __init__(self, config: Config):
        super(ReservoirMLP, self).__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(in_features=config.RESERVOIR_UNITS, out_features=config.MLP_HIDDEN1),
            nn.ReLU(),
            nn.Linear(in_features=config.MLP_HIDDEN1, out_features=config.MLP_OUTPUT_DIM)
        )

    def forward(self, x):
        return self.decoder_net(x)


def build_model(config: Config):
    model = ReservoirMLP(config).to(config.DEVICE)
    print("\n[Model] Model Structure:")
    print(model)
    return model


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, config: Config):
    model.train()
    loss_history = []

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            if config.TASK_TYPE == "classification":
                loss = criterion(outputs, batch_y.argmax(dim=1))
            else:
                loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Train] Epoch [{epoch + 1}/{config.EPOCHS}], Average Loss: {avg_loss:.6f}")

    return loss_history, model


def save_model(model: nn.Module, save_path: str):
    torch.save(model.state_dict(), save_path)
    print(f"\n[Save] Model Saved to: {save_path}")


def load_model(config: Config, save_path: str):
    model = ReservoirMLP(config).to(config.DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=config.DEVICE))
    model.eval()
    print(f"[Load] Model Loaded from {save_path}")
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, config: Config):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    return all_preds, all_trues


def predict_with_loaded_model(model: nn.Module, X_test_ts: np.ndarray, config: Config):
    X_test_tensor = torch.tensor(X_test_ts, dtype=torch.float32).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()

    return y_pred


def visualize_results(loss_history: list, y_true: np.ndarray, y_pred: np.ndarray,
                      X_test_features: np.ndarray, config: Config):
    errors = y_true - y_pred
    test_sample_num = y_true.shape[0]
    sample_indices = np.arange(test_sample_num)

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.plot(range(1, config.EPOCHS + 1), loss_history, 'b-', linewidth=1.5)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Training Loss", fontsize=10)
    plt.title("1. Training Loss Curve", fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.plot(sample_indices, y_true[:, 0], 'b-', label='True Value', linewidth=1.5, alpha=0.8)
    plt.plot(sample_indices, y_pred[:, 0], 'orange', label='Predicted Value', linewidth=1.2, linestyle='--')
    plt.xlabel("Sample Index (Test Set)", fontsize=10)
    plt.ylabel("Output Value", fontsize=10)
    plt.title("2. Output Dim 1: True vs Predicted (Sample Order)", fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.plot(sample_indices, errors[:, 0], 'r-', linewidth=1.2, alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, label='Zero Error')
    plt.xlabel("Sample Index (Test Set)", fontsize=10)
    plt.ylabel("Error (True - Predicted)", fontsize=10)
    plt.title("3. Output Dim 1: Prediction Error (Residual)", fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, c='g', s=30)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()],
             [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', linewidth=1.5)
    plt.xlabel("True Value (Output Dim 1)", fontsize=10)
    plt.ylabel("Predicted Value (Output Dim 1)", fontsize=10)
    plt.title("4. Output Dim 1: True vs Predicted (Scatter)", fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 5)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, c='orange', s=30)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()],
             [y_true[:, 1].min(), y_true[:, 1].max()], 'r--', linewidth=1.5)
    plt.xlabel("True Value (Output Dim 2)", fontsize=10)
    plt.ylabel("Predicted Value (Output Dim 2)", fontsize=10)
    plt.title("5. Output Dim 2: True vs Predicted (Scatter)", fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 6)
    corr = np.corrcoef(X_test_features.T, y_pred[:, 0])[:-1, -1]
    plt.bar(range(config.RESERVOIR_UNITS), corr, color='skyblue', alpha=0.7)
    plt.xlabel(f"Reservoir Feature Index (0-{config.RESERVOIR_UNITS - 1})", fontsize=10)
    plt.ylabel("Correlation with Output Dim 1", fontsize=10)
    plt.title("6. Feature-Output Correlation (Reservoir)", fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.tight_layout()

    plt.savefig(f"{config.DATASET}/train_visual_{config.DIM}.png", dpi=300, bbox_inches='tight')

    plt.show()


def main_1D(hdc_input_state, hdc_target):
    print("\n[Main] Loading Data...")
    cann_state = np.loadtxt(hdc_input_state)
    X_train, X_test = train_test_split(cann_state, test_size=0.2, random_state=Config.SEED)

    yaw_sum_sin_cos = np.loadtxt(hdc_target)

    y_train, y_test = train_test_split(yaw_sum_sin_cos, test_size=0.2, random_state=Config.SEED)

    scaler, dataloader_dict, tensor_dict = preprocess_data(
        X_train, X_test, y_train, y_test, Config
    )

    model = build_model(Config)

    if Config.TASK_TYPE == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print("\n[Main] Start Training...")
    train_loss, model = train_model(
        model, dataloader_dict["train"], criterion, optimizer, Config
    )

    save_model(model, Config.MODEL_SAVE_PATH)

    print("\n[Main] Evaluating Original Model...")
    y_pred, y_true = evaluate_model(model, dataloader_dict["test"], Config)
    if Config.TASK_TYPE == "regression":
        mse = mean_squared_error(y_true, y_pred)
        print(f"[Evaluate] Original Model Test Set MSE: {mse:.6f}")
    else:
        acc = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        print(f"[Evaluate] Original Model Test Set Accuracy: {acc:.4f}")

    print("\n[Main] Evaluating Loaded Model...")
    loaded_model = load_model(Config, Config.MODEL_SAVE_PATH)
    y_pred_loaded = predict_with_loaded_model(loaded_model, X_test, Config)
    if Config.TASK_TYPE == "regression":
        loaded_mse = mean_squared_error(y_test, y_pred_loaded)
        print(f"[Evaluate] Loaded Model Test Set MSE: {loaded_mse:.6f}")
    else:
        loaded_acc = accuracy_score(y_test.argmax(axis=1), y_pred_loaded.argmax(axis=1))
        print(f"[Evaluate] Loaded Model Test Set Accuracy: {loaded_acc:.4f}")

    visualize_results(train_loss, y_true, y_pred, X_test, Config)


def main_3D(gc_input_state, gc_target):
    print("\n[Main] Loading Data...")
    cann_state = np.loadtxt(gc_input_state)
    print("cann_state shape:", cann_state.shape)
    X_train, X_test = train_test_split(cann_state, test_size=0.2, random_state=Config_3D.SEED)

    x_sum_sin_cos = np.loadtxt(os.path.join(gc_target, "x_sum_sin_cos.txt"))
    y_sum_sin_cos = np.loadtxt(os.path.join(gc_target, "y_sum_sin_cos.txt"))
    z_sum_sin_cos = np.loadtxt(os.path.join(gc_target, "z_sum_sin_cos.txt"))
    xyz = np.hstack((x_sum_sin_cos, y_sum_sin_cos, z_sum_sin_cos))

    print("xyz shape:", xyz.shape)

    y_train, y_test = train_test_split(xyz, test_size=0.2, random_state=Config_3D.SEED)

    scaler, dataloader_dict, tensor_dict = preprocess_data(
        X_train, X_test, y_train, y_test, Config_3D
    )

    model = build_model(Config_3D)

    if Config_3D.TASK_TYPE == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config_3D.LEARNING_RATE)

    print("\n[Main] Start Training...")
    train_loss, model = train_model(
        model, dataloader_dict["train"], criterion, optimizer, Config_3D
    )

    save_model(model, Config_3D.MODEL_SAVE_PATH)

    print("\n[Main] Evaluating Original Model...")
    y_pred, y_true = evaluate_model(model, dataloader_dict["test"], Config_3D)
    if Config_3D.TASK_TYPE == "regression":
        mse = mean_squared_error(y_true, y_pred)
        print(f"[Evaluate] Original Model Test Set MSE: {mse:.6f}")
    else:
        acc = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        print(f"[Evaluate] Original Model Test Set Accuracy: {acc:.4f}")

    print("\n[Main] Evaluating Loaded Model...")
    loaded_model = load_model(Config_3D, Config_3D.MODEL_SAVE_PATH)
    y_pred_loaded = predict_with_loaded_model(loaded_model, X_test, Config_3D)
    if Config_3D.TASK_TYPE == "regression":
        loaded_mse = mean_squared_error(y_test, y_pred_loaded)
        print(f"[Evaluate] Loaded Model Test Set MSE: {loaded_mse:.6f}")
    else:
        loaded_acc = accuracy_score(y_test.argmax(axis=1), y_pred_loaded.argmax(axis=1))
        print(f"[Evaluate] Loaded Model Test Set Accuracy: {loaded_acc:.4f}")

    visualize_results(train_loss, y_true, y_pred, X_test, Config_3D)


if __name__ == "__main__":
    dataset = "KITTI_00"


    # # main_1D()
    # Config.DATASET = dataset
    # # Config.LEARNING_RATE = 1e-4
    # Config.MODEL_SAVE_PATH = f"{dataset}/reservoir_mlp_hdc_model.pth"
    # hdc_input_state = f"../1D_data/{dataset}/hdc_cann_state.txt"
    # hdc_target = f"../1D_data/{dataset}/yaw_sum_sin_cos.txt"
    # main_1D(hdc_input_state, hdc_target)

    # main_3D()
    Config_3D.DATASET = dataset
    Config_3D.LEARNING_RATE = 1e-2
    Config_3D.MODEL_SAVE_PATH = f"{dataset}/reservoir_mlp_gc_model.pth"
    gc_input_state = f"../3D_data/{dataset}/cann_state_x.txt"
    gc_target = f"../3D_data/{dataset}"
    main_3D(gc_input_state, gc_target)



