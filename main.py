import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from args import parse_args
# Custom imports
import iTransformer
from utils.timefeatures import time_features

plt.rc('font', family='Arial')
plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========================
# Random Seed Fix Function
# ========================
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Data Loader for Time Series
# =============================
def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


# =============================
# Model Training and Validation
# =============================
def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device, early_patience=0.15, print_train=False):
    start_time = time.time()
    train_loss, val_loss = [], []
    print_frequency = max(1, int(num_epochs / 20))
    early_patience_epochs = int(early_patience * num_epochs)
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        for datapoints, labels, datapoints_mark, labels_mark in train_loader:
            datapoints, labels = datapoints.to(device), labels.to(device)
            datapoints_mark, labels_mark = datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()
            labels = labels[:, -length_size:].squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        net.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_x_mark, val_y_mark = val_x_mark.to(device), val_y_mark.to(device)
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()
                val_y = val_y[:, -length_size:].squeeze()
                val_loss_batch = criterion(pred_val_y, val_y)
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            scheduler.step(avg_val_loss)

        if print_train and ((epoch + 1) % print_frequency == 0 or epoch == num_epochs - 1):
            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    net.train()
    return net, train_loss, val_loss, epoch + 1


# ====================
# Evaluation Metrics
# ====================
def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100

    df_eval = pd.DataFrame({
        'R2': [r2],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'MAPE': [mape]
    }, index=['Eval'])

    return df_eval


# ===============
# Main Pipeline
# ===============
if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.seed)

    # === Data Loading ===
    if args.dataset == 'SRU':
        df = pd.read_csv(args.SRU_path, sep=',')
        data_target = df['7']  # 预测的目标变量
    elif args.dataset == 'Debutanizer':
        df = pd.read_csv(args.Debutanizer_path, sep='\s+')
        data_target = df['y']
    else:
        raise ValueError(
            "Dataset setting error: please check the 'args.dataset' value. Supported options are 'SRU' and 'Debutanizer'.")

    data_dim = df[df.columns.drop('date')].shape[1]
    data = df[df.columns.drop('date')]

    df_stamp = df[['date']]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    data_stamp = time_features(df_stamp, timeenc=1, freq='B')

    scaler = MinMaxScaler()
    data_inverse = scaler.fit_transform(np.array(data))
    data_length = len(data_inverse)

    train_ratio, val_ratio = 0.6, 0.8
    train_size = int(data_length * train_ratio)
    val_size = int(data_length * val_ratio)
    data_train = data_inverse[:train_size, :]
    data_val = data_inverse[train_size: val_size, :]
    data_test = data_inverse[val_size:, :]
    data_train_mark = data_stamp[:train_size, :]
    data_val_mark = data_stamp[train_size: val_size, :]
    data_test_mark = data_stamp[val_size:, :]

    if args.dataset == 'SRU':
        args.window = 3
    elif args.dataset == 'Debutanizer':
        args.window = 5

    # === Data Loader ===
    train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(
        args.window, args.length_size, args.batch_size, data_train, data_train_mark)
    val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(
        args.window, args.length_size, args.batch_size, data_val, data_val_mark)
    test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(
        args.window, args.length_size, args.batch_size, data_test, data_test_mark)

    # === Device & Update args ===
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.seq_len = args.window
    args.label_len = int(args.window / 2)
    args.pred_len = args.length_size
    args.enc_in = data_dim
    args.dec_in = data_dim
    args.c_out = 1
    args.task_name = 'short_term_forecast'
    args.embed = 'timeF'
    args.freq = 'b'
    args.output_attention = 0
    args.num_class = 1

    # === Model, Optimizer, Scheduler ===
    net = iTransformer.Model(args).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler_patience = int(args.scheduler_patience * args.epochs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)

    # === Training ===
    trained_model, train_loss, val_loss, final_epoch = model_train_val(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        length_size=args.length_size,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        early_patience=args.early_patience,
        print_train=True
    )

    # === Prediction & Inverse Transform ===
    trained_model.eval()
    pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
    true = y_test[:, -args.length_size:, -1:].detach().cpu()
    pred = pred.detach().cpu()

    true = true[:, :, -1]
    pred = pred[:, :, -1]

    # Inverse transform
    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    pred_uninverse = scaler.inverse_transform(pred[:, -1:])
    true_uninverse = scaler.inverse_transform(true[:, -1:])

    true, pred = true_uninverse, pred_uninverse

    # === Evaluation & Plot ===
    df_eval = cal_eval(true, pred)
    print(df_eval)

    df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
    df_pred_true.plot(figsize=(12, 4))
    plt.title('DMLSTM+iTransformer Result')
    plt.show()

    result_df = pd.DataFrame({'real': true.flatten(), 'predict': pred.flatten()})
    result_df.to_csv(str(args.dataset) + '.csv', index=False, encoding='utf-8')
    print('saved results of ' + str(args.dataset) + '.csv')

