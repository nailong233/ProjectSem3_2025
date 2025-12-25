import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def prepare_linear_regression_data(df, forecast_hours=12, interval_minutes=30):
    """准备线性回归数据：用历史数据预测未来12小时湿度"""
    # 计算需要的历史数据点数量（12小时 / 30分钟 = 24个点）
    n_steps = int(forecast_hours * 60 / interval_minutes)
    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df["normalized_humidity"].iloc[i-n_steps:i].values)  # 历史n_steps个数据
        y.append(df["normalized_humidity"].iloc[i])  # 目标：第i个数据点
    return np.array(X), np.array(y), n_steps

def linear_regression_model(X_train, y_train, X_test, y_test, scaler):
    """训练线性回归模型并评估"""
    # 训练模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    # 预测
    y_pred = lr_model.predict(X_test)
    # 反归一化（还原为真实湿度值）
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # 评估指标
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    print("线性回归模型评估：")
    print(f"均方误差（MSE）：{mse:.4f}")
    print(f"决定系数（R²）：{r2:.4f}")
    # 可视化预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual, label="真实湿度", color="blue")
    plt.plot(y_pred_actual, label="预测湿度", color="red", linestyle="--")
    plt.xlabel("测试数据点索引")
    plt.ylabel("相对湿度（%）")
    plt.title("线性回归模型：12小时湿度预测结果")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("linear_regression_prediction.png", dpi=300)
    plt.show()
    return lr_model, mse, r2

def create_lstm_dataset(data, n_steps=24):
    """创建LSTM输入数据集：n_steps=时间步长（默认24个点=12小时）"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])  # 历史n_steps个数据
        y.append(data[i, 0])  # 目标：第i个数据点
    return np.array(X), np.array(y)

def lstm_model(X_train, y_train, X_test, y_test, scaler, n_steps=24):
    """训练轻量LSTM模型并评估"""
    # 重塑输入数据：[样本数, 时间步长, 特征数]（LSTM要求的输入格式）
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(n_steps, 1)))  # 50个LSTM单元
    model.add(Dense(1))  # 输出层
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)
    # 预测
    y_pred = model.predict(X_test)
    # 反归一化
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # 评估指标
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    print("LSTM模型评估：")
    print(f"均方误差（MSE）：{mse:.4f}")
    print(f"决定系数（R²）：{r2:.4f}")
    # 可视化预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual, label="真实湿度", color="blue")
    plt.plot(y_pred_actual, label="预测湿度", color="green", linestyle="--")
    plt.xlabel("测试数据点索引")
    plt.ylabel("相对湿度（%）")
    plt.title("LSTM模型：周期性湿度预测结果")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstm_prediction.png", dpi=300)
    plt.show()
    # 保存模型
    model.save("lstm_humidity_model.h5")
    print("LSTM模型已保存至：lstm_humidity_model.h5")
    return model, mse, r2

if __name__ == "__main__":
    # 加载预处理后的数据和归一化器
    preprocessed_df = pd.read_csv("preprocessed_data.csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(preprocessed_df[["denoised_humidity"]])  # 复用之前的归一化器
    
    # 准备数据（80%训练集，20%测试集）
    data = preprocessed_df["normalized_humidity"].values.reshape(-1, 1)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size, :]
    test_data = data[train_size:, :]
    
    # 1. 线性回归模型（12小时预测）
    X_lr_train, y_lr_train, n_steps = prepare_linear_regression_data(preprocessed_df[:train_size])
    X_lr_test, y_lr_test, _ = prepare_linear_regression_data(preprocessed_df[train_size:], n_steps=n_steps)
    linear_regression_model(X_lr_train, y_lr_train, X_lr_test, y_lr_test, scaler)
    
    # 2. LSTM模型（周期性预测）
    X_lstm_train, y_lstm_train = create_lstm_dataset(train_data, n_steps=n_steps)
    X_lstm_test, y_lstm_test = create_lstm_dataset(test_data, n_steps=n_steps)
    lstm_model(X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test, scaler, n_steps=n_steps)
