import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path="sensor_data.csv"):
    """加载采集的原始数据"""
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # 转换时间戳格式
    df.set_index("timestamp", inplace=True)  # 设时间为索引
    return df

def denoise_moving_average(df, window_size=3):
    """移动平均去噪：window_size=窗口大小（默认3个数据点）"""
    df["denoised_humidity"] = df["relative_humidity"].rolling(window=window_size, center=True).mean()
    df = df.dropna()  # 删除空值
    return df

def normalize_data(df):
    """归一化：将湿度数据缩放至[0,1]区间"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["normalized_humidity"] = scaler.fit_transform(df[["denoised_humidity"]])
    return df, scaler

def preprocess_pipeline(file_path="sensor_data.csv"):
    """预处理流水线：加载→去噪→归一化"""
    # 1. 加载数据
    df = load_data(file_path)
    # 2. 去噪（移动平均法）
    df = denoise_moving_average(df)
    # 3. 归一化
    df, scaler = normalize_data(df)
    # 保存预处理后的数据
    df.to_csv("preprocessed_data.csv")
    print("预处理完成，文件已保存至：preprocessed_data.csv")
    return df, scaler

if __name__ == "__main__":
    preprocessed_df, scaler = preprocess_pipeline()
    print("预处理后的数据预览：")
    print(preprocessed_df.head())
