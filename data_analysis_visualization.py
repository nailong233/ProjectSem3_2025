import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体（若演示需英文，可注释此行）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_preprocessed_data(file_path="preprocessed_data.csv"):
    """加载预处理后的数据"""
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def descriptive_statistics(df):
    """计算描述性统计量"""
    stats = df[["relative_humidity", "denoised_humidity", "normalized_humidity"]].describe()
    print("描述性统计结果：")
    print(stats)
    stats.to_csv("descriptive_statistics.csv")  # 保存统计结果
    return stats

def plot_time_series(df):
    """绘制时间序列趋势图（原始湿度vs去噪湿度）"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["relative_humidity"], label="原始湿度", color="lightgray", alpha=0.7)
    plt.plot(df.index, df["denoised_humidity"], label="去噪湿度（移动平均）", color="blue", linewidth=2)
    plt.xlabel("时间")
    plt.ylabel("相对湿度（%）")
    plt.title("土壤湿度时间序列趋势")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 格式化x轴时间显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))  # 每6小时显示1个刻度
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("humidity_time_series.png", dpi=300)
    plt.show()
    print("时间序列图已保存至：humidity_time_series.png")

if __name__ == "__main__":
    df = load_preprocessed_data()
    # 计算描述性统计
    descriptive_statistics(df)
    # 绘制趋势图
    plot_time_series(df)
