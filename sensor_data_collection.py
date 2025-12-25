import RPi.GPIO as GPIO
import time
import csv
from datetime import datetime

# 传感器引脚配置（DO为数字输出）
SENSOR_PIN = 18  # 树莓派GPIO18引脚
DATA_FILE = "sensor_data.csv"  # 数据存储文件

def setup_sensor():
    """初始化传感器"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SENSOR_PIN, GPIO.IN)  # 配置为输入模式

def read_moisture():
    """读取土壤湿度（数字输出：0=湿润，1=干燥）"""
    # 若使用模拟输出（AO），需通过ADC模块读取，此处以数字输出为例
    moisture_value = GPIO.input(SENSOR_PIN)
    # 转换为相对湿度（示例映射，需根据实际传感器校准）
    relative_humidity = 100 - (moisture_value * 100)  # 仅为示例，实际需校准
    return relative_humidity

def collect_data(duration_hours=48, interval_minutes=30):
    """持续采集数据：duration_hours=采集时长（小时），interval_minutes=采集间隔（分钟）"""
    setup_sensor()
    total_intervals = int((duration_hours * 60) / interval_minutes)
    
    # 创建CSV文件并写入表头
    with open(DATA_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "relative_humidity"])  # 时间戳、相对湿度
    
    try:
        for i in range(total_intervals):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            humidity = read_moisture()
            # 写入数据
            with open(DATA_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, round(humidity, 2)])
            print(f"采集完成 {i+1}/{total_intervals} | 时间：{timestamp} | 湿度：{humidity:.2f}%")
            time.sleep(interval_minutes * 60)  # 间隔等待
    except KeyboardInterrupt:
        print("数据采集被手动终止")
    finally:
        GPIO.cleanup()  # 清理GPIO引脚

if __name__ == "__main__":
    print("开始土壤湿度数据采集...")
    collect_data(duration_hours=48, interval_minutes=30)  # 采集48小时，每30分钟1次
    print("数据采集完成，文件已保存至：", DATA_FILE)
