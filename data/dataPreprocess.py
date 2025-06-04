import os
import pandas as pd
from data.qlib_dump_bin import DumpDataAll  # ← 你已将 dump_bin.py 中类提取为 data/qlib_dump_bin.py

def convert_csv(csv_file, output_csv_file):
    """
    将原始CSV转换为Qlib标准字段格式
    """
    df = pd.read_csv(csv_file)
    df.rename(columns={
        "ts_code": "instrument",
        "trade_date": "datetime",
        "vol": "volume",
    }, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df.sort_values(by=["instrument", "datetime"], inplace=True)
    df.to_csv(output_csv_file, index=False)
    print(f"✅ 已保存转换后的文件：{output_csv_file}")
    return output_csv_file

def run_dump_bin(processed_csv_path, output_qlib_path):
    """
    使用 DumpDataAll 构建 Qlib 本地数据
    """
    print("🚀 正在构建 Qlib 数据，请稍候...")
    dumpper = DumpDataAll(
        csv_path=processed_csv_path,
        qlib_dir=output_qlib_path,
        freq="day",
        date_field_name="datetime",
        symbol_field_name="instrument",
        exclude_fields = "instrument,datetime"
    )
    dumpper()
    print(f"✅ Qlib 数据已构建至：{output_qlib_path}")


if __name__ == "__main__":
    # === 参数配置 ===
    filename = "不锈钢-SHF.csv"
    base_path = os.path.abspath(os.path.dirname(__file__))

    csv_path = os.path.join(base_path, filename)
    processed_path = os.path.join(base_path, f"processed_{filename}")
    output_dir = os.path.join(base_path, "qlib_local_data")

    # === 执行流程 ===
    convert_csv(csv_path, processed_path)
    run_dump_bin(processed_path, output_dir)
