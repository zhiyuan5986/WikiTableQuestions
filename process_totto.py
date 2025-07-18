import json
import os
import datasets
import pandas as pd
from tqdm import tqdm

dataset = datasets.load_dataset(
    "json",
    data_files={
        "train": "/home/qiaoan/data/totto_data/totto_train_data.jsonl",
        # "validation": "/home/qiaoan/data/totto_data/totto_dev_data.jsonl",
    },
)
train_dataset = dataset["train"]
print("First train sample:", train_dataset[0]['table'])
# validation_dataset = dataset["validation"]
# print("Number of validation samples:", len(validation_dataset))
# print("First validation sample:", validation_dataset[0])


def to_dataframe(totto_table):
    """
    将 TOTTO 表格转换为严格对齐的 pandas.DataFrame。
    
    规则：
    - 展开 row_span / column_span；
    - 取最后一行 header；
    - 删除表头为空的列（不填充，不命名）；
    - 将内容为 "Vacant" 的单元格替换为空值；
    - 丢弃所有列数 != 原始 header 列数 的 data 行；
    - 返回严格对齐的 DataFrame。
    """
    raw_table = totto_table["table"]

    def get_true_max_cols(rows):
        return max(
            (sum(cell.get("column_span", 1) for cell in row) for row in rows),
            default=0,
        )

    def expand_table(rows, max_cols):
        grid = []
        span_map = {}  # {(src_row, col): (val, rows_left)}

        for row_idx, row in enumerate(rows):
            current_row = [None] * max_cols
            row_flags = [False] * max_cols

            # 补上跨行的单元格
            for (r, c), (val, rspan_left) in list(span_map.items()):
                if r + 1 == row_idx and c < max_cols:
                    current_row[c] = val
                    row_flags[c] = True
                    if rspan_left > 1:
                        span_map[(row_idx, c)] = (val, rspan_left - 1)
                    del span_map[(r, c)]

            # 当前行展开
            col_ptr = 0
            for cell in row:
                val = cell["value"]
                r_span = cell.get("row_span", 1)
                c_span = cell.get("column_span", 1)

                # 替换 Vacant
                if isinstance(val, str) and val.strip() == "Vacant":
                    val = None

                while col_ptr < max_cols and row_flags[col_ptr]:
                    col_ptr += 1

                for offset in range(c_span):
                    idx = col_ptr + offset
                    if idx >= max_cols:
                        break
                    current_row[idx] = val
                    row_flags[idx] = True
                    if r_span > 1:
                        span_map[(row_idx, idx)] = (val, r_span - 1)
                col_ptr += c_span

            grid.append(current_row)

        return grid

    # 拆分 header / data
    header_rows = []
    data_rows = []
    for row in raw_table:
        if all(cell.get("is_header", False) for cell in row):
            header_rows.append(row)
        else:
            data_rows.append(row)

    if not header_rows:
        return pd.DataFrame()

    max_cols = max(get_true_max_cols(header_rows), get_true_max_cols(data_rows))
    if max_cols == 0:
        return pd.DataFrame()

    header_grid = expand_table(header_rows, max_cols)
    data_grid = expand_table(data_rows, max_cols)

    last_header_row = header_grid[-1]
    orig_header_len = len(last_header_row)

    keep_idx = []
    keep_names = []
    for i, cell in enumerate(last_header_row):
        name = "" if cell is None else str(cell).strip()
        if name:
            keep_idx.append(i)
            keep_names.append(name)

    if not keep_idx:
        return pd.DataFrame()

    clean_data = []
    for row in data_grid:
        if len(row) != orig_header_len:
            continue
        filtered = [row[i] if i < len(row) else None for i in keep_idx]
        clean_data.append(filtered)

    if not clean_data:
        return pd.DataFrame(columns=keep_names)

    df = pd.DataFrame(clean_data, columns=keep_names)
    return df


output_dir = "/mnt/hdd-storage/storage_all_users/qiaoan/totto/train"
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集，保存每个表格为 CSV
saved_files = []
for idx, sample in enumerate(train_dataset):

    try:
        df = to_dataframe(sample)

        # 构建文件路径
        file_path = os.path.join(output_dir, f"{idx}.csv")

        # 保存为 UTF-8 编码的 CSV
        df.to_csv(file_path, index=False, encoding="utf-8")

        saved_files.append(f"{idx}.csv")
        
        if idx % 1000 == 0:
            print(f"Saved {idx} tables...")

    except Exception as e:
        print(f"[Error] Failed on index {idx}: {e}")

# === 保存文件名为 context DataFrame ===
context_df = pd.DataFrame({"context": saved_files})
context_csv_path = os.path.join(output_dir, "saved_file_list.csv")
context_df.to_csv(context_csv_path, index=False, encoding="utf-8")

print(f"\n✅ Done. Saved {len(saved_files)} tables.")
print(f"📄 File list written to: {context_csv_path}")