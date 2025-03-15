import os

# 设置文件夹路径
folder_path = "/home/data/wyq/ABAW_VA/ABAW2-test/ABAW2-EXP/data_all/cropped_aligned"

# 统计子文件夹数量
num_subfolders = len([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])

print(f"文件夹 {folder_path} 中有 {num_subfolders} 个子文件夹。")

folder_path = "/home/data/wyq/ABAW_VA/ABAW2-test/ABAW2-EXP/data_all/raw_video"  # 文件夹路径

# 遍历文件夹，并计数
file_count = sum([len(files) for root, dirs, files in os.walk(folder_path)])

print(f"文件夹 {folder_path} 中共有 {file_count} 个文件。")
