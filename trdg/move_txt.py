##把txt文件前45000行移到train，后5000行移到test
import os

original_folder = "./texts_new"
train_folder = "./count_characters/train"
test_folder = "./count_characters/test"
lines_train = 44999
lines_test = 50000
encoding = "utf-8"  # 修改为原文件的编码方式

# 创建train和test文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 遍历原文件夹中的所有txt文件
for file_name in os.listdir(original_folder):
    if file_name.endswith('.txt'):
        original_file_path = os.path.join(original_folder, file_name)
        train_file_path = os.path.join(train_folder, file_name)
        test_file_path = os.path.join(test_folder, file_name)

        try:
            # 读取原文件的内容并写入train和test文件
            with open(original_file_path, 'r', encoding=encoding) as original_file, \
                    open(train_file_path, 'w', encoding=encoding) as train_file, \
                    open(test_file_path, 'w', encoding=encoding) as test_file:

                for i, line in enumerate(original_file):
                    if i < lines_train:
                        train_file.write(line)
                    elif i < lines_test:
                        test_file.write(line)

        except FileNotFoundError:
            print(f"找不到文件 '{original_file_path}'")

