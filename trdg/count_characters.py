# filename = './texts_new/az.txt'
# encoding = 'utf-8'
#
# try:
#     with open(filename, 'r', encoding=encoding) as file:
#         text = file.read()
#
#         # 统计字符数量
#         char_count = {}
#         for char in text:
#             char_count[char] = char_count.get(char, 0) + 1
#
#         # 统计字符种类数量
#         num_unique_chars = len(char_count)
#
#         # 打印结果
#         print(f"总共有 {num_unique_chars} 种字符")
#         for char, count in char_count.items():
#             print(f"字符 '{char}' 出现了 {count} 次")
#
# except FileNotFoundError:
#     print(f"找不到文件 '{filename}'")
# except UnicodeDecodeError:
#     print(f"无法使用编码 '{encoding}' 读取文件 '{filename}'")


# ------------------------------------------------------------------------------------------------------

import os
import csv

folder_path = './data_latest/vi/'
output_file = './data_latest/vi/test/chact.csv'
encoding = 'utf-8'

# 用于统计字符数量的字典
char_count = {}

# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                # 统计字符数量
                for char in text:
                    char_count[char] = char_count.get(char, 0) + 1
        except UnicodeDecodeError:
            print(f"无法使用编码 '{encoding}' 读取文件 '{file_path}'")

# 统计字符种类数量
num_unique_chars = len(char_count)

# 输出到CSV文件
with open(output_file, 'w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Character', 'Count'])
    for char, count in char_count.items():
        writer.writerow([char, count])

# 打印结果
print(f"文件夹 '{folder_path}' 中共有 {num_unique_chars} 种字符")
for char, count in char_count.items():
    print(f"字符 '{char}' 出现了 {count} 次")
print(f"结果已保存到文件 '{output_file}'")
