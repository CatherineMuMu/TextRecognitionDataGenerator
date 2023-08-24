import random
import csv

# 读取CSV文件
with open('./texts_new/all_char.csv', 'r',encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    data = list(reader)

# 根据CSV文件内容生成字符串
characters = []
for row in data:
    characters.extend([row[0]] * int(row[1]))


with open('./texts_new/rare_new.txt', 'w',encoding='utf-8') as file:
    while characters:
        sentence_length = random.randint(6, 16)
        if len(characters) >= sentence_length:
            sentence = random.sample(characters, sentence_length)
            file.write(''.join(sentence) + '\n')
            for char in sentence:
                characters.remove(char)
