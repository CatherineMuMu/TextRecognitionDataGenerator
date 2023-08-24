import shutil
import os
import random
import json
import tqdm
from trdg.generators import GeneratorFromDict, GeneratorFromRandom, GeneratorFromStrings, GeneratorFromWikipedia
# from rich.progress import Progress
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# space_width, word_split
def gen_genter(language, mode, count, fonts, texts):
    if mode == 's' and not texts:
        raise ValueError("mode is 's', but texts is None")
    if mode == 's' and texts:
        genter = GeneratorFromStrings(texts,
                                      count=count,
                                      # length=2, # Random为2
                                      size=80,
                                      language=language,
                                      fonts=[fonts],
                                      image_dir='./images',
                                      is_handwritten=False,
                                      random_blur=False,  # True
                                      blur=0,  # 2
                                      stroke_width=0,  # 笔画轮廓，默认0
                                      stroke_fill="#282828",  # 笔画轮廓颜色，默认"#282828"
                                      text_color="#000000,#282828",  # 文本颜色，默认"#282828" #000000,#282828
                                      random_skew=True,  # 随机倾斜
                                      skewing_angle=0,  # 倾斜角度  2
                                      alignment=0,  # 对其方式0左1中2右
                                      width=-1,  # 图像宽度，默认-1为文本宽度+10
                                      background_type=1,  # 背景类型，0高斯1纯白2准晶体3图片
                                      distorsion_type=0,  # 扭曲类型0无（默认）1正弦波2余弦波3随机(毛刺) 改成列表[0,2,3]
                                      distorsion_orientation=2,  # 扭曲方向0上下1左右2都有
                                      space_width=1.0,  # 单词间空格宽度，2.0为正常宽度2倍 1.0
                                      character_spacing=0,  # 字符之间空格宽度，2为2个像素
                                      margins=(4, 4, 4, 4),  # 文本周边的边距，单位像素
                                      fit=True,  # 生成文本周围使用紧密裁剪
                                      word_split=True,  # 拆分单词而不是字符（保留连字，无字符间距）
                                      )
    elif mode == 'd':
        genter = GeneratorFromDict(  # ['Когда все закончится, они будут жить','Ко'],
            count=count,
            length=3,  # Random为2
            size=80,
            language=language,
            fonts=[fonts],
            image_dir='./images',
            is_handwritten=False,
            random_blur=True,  # True
            blur=2,  # 2
            stroke_width=0,  # 笔画轮廓，默认0
            stroke_fill="#282828",  # 笔画轮廓颜色，默认"#282828"
            text_color="#000000,#282828",  # 文本颜色，默认"#282828" #000000,#282828
            random_skew=True,  # 随机倾斜  True
            skewing_angle=2,  # 倾斜角度
            alignment=0,  # 对其方式0左1中2右
            width=-1,  # 图像宽度，默认-1为文本宽度+10
            background_type=3,  # 背景类型，0高斯1纯白2准晶体3图片
            distorsion_type=[0, 2, 3],  # 扭曲类型0无（默认）1正弦波2余弦波3随机(毛刺)[0,2,3]
            distorsion_orientation=2,  # 扭曲方向0上下1左右2都有
            space_width=1.0,  # 单词间空格宽度，2.0为正常宽度2倍
            character_spacing=0,  # 字符之间空格宽度，2为2个像素
            margins=(4, 4, 4, 4),  # 文本周边的边距，单位像素
            fit=True,  # 生成文本周围使用紧密裁剪
            word_split=True,  # 拆分单词而不是字符（保留连字，无字符间距）
        )
    elif mode == 'r':
        genter = GeneratorFromRandom(  # ['Когда все закончится, они будут жить','Ко'],
            count=count,
            length=1,  # Random为2
            size=80,
            language=language,
            fonts=[fonts],
            image_dir='./images',
            is_handwritten=False,
            random_blur=False,  # True
            blur=0,  # 2
            stroke_width=0,  # 笔画轮廓，默认0
            stroke_fill="#282828",  # 笔画轮廓颜色，默认"#282828"
            text_color="#000000,#282828",  # 文本颜色，默认"#282828" #000000,#282828
            random_skew=True,  # 随机倾斜
            skewing_angle=2,  # 倾斜角度
            alignment=0,  # 对其方式0左1中2右
            width=-1,  # 图像宽度，默认-1为文本宽度+10
            background_type=3,  # 背景类型，0高斯1纯白2准晶体3图片
            distorsion_type=0,  # 扭曲类型0无（默认）1正弦波2余弦波3随机(毛刺)
            distorsion_orientation=0,  # 扭曲方向0上下1左右2都有
            space_width=1.0,  # 单词间空格宽度，2.0为正常宽度2倍
            character_spacing=0,  # 字符之间空格宽度，2为2个像素
            margins=(4, 4, 4, 4),  # 文本周边的边距，单位像素
            fit=False,  # 生成文本周围使用紧密裁剪
            word_split=False,  # 拆分单词而不是字符（保留连字，无字符间距）
        )
    else:
        raise ValueError(f"have no {mode}")
    return genter


def gen_text(gen_dir, sub_dir, language, mode, all_fonts, all_count, texts_path, ratio=(0.9, 0.1)):
    for name in ['train', 'test']:  # _no_art
        if os.path.exists(f"{gen_dir}/{sub_dir}/{name}"):
            shutil.rmtree(f"{gen_dir}/{sub_dir}/{name}")
        os.makedirs(f"{gen_dir}/{sub_dir}/{name}")

    if os.path.exists(f"{gen_dir}/{sub_dir}.log"):
        os.remove(f"{gen_dir}/{sub_dir}.log")

    if len(all_count) != 1 and len(all_fonts) != len(all_count):
        raise ValueError("all_fonts length must be == all_count length or all_count == 1")
    if len(all_count) == 1:
        all_count = all_count * len(all_fonts)

    if texts_path:
        with open(texts_path, 'r') as f:
            jf_dic = json.load(f)
        texts = [label for _, label in jf_dic.items()]
        all_count = [count * len(texts) for count in all_count]  # 限制上限count*4000
    else:
        texts = []  # ['หม่อมแก้ว'] ['รัฐบริติชโคลัมเบีย เข็มขาว']  # 可添加

    # with open(f"{gen_dir}/train_labels.json", 'w') as tr, open(f"{gen_dir}/test_labels.json", 'w') as te:
    with Progress() as progress:
        task1 = progress.add_task("[red]overall...", total=sum(all_count))
        train_dict, test_dict = {}, {}
        for fonts, count in zip(all_fonts, all_count):
            ttf_name = os.path.split(os.path.splitext(fonts)[0])[-1]
            task2 = progress.add_task(f"[green]{ttf_name}...", total=count)

            if os.path.exists(f"{gen_dir}/{ttf_name}"):
                shutil.rmtree(f"{gen_dir}/{ttf_name}")
            os.mkdir(f"{gen_dir}/{ttf_name}")

            # texts = random.sample(texts, count) if texts else [] # 随机抽取样本
            genter = gen_genter(language, mode, count, fonts, texts)

            pre_cnt, pre_cnt_test = 0, 0
            with open(f"{gen_dir}/{ttf_name}/{ttf_name}_labels.txt", 'w') as f:
                for img, lbl in genter:
                    if img:
                        progress.advance(task1, advance=1)
                        progress.advance(task2, advance=1)
                        pre_cnt += 1
                        # 保存在每个字体名字下面
                        img.save(f"{gen_dir}/{ttf_name}/{pre_cnt}_{sub_dir}_{ttf_name}.jpg")
                        f.write(f"{pre_cnt}_{sub_dir}_{ttf_name}.jpg\t{lbl}\n")
                        # train, test
                        labelme_json = {
                            "label": lbl,
                            "size": [img.size[1], img.size[0]],  # 1高0宽
                            "scene": "general"
                        }
                        rand = random.random()
                        if 0 < rand <= ratio[0]:
                            # print(f'{pre_cnt} train')
                            img.save(f"{gen_dir}/{sub_dir}/train/{pre_cnt}_{sub_dir}_{ttf_name}.jpg")  # _no_art
                            train_dict[f"{pre_cnt}_{sub_dir}_{ttf_name}.jpg"] = labelme_json
                            # tr.write(f"{cnt}_{ttf_name}.jpg\t{lbl}\n")  # txt
                        elif ratio[0] < rand <= ratio[0] + ratio[1]:
                            pre_cnt_test += 1
                            # print(f'{pre_cnt} test')
                            img.save(f"{gen_dir}/{sub_dir}/test/{pre_cnt}_{sub_dir}_{ttf_name}.jpg")
                            test_dict[f"{pre_cnt}_{sub_dir}_{ttf_name}.jpg"] = labelme_json
                            # te.write(f"{cnt}_{ttf_name}.jpg\t{lbl}\n")  # txt
                if pre_cnt:
                    with open(f"{gen_dir}/{sub_dir}/{sub_dir}.log", 'a') as lo:
                        message = f"{ttf_name}: train {((pre_cnt - pre_cnt_test) / pre_cnt) * 100:.3f}%, test {(pre_cnt_test / pre_cnt) * 100:.3f}%; train {pre_cnt - pre_cnt_test} test {pre_cnt_test}"
                        print(message)
                        lo.write(f"{message}\n")
        # _no_art
        with open(f"{gen_dir}/{sub_dir}/{sub_dir}_train.json", 'w') as tr, \
                open(f"{gen_dir}/{sub_dir}/{sub_dir}_test.json", 'w') as te, \
                open(f"{gen_dir}/{sub_dir}/epmty.json", 'w') as em:
            train_cnt, test_cnt = len(train_dict), len(test_dict)
            if (train_cnt + test_cnt):
                with open(f"{gen_dir}/{sub_dir}/{sub_dir}.log", 'a') as lo:
                    message = f"all: train {(train_cnt / (train_cnt + test_cnt)) * 100:.3f}%, test {(test_cnt / (train_cnt + test_cnt)) * 100:.3f}%; train {train_cnt} test {test_cnt}"
                    print(message)
                    lo.write(f"{message}\n")
            json.dump(train_dict, tr, ensure_ascii=False)
            json.dump(test_dict, te, ensure_ascii=False)
            json.dump({}, em, ensure_ascii=False)


if __name__ == '__main__':
    # gen_dir = '1.gen_ru'
    # sub_dir = 'ru_char_random'  # 不用了，有英文I和i
    # sub_dir = 'rube_char_random'  # 使用高斯背景
    # sub_dir = 'rube_charimg_random' # 使用图片
    # sub_dir = 'rube_similar_string' # 使用默认高斯
    # sub_dir = 'rube_badcase_string' # 使用背景+新表+随机抽取+增强+多字体

    gen_dir = '1.th_9w(para)'
    sub_dir = 'th_9w(para)'

    language = 'th'
    mode = 'd'
    all_count = [3000]  # 3000  1500  800  500  350

    # 俄语
    # all_fonts = [
    #  './fonts/ru/arial.ttf',                        # 印刷体    Arial
    #  './fonts/ru/ariblk.ttf',                       # 印刷体    Arial Black   粗
    #  './fonts/ru/arialbd.ttf',                      # 印刷体  Arial Bold    粗
    #  './fonts/ru/arialbi.ttf',                      # 手写体  Arial Bold Italic 粗
    #  './fonts/ru/ariali.ttf',                       # 手写体    Arial Italic
    #  './fonts/ru/ARIALN.TTF',                       # 印刷体    Arial Narrow    缺失₽
    #  './fonts/ru/ARIALNB.TTF',                      # 印刷体    Arial Narrow Bold   粗,缺失₽
    #  './fonts/ru/ARIALNBI.TTF',                     # 手写体    Arial Narrow Bold Italic    粗,缺失₽
    #  './fonts/ru/ARIALNI.TTF',                      # 手写体    Arial Narrow Italic    缺失₽
    #  './fonts/ru/ANTQUAB.TTF',                      # 印刷体    Book Antiqua     粗
    #  './fonts/ru/ANTQUABI.TTF',                     # 手写体    Book Antiqua     稍粗
    #  './fonts/ru/ANTQUAI.TTF',                      # 手写体    Book Antiqua     较细
    #  './fonts/ru/BKANT.TTF',                        # 印刷体    Book Antiqua     符号两边距离大些
    #  './fonts/ru/calibri.ttf',                      # 印刷体    Calibri    稍粗
    #  './fonts/ru/calibrii.ttf',                     # 手写体    Calibri
    #  './fonts/ru/Candara.ttf',                      # 印刷体    Candara     稍粗
    #  './fonts/ru/Candarali.ttf',                     # 手写体    Candara
    #  './fonts/ru/comici.ttf',                       # 印刷体    Comic Sans Ms
    #  './fonts/ru/impact.ttf',                       # 印刷体    Impact
    #  './fonts/ru/pala.ttf',                      # 印刷体    Palatino Linotype
    #  './fonts/ru/palai.ttf',                      # 手写体    Palatino Linotype 稍粗
    #  './fonts/ru/segoepr.ttf',                      # 手写体    Segoe Print
    #  './fonts/ru/segoesc.ttf',                      # 手写体    Segoe Script  字符i错误
    #  './fonts/ru/Gabriola.ttf',                     # 手写体    Gabriola
    # # './fonts/ru/AGCooper.ttf',                     # 印刷体粗    有多个字符缺失
    #  './fonts/ru/CormorantGaramond-Regular.ttf',    # 印刷体
    #  './fonts/ru/decor.ttf',                        # 带尾边手写体
    #  './fonts/ru/Elzevir.ttf',                      # 带连笔手写体
    # ]

    gen_text(gen_dir, sub_dir, language, mode, all_fonts, all_count, None)