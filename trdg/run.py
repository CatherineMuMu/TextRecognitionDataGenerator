import argparse
import errno
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random as rnd
import string
import sys
from multiprocessing import Pool

from tqdm import tqdm

from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly,
)
from trdg.utils import load_dict, load_fonts


def margins(margin):
    margins = margin.split(",")
    if len(margins) == 1:
        return [int(margins[0])] * 4
    return [int(m) for m in margins]


def parse_arguments():
    """
    Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        # 输出路径
        "--output_dir", type=str, nargs="?", help="The output directory", default="out/Test"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        # default="../trdg/texts/du/dutch-new.txt",
        # default="../trdg/texts/random_1.txt",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        # 语言
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), ar (Arabic), cn (Chinese), ja (Japanese) or hi (Hindi)",
        default="cro",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        # 生成数量
        # default=3,
        help="The number of images to be created.",
        required=True,
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False,
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        #单词数量
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=3,
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        # 以-w设置的单词数为上限，随机生成不同单词数的图片
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        # 生成图片的像素高度（水平排版），生成图片的像素宽度（竖直排版）
        help="Define the height of the produced images if horizontal, else the width",
        default= 100,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        # 运行程序使用的线程数，实测8线程下，生成一万张图片仅需 6s，设置较高的线程可以明显提速
        help="Define the number of thread to use for image generation",
        default=8,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        # 生成图片的保存格式，默认”jpg“
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        # 文字在图片中的倾斜角度
        help="Define skewing angle of the generated text. In positive degrees",
        default=1,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        # 在倾斜角度 -k 设置的情况下，比如设为 a，则生成图片文字的倾斜角度在 -a~a之间随机选择
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=True,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        # 用维基百科作为单词源，
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        # default=True,
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        # 对结果样本应用高斯模糊。应该是定义模糊半径的整数
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        # 在设定高斯模糊值 -rbl 的情况下，设bl，则生成图片的高斯模糊值在 0～bl之间随机取值
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        # 设置图片的背景，0-高斯噪声； 1-白色背景； 2-图片
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image",
        default=3,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help='Define if the data will be "handwritten" by an RNN',
        default=False,
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=2,
    )
    parser.add_argument(
        "-pre",
        "--prefix",
        type=str,
        help="prefix of pictures when saved",
        default="en",
    )
    parser.add_argument(
        "-om",
        "--output_mask",
        type=int,
        # 对于每一张生成的图片，输出同样尺寸的掩码（全黑图片），训练的时候作为一种trick
        help="Define if the generator will return masks for the text",
        default=0,
    )
    parser.add_argument(
        "-obb",
        "--output_bboxes",
        type=int,
        help="Define if the generator will return bounding boxes for the text, 1: Bounding box file, 2: Tesseract format",
        default=2,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        # 对生成图片中的文字进行扭曲，默认为0。1-正弦扭曲，2-余弦扭曲
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0,
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        # 在 -d 设定为正弦扭曲或者余弦扭曲的情况下，设定扭曲方向，0 - 竖直方向上的扭曲 1-横向扭曲
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0,
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        # 设定图片的像素宽度，在不指定的情况下，宽度为文本的宽度+10，假如设定宽度，过短会截取部分文本
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1,
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        # 在设定文本宽度参数 -wd的情况下，截取文本的方式，0 -从左侧开始截取 1- 从中心向两边截取 2-从右侧开始截取
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=0,
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        # 文本在图片中的排版，0- 横向排版，1- 竖向排版，默认横向排版
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0,
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        # 文本的颜色，通过设定的颜色，或者颜色范围，生成特定颜色的文本，颜色格式为16进制 如：#282828，（#000000，#282828）
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default="#000000,#888888",
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        # 设定图片中单词之间的像素间隔，默认为1像素
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0,
    )
    parser.add_argument(
        "-cs",
        "--character_spacing",
        type=int,
        nargs="?",
        # 设定图片中字符之间的像素间隔，默认为0像素
        help="Define the width of the spaces between characters. 2 means two pixels",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        # 	设定图片中文本，上下左右的空白间隔，以间隔的像素值表示，默认（5,5,5,5,）
        help="Define the margins around the text when rendered. In pixels",
        default=(4, 4, 4, 4),
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        # 	是否按文本裁切图片，使图片中文本上下左右的间隔均为0，默认为 False
        help="Apply a tight crop around the rendered text",
        default=False,
    )
    parser.add_argument(
        # 设定生成文本所用的字体文件（.ttf）格式
        "-ft", "--font", type=str, nargs="?", help="Define font to be used",
        # default="SourceHanSans-Normal.ttf",
    )
    parser.add_argument(
        "-fd",
        "--font_dir",
        type=str,
        nargs="?",
        # 设定生成文本所用字体的文件夹，生成的图片从文件夹中随机选择字体
        help="Define a font directory to be used",
        default="../trdg/fonts/latin_common",
    )
    # parser.add_argument(
    #     "-td",
    #     "--text_dir",
    #     type=str,
    #     nargs="?",
    #     # 所用文本所用字体的文件夹
    #     help="Define a text directory to be used",
    #     default="../trdg/texts/du",
    # )
    parser.add_argument(
        "-id",
        "--image_dir",
        type=str,
        nargs="?",
        # 在设定背景参数 -b 的值为2（即图片）的情况下，从指定的图片文件夹中读取图片作为背景。
        help="Define an image directory to use when background is set to image",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
    )
    parser.add_argument(
        "-ca",
        "--case",
        type=str,
        nargs="?",
        # 设定图片中生成的文字大小写：upper/lower
        help="Generate upper or lowercase only. arguments: upper or lower. Example: --case upper",
        # default= "upper",
    )
    parser.add_argument(
        # 设定从字典文件（路径）中选择单词生成图片
        "-dt", "--dict", type=str, nargs="?", help="Define the dictionary to be used"
    )
    parser.add_argument(
        "-ws",
        "--word_split",
        action="store_true",
        # 设定是设定根据单词还是字符分隔文字，True-根据单词 Talse-根据字符
        help="Split on words instead of on characters (preserves ligatures, no character spacing)",
        default=False,
    )
    parser.add_argument(
        "-stw",
        "--stroke_width",
        type=int,
        nargs="?",
        help="Define the width of the strokes",
        default=0,
    )
    parser.add_argument(
        "-stf",
        "--stroke_fill",
        type=str,
        nargs="?",
        help="Define the color of the contour of the strokes, if stroke_width is bigger than 0",
        default="#282828",
    )
    parser.add_argument(
        "-im",
        "--image_mode",
        type=str,
        nargs="?",
        help="Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.",
        default="RGB",
    )
    return parser.parse_args()


def main():
    """
    Description: Main function
    """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    if args.dict:
        lang_dict = []
        if os.path.isfile(args.dict):
            with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
                lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
        else:
            sys.exit("Cannot open dict")
    else:
        lang_dict = load_dict(
            os.path.join(os.path.dirname(__file__), "dicts", args.language + ".txt")
        )

    # Create font (path) list
    if args.font_dir:
        fonts = [
            os.path.join(args.font_dir, p)
            for p in os.listdir(args.font_dir)
            if os.path.splitext(p)[1] == ".ttf"
        ]
    elif args.font:
        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")
    else:
        text = load_fonts(args.language)

    # # Create text (path) list
    # if args.text_dir:
    #     text = [
    #         os.path.join(args.text_dir, p)
    #         for p in os.listdir(args.text_dir)
    #         if os.path.splitext(p)[1] == ".txt"
    #     ]
    # elif args.text:
    #     if os.path.isfile(args.text):
    #         text = [args.text]
    #     else:
    #         sys.exit("Cannot open text")
    # else:
    #     fonts = load_fonts(args.language)

    # Creating synthetic sentences (or word)
    strings = []

    if args.use_wikipedia:
        strings = create_strings_from_wikipedia(args.length, args.count, args.language)
    elif args.input_file != "":
        strings = create_strings_from_file(args.input_file, args.count)
    elif args.random_sequences:
        strings = create_strings_randomly(
            args.length,
            args.random,
            args.count,
            args.include_letters,
            args.include_numbers,
            args.include_symbols,
            args.language,
        )
        # Set a name format compatible with special characters automatically if they are used
        if args.include_symbols or True not in (
            args.include_letters,
            args.include_numbers,
            args.include_symbols,
        ):
            args.name_format = 2
    else:
        strings = create_strings_from_dict(
            args.length, args.random, args.count, lang_dict
        )

    if args.language == "ar":
        from arabic_reshaper import ArabicReshaper
        from bidi.algorithm import get_display

        arabic_reshaper = ArabicReshaper()
        strings = [
            " ".join(
                [get_display(arabic_reshaper.reshape(w)) for w in s.split(" ")[::-1]]
            )
            for s in strings
        ]
    if args.case == "upper":
        strings = [x.upper() for x in strings]
    if args.case == "lower":
        strings = [x.lower() for x in strings]

    string_count = len(strings)

    p = Pool(args.thread_count)
    for _ in tqdm(
        p.imap_unordered(
            FakeTextDataGenerator.generate_from_tuple,
            zip(
                [i for i in range(0, string_count)],
                strings,
                [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                [args.output_dir] * string_count,
                [args.format] * string_count,
                [args.extension] * string_count,
                [args.skew_angle] * string_count,
                [args.random_skew] * string_count,
                [args.blur] * string_count,
                [args.random_blur] * string_count,
                [args.background] * string_count,
                [args.distorsion] * string_count,
                [args.distorsion_orientation] * string_count,
                [args.handwritten] * string_count,
                [args.name_format] * string_count,
                [args.prefix] * string_count,
                [args.width] * string_count,
                [args.alignment] * string_count,
                [args.text_color] * string_count,
                [args.orientation] * string_count,
                [args.space_width] * string_count,
                [args.character_spacing] * string_count,
                [args.margins] * string_count,
                [args.fit] * string_count,
                [args.output_mask] * string_count,
                [args.word_split] * string_count,
                [args.image_dir] * string_count,
                [args.stroke_width] * string_count,
                [args.stroke_fill] * string_count,
                [args.image_mode] * string_count,
                [args.output_bboxes] * string_count,
            ),
        ),
        total=args.count,
    ):
        pass
    p.terminate()

    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(
            os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf8"
        ) as f:
            for i in range(string_count):
                file_name = args.prefix + "_" + str(i) + "." + args.extension
                label = strings[i]
                if args.space_width == 0:
                    label = label.replace(" ", "")
                f.write("{} {}\n".format(file_name, label))


if __name__ == "__main__":
    main()

# --output_dir "out/Cro_random" # 输出路径
# --input_file  # 不为空的话，指定txt输入路径
# --language "cro" # 语言
# --count 2000  # 生成数量
# --length 3   #单词数量
# --random False # 以-length设置的单词数为上限，随机生成不同单词数的图片
# --format 100 # 生成图片的像素高度（水平排版），生成图片的像素宽度（竖直排版）
# --thread_count 8 #使用的线程数
# --skew_angle 10 #倾斜角度
# --random_skew True # 倾斜角度随机选择
# --use_wikipedia False
# --blur 0 # 高斯模糊半径
# --random_blur False # 高斯模糊随机
# --handwritten False # 手写字体
# --name_format 2 #生成图片的命名格式
# --distorsion 0 #文字扭曲类型
# --distorsion_orientation 0 #文字扭曲方向
# --width -1 # 设定图片的像素宽度
# --alignment 1
# --orientation 0 # 文本在图片中的排版，0- 横向排版，1- 竖向排版
# --text_color "#000000,#888888" # 颜色范围
# --space_width 1.0 # 图片中单词之间的像素间隔
# --character_spacing 0
# --font_dir "../trdg/fonts/latin_common"
# --stroke_fill "#282828" # 笔画轮廓颜色
