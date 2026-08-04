import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from PIL import Image
import argparse
#import sys
import os
import subprocess
from subprocess import PIPE
import re
import json

import vlm_ocr
import stage1
import stage2

parser = argparse.ArgumentParser(description='Detect Hanzi-components from image file and generate IDS if possible.')

parser.add_argument('image_files', nargs='*', help='Image file name to process')
parser.add_argument('--model', help='MLX-VLM model path', default='froggeric/Qwen3.6-27B-Uncensored-Heretic-v2-MLX-8bit') 
parser.add_argument('--git', help='Run git add, commit and push') 

args = parser.parse_args()


# Load the model
model_path = args.model
model_separator_pos = model_path.find('/')
model_name = model_path[model_separator_pos + 1:]

model, processor = load(model_path)
config = load_config(model_path)


position_prompt_en_p = 'position (left/right/above/below/surround-from-upper-left/surround-from-lower-left/full-surround/surround-from-above/surround-from-left/surround-from-upper-right/surround-from-below/surround-from-right/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/sandwiched-from-left-and-right/sandwiched-from-above-and-below/inserted-from-below/inserted-from-left/inserted-from-right)'

position_prompt_ja_p = '相対位置(left(偏)/right(旁)/above(冠)/below(脚)/surround-from-upper-left(垂)/surround-from-lower-left(繞)/full-surround(箱構)/surround-from-above(上構)/surround-from-left/surround-from-upper-right/surround-from-below/surround-from-lower-right/upper-left(左上)/upper-right(右上)/lower-left(左下)/lower-right(右下)/enclosed(構の中)/surround-from-lower-right/sandwiched-from-left-and-right(左右の間)/sandwiched-from-above-and-below(上下の間)/inserted-from-below/inserted-from-left/inserted-from-right)'

position_prompt_ja_p_s = '相対位置(left(偏)/right(旁)/above(冠)/below(脚)/upper-left(左上)/upper-right(右上)/lower-left(左下)/lower-right(右下)/surround(構、垂、繞など)/enclosed(構の中)/sandwiched(挟まれた)/inserted-from-below/inserted-from-left(左から差し込む)/inserted-from-right(右から差し込む))'

prompt_E6p = f'''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_C6p = f'''<image> Locate every component of the Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_J6p = f'''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_cJ = '''<image> Locate every component of the classical Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_JcE = '''<image> Locate every component of the Kanji (or classical Chinese character).
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_ja6p = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_ja_p}
'''

prompt_ja6p_s = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_ja_p_s}
'''

#prompt = prompt_E6p
#prompt = prompt_cE
#prompt = prompt_C
#prompt = prompt_ja
#prompt = prompt_ja6p
#prompt = prompt_zh_TW
#prompt = prompt_E5
#prompt = prompt_E5s
#retry_prompt = prompt_ja6p
#retry_prompt = prompt_ja
#prompt = prompt_C6p

# prompt = prompt_E6p
# retry_prompt = prompt_ja
# retry_prompt2 = prompt_C

# prompt = prompt_J6p
# retry_prompt = prompt_ja
# retry_prompt2 = prompt_zh_TW
# retry_prompt3 = prompt_E5

prompt = prompt_ja6p_s

# prompt = prompt_E6p
# retry_prompt = prompt_ja
# retry_prompt2 = prompt_E5
# retry_prompt3 = prompt_zh_TW

# prompt = prompt_J6p
# retry_prompt = prompt_ja
# retry_prompt2 = prompt_E6p


simpler_prompt = f'''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (left/right/above/below/surround/upper-left/lower-left/lower-right/middle)
'''

# character_prompt = '''<image> Run Kanji (or traditional Hanzi) OCR and output the result.
# '''
character_prompt = '''<image> 漢字用 OCR を実行し、結果の文字を返してください。
'''

retry_prompt_L2R = stage2.L2R_prompt_ja
retry_prompt_A2B = stage2.A2B_prompt_ja
retry_prompt_default = stage2.prompt_ja

def manage_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH):
    basename = os.path.splitext(os.path.basename(image_file))[0]
    ids_file_name  = f'{OUTPUT_PATH}/{basename}_ids.txt'
    full_file_name = f'{OUTPUT_PATH}/{basename}_full.txt'
    if os.path.isfile(ids_file_name):
        print( f'{ids_file_name} already exists.')
        ids_destfile = open (ids_file_name, 'r', encoding = 'utf-8')
        ids = ids_destfile.read()
        ids_destfile.close()
        return ids
    elif os.path.isfile(full_file_name):
        print( f'{full_file_name} already exists.')
        full_destfile = open (full_file_name, 'r', encoding = 'utf-8')
        full = full_destfile.read()
        full_destfile.close()
        return full

    im = Image.open(image_file)
    image_width, image_height = im.size

    images = [ image_file ]
    print (f'character prompt = "{character_prompt}".')

    char_response = vlm_ocr.run_VLM (images, character_prompt, model, processor, config)
    print (f'character = "{char_response}".')
    char_res = re.match ('^(.)[.,。、．]$', char_response)
    if char_res:
        char_response = char_res.group(1)
        print (f'character = "{char_response}".')
    else:
        char_res = re.match ('^「(.)」$', char_response)
        if char_res:
            char_response = char_res.group(1)
            print (f'character = "{char_response}".')

    X1, Y1, X2, Y2, Component_Text, Component_Position = stage1.run_OCR_for_glyph_image (image_file_name,
                                                                                         prompt,
                                                                                         TSV_OUTPUT_PATH,
                                                                                         OUTPUT_PATH,
                                                                                         model, processor, config)
    retry_flag = False
    retry_prompt = retry_prompt_default
    if ( ( len(char_response) == 1 ) and
         ( any ( comp == char_response for comp in Component_Text ) ) and
         ( ( len(Component_Text) < 2 ) or
           ( Component_Text[0] != '囗' ) ) ):
        with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
            print(char_response, file=full_destfile)
            return char_response

    if len(Component_Text) == 1:
        if (os.path.isfile(full_file_name)):
            os.remove(full_file_name)

        if ( ( Component_Position[0] == 'full-surround' ) or
             ( Component_Position[0] == 'full' ) ):
            with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                print(Component_Text[0], file=full_destfile)
            return Component_Text[0]
        elif len(char_response) == 1:
            with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                print(char_response, file=full_destfile)
                return char_response

    elif len(Component_Text) == 2:
        if ( ( X1[0] == X1[1] ) and
             ( Y1[0] == Y1[1] ) and
             ( X2[0] == X2[1] ) and
             ( Y2[0] == Y2[1] ) ):
            retry_flag = True
            if ( Component_Position[0] == 'above' ):
                retry_prompt = retry_prompt_A2B
            elif ( Component_Position[0] == 'left' ):
                retry_prompt = retry_prompt_L2R
            else:
                retry_prompt = retry_prompt_default

        elif ( ( max ( X2[0] - X1[0], X2[1] - X1[1] ) < image_width * 0.7 ) and
               ( max ( Y2[0] - Y1[0], Y2[1] - Y1[1] ) < image_height * 0.7 ) ):
            retry_flag = True
            if ( Component_Position[0] == 'above' ):
                retry_prompt = retry_prompt_A2B
            elif ( Component_Position[0] == 'left' ):
                retry_prompt = retry_prompt_L2R
            else:
                retry_prompt = retry_prompt_default

        else:
            match Component_Position[0]:
                case 'left':
                    if Component_Position[1] == 'surround-from-above':
                        retry_flag = True
                        retry_prompt = retry_prompt_L2R
                    elif ( Component_Position[1] == 'right' ):
                        if ( ( Component_Text[0] == '刂' ) or
                             ( Component_Text[0] == '力' ) or
                             ( Component_Text[0] == '門' ) or
                             ( Component_Text[1] == '⻖' ) or
                             ( Component_Text[1] == '僉' ) or
                             ( Component_Text[1] == '攸' ) or
                             ( Component_Text[1] == '冏' ) or
                             ( Component_Text[1] == '乚' ) or
                             ( Component_Text[1] == '隹' ) or
                             ( Component_Text[1] == '夊' ) or
                             ( Component_Text[1] == '夂' ) or
                             ( Component_Text[1] == '堇' ) or
                             ( Component_Text[1] == '襄' ) or
                             ( Component_Text[1] == '侯' ) or
                             ( Component_Text[1] == '頁' ) ):
                            retry_flag = True
                            retry_prompt = retry_prompt_L2R
                    elif ( Component_Position[1] == 'below' ):
                        retry_flag = True
                        retry_prompt = retry_prompt_default
                    elif ( ( Component_Text[1] == '夊' ) or
                           ( Component_Text[1] == '冂' ) or
                           ( Component_Text[1] == '攸') or
                           ( Component_Text[0] == '饣') or
                           ( Component_Text[0] == '𠂉') ):
                        retry_flag = True
                        retry_prompt = retry_prompt_L2R
                    elif ( ( Component_Text[0] == '亻' ) and
                           ( ( Component_Text[1] == '優' ) or
                             ( Component_Text[1] == '偶' ) ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_L2R

                case 'above':
                    if ( ( Component_Text[0] == '⺌' ) or
                         ( Component_Text[0] == '亠' ) or
                         ( Component_Text[0] == '立' ) or
                         ( Component_Text[0] == '宀' ) or
                         ( Component_Text[0] == '殳' ) or
                         ( Component_Text[0] == '頁' ) or
                         ( Component_Text[0] == '學' ) or
                         ( Component_Text[1] == '八' ) or
                         ( Component_Text[1] == '十' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_A2B
                    elif ( Component_Position[1] == 'below' ):
                        if ( ( Component_Text[1] == '由' ) or
                             ( Component_Text[1] == '木' ) or
                             ( Component_Text[0] == '羊' ) ):
                            retry_flag = True
                            retry_prompt = retry_prompt_A2B
                        elif ( Component_Text[1] == '弄' ):
                            retry_flag = True
                            retry_prompt = retry_prompt_A2B
                    # elif ( Component_Position[1] == 'enclosed' ):
                    #     retry_flag = True
                    #     retry_prompt = retry_prompt_A2B

                case 'upper-left':
                    if ( ( Component_Text[0] == '亠' ) or
                         ( Component_Text[0] == '宀' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_A2B
                    elif ( ( Component_Position[1] == 'lower-right' ) and
                           ( Component_Text[1] == '支' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_default
                    elif ( ( Component_Position[1] == 'upper-right' ) and
                           ( Component_Text[0] == '立' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_A2B

                case 'lower-left':
                    if Component_Text[0] == '口':
                        retry_flag = True
                        retry_prompt = retry_prompt_default
                    elif ( ( Component_Text[0] == '辶' ) and
                           ( Component_Text[1] == '止' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_default

                case 'full-surround':
                    if Component_Text[0] == '儿':
                        retry_flag = True
                        retry_prompt = retry_prompt_default

                case 'surround-from-lower-left':
                    if ( ( Component_Text[0] == '勹' ) or
                         ( Component_Text[0] == '門' ) or
                         ( Component_Text[0] == '冂' ) ):
                        retry_flag = True
                        retry_prompt = retry_prompt_default

                case 'surround':
                    if ( Component_Text[0] == '𠁣' ):
                        retry_flag = True
                        retry_prompt = retry_prompt_default

    elif len(Component_Text) == 3:
        if ( ( Component_Position[0] == 'above' ) and
             ( Component_Position[1] == 'lower-left' ) and
             ( Component_Position[2] == 'lower-right' ) and
             ( ( ( Component_Text[1] == '丷' ) and
                 ( Component_Text[2] == '丷' ) )
               or
               ( ( Component_Text[1] == '十' ) and
                 ( Component_Text[2] == '十' ) ) ) ):
            retry_flag = True
            retry_prompt = retry_prompt_A2B
        elif ( ( Component_Position[0] == 'left' ) and
               ( Component_Position[1] == 'upper-right' ) and
               ( Component_Position[2] == 'lower-right' ) and
               ( ( ( Component_Text[1] == '田' ) and
                   ( Component_Text[2] == '日' ) )
                 or
                 ( ( Component_Text[1] == '又' ) and
                   ( Component_Text[2] == '又' ) ) ) ):
            retry_flag = True
            retry_prompt = retry_prompt_L2R
        elif ( ( Component_Position[0] == 'below' ) and
               ( Component_Position[1] == 'left' ) and
               ( Component_Position[2] == 'surround' ) ):
            retry_flag = True
            retry_prompt = retry_prompt_A2B

    elif len(Component_Text) > 7:
        retry_flag = True
        if ( Component_Position[0] == 'above' ):
            retry_prompt = retry_prompt_A2B
        elif ( Component_Position[0] == 'left' ):
            retry_prompt = retry_prompt_L2R
        else:
            retry_prompt = retry_prompt_default

    if retry_flag:
        print ('Retry')
        X1, Y1, X2, Y2, Component_Text, Component_Position, Mother = stage2.run_OCR_for_glyph_image (image_file_name,
                                                                                                     stage2.prompt,
                                                                                                     TSV_OUTPUT_PATH,
                                                                                                     OUTPUT_PATH,
                                                                                                     model, processor,
                                                                                                     config)
        ids = stage2.detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position, Mother)
    else:
        ids = stage1.detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
        print ('Retry')
        X1, Y1, X2, Y2, Component_Text, Component_Position, Mother = stage2.run_OCR_for_glyph_image (image_file_name,
                                                                                                     retry_prompt,
                                                                                                     TSV_OUTPUT_PATH,
                                                                                                     OUTPUT_PATH,
                                                                                                     model, processor,
                                                                                                     config)
        if ( ( len(Component_Text) > 2 ) and
             ( Component_Position[1] == 'left') and
             ( Component_Text[1] == '皿' ) ):
            ids = None
        elif ( ( len(Component_Text) > 2 ) and
               ( Component_Position[2] == 'below') and
               ( Component_Text[2] == '⺘' ) ):
            ids = None
        else:
            ids = stage2.detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position, Mother)

    if ids:
        with open(ids_file_name, 'w', encoding = 'utf-8') as ids_destfile:
            print(ids, file=ids_destfile)
        return ids
    else:
        if ( len(char_response) == 1 ):
            with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                print(char_response, file=full_destfile)
            return char_response

proc = subprocess.run("ipfs add -- | cut -d' ' -f2", shell=True, input=prompt, stdout=PIPE, stderr=PIPE, text=True)
stage1_CID = proc.stdout.rstrip('\r\n')
print (f'stage1 CID = {stage1_CID}')

proc = subprocess.run("ipfs add -- | cut -d' ' -f2", shell=True, input=stage2.prompt, stdout=PIPE, stderr=PIPE, text=True)
stage2_CID = proc.stdout.rstrip('\r\n')
print (f'stage2 CID = {stage2_CID}')

OUTPUT_PATH = f'hybrid_{model_name}/{stage1_CID}+{stage2_CID}/tsv_pct100'
os.makedirs(OUTPUT_PATH, exist_ok=True)
TSV_OUTPUT_PATH = f'hybrid_{model_name}/{stage1_CID}+{stage2_CID}/tsv_pct100'
os.makedirs(TSV_OUTPUT_PATH, exist_ok=True)

print (OUTPUT_PATH, TSV_OUTPUT_PATH)

for image_file_name in args.image_files:
    ids = manage_OCR_for_glyph_image (image_file_name, prompt,
                                      TSV_OUTPUT_PATH, OUTPUT_PATH)
    if args.git:
        subprocess.run("git pull", shell=True)
        subprocess.run(f"git add {TSV_OUTPUT_PATH}/*.txt {TSV_OUTPUT_PATH}/*.tsv", shell=True)
        subprocess.run("git pull", shell=True)
        subprocess.run(f"git commit {TSV_OUTPUT_PATH}/*.txt {TSV_OUTPUT_PATH}/*.tsv -m 'New files.'", shell=True)
        subprocess.run("git pull", shell=True)
        subprocess.run("git push origin main", shell=True)

    print (f'stage1 prompt CID = {stage1_CID}')
    print (f'stage2 prompt CID = {stage2_CID}')
    print (f'{image_file_name} : {ids}\n')
