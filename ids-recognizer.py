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


parser = argparse.ArgumentParser(description='Detect Hanzi-components from image file and generate IDS if possible.')

parser.add_argument('image_files', nargs='*', help='Image file name to process')
parser.add_argument('--model', help='MLX-VLM model path', default='mlx-community/Qwen3.5-27B-heretic-8bit') 

args = parser.parse_args()


# Load the model
model_path = args.model
model_separator_pos = model_path.find('/')
model_name = model_path[model_separator_pos + 1:]

model, processor = load(model_path)
config = load_config(model_path)

prompt_E = '''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

component_examples_of_left = '口(叶),亻,扌,忄,𤣩,彳,糹,釒,訁,飠,⻖,犭,衤,礻,⺬,⻊,冫,氵,支,木,糸,舌,孚,辛,甫,𠦝,亨,豈,歹,文,龠,禾,齒,雚,蒙,𠁣,𩰋,音,𨥫,坐,了,共,龺(朝),同,𫵖,子,孑,斉,咅,仌,丁,亭,彦,矢(知),㠯(𰀥),𦰩(難),巳(𠨎),己(𠨎),刃(𱐔),彡(須),立(䇃)'

component_examples_of_right = '刂,⻏,卩,攵(牧),攴,口,乚,夌,垔,豈,栗,倉,冥,彔,欠,𠃛,𩰊,㣊,矣,菐,犬,丣,坐,與,舁,頃,了,共,巽,同,子,齊,斉,咅,仌,互,亟,丁,亭,彦,矢,難(儺),㠯(佀),矣(俟),尹(伊),巳(𠨎),己(𠨎),巽(撰),刃(仞),彡(杉),頁(頃),立(位),冉(呥),厶(私)'

component_examples_of_above = '亠,宀,冖,⺮,艹,癶,𭼽,𪱙,罒,覀,⺷(義),屮(㞷),爫,彐,彑,夂(冬),文,䒑,业,兴,加,卯,次,所,𣅀,立(音),㐭,𠆢,亼,亽,六,𫩠,八,⺜,𠀎,准,禾,𦥯,龹,𤇾,𰃮,𫇦,髟,冎,𡨄,𣦼,殸,攸,氶(丞),处,丣,一,𦥑,與,頃,了,共,龻,𰀉,灾,𦥔,吅,咅,亟,𦭝(蔑),難(臡),𠀐(貴),亞(亞),㇇(氶),𠨎(巽),刃(忍),彡(辵),攵(㣊),不(否),冉(𣅾),厶(弁)'

component_examples_of_below = '⺗,灬,龰,夂(夏),夊,口,二,儿,几,了,子,旦,丂,𰆊,八,大,犬,𬺢(具),厶(去),彡,難,菐,廾,𪱙,丣,坐,一(丞),與,舁,頃,了,共,巽,同,吅,互(𦬚),亟,丁,亭,屮(𡗡),矢(矣),難(𦍀),㠯(官),&CDP-8DE0;(𪟊),𪟊(寡),巽(𦺈),刃(𦬄),彡(㐱),立(笠),不(示),冉(再)'

component_examples_of_surround_from_upper_left = '厂,𠂆,𠂋(后),厃(危),疒,尸,广,戸,虍,𬻉,倝(幹),产,𠂇(右),麻(磨),鹿,⺶,攸,𠩵,耂,尹(君)'

component_examples_of_surround_from_lower_left = '⻌,廴,走,鬼,麥,麦,風,支,爪,毛,夊,鼠,文,几,乙,𠃊,元,克,光,是'

component_examples_of_full_surround = '囗,行,衣,井,𦥑,二,㗊,互(𠀕)'

component_examples_of_surround_from_above = '門(聞),鬥(闘),几(凧),冂(囘),𰃦（向),凡(風),齊,斉,𣎆,𦝠,戌'

component_examples_of_surround_from_left = '匚,匸'

component_examples_of_surround_from_upper_right = '勹,气,戈,弋,⺄'

component_examples_of_surround_from_below = '凵,𠒂,舁'

component_examples_of_enclosed = '丶(丼),口(哀),歹(夙),女(威),日(間),同(興),王(匡),仌(𠕎)'

component_examples_of_middle = '丩(嘂),頁(囂),言(龻),日(龺),𡵉(微),合(搿),分(椕),冖(牵),田(畫),⺣(稥),厶(窓),冖(亭),𠀎(𡨄),丨(攸),一(兴),𦰩(攤),水(丞),亅(水),厶(𣏋)'

component_examples_of_upper_right = '力(勉),匕(匙),㠯(𲏘),巽(選),彡(尨)'

component_examples_of_lower_right = '彡(修),其(旗),力(勝),㔾(卮),子(㞌),亟(𢉗),丁(庁),矢(侯),矣(𡱢),㠯(𢈂),立(𢨶),攵(䖍),冉(㾆)'

component_examples_of_lower_left = '十(卂),口(句),口(命),𬺣(或),立(𣱠)'

component_examples_of_upper_left = '土(敖),氵(柒),氵(染),叕(歠),日(猒),瓜(瓥),耳(聖)'

component_examples_of_sandwiched = '了(氶),𬼶(亟),⿱丂一(亟)'

component_examples_of_middle_left = '口(亟),㇇(丞)'

component_examples_of_middle_right = '又(亟),品(區),矢(医),&CDP-85BF;(丞),厶(鬼)'

prompt_E5 = f'''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (left(e.g.{component_examples_of_left})/right(e.g.{component_examples_of_right})/above(e.g.{component_examples_of_above})/below(e.g.{component_examples_of_below})/surround-from-upper-left(e.g.{component_examples_of_surround_from_upper_left})/surround-from-lower-left(e.g.{component_examples_of_surround_from_lower_left})/full-surround(e.g.{component_examples_of_full_surround})/surround-from-above(e.g.{component_examples_of_surround_from_above})/surround-from-left(e.g.{component_examples_of_surround_from_left})/surround-from-upper-right(e.g.{component_examples_of_surround_from_upper_right})/surround-from-below(e.g.{component_examples_of_surround_from_below})/upper-left(e.g.{component_examples_of_upper_left})/upper-right(e.g.{component_examples_of_upper_right})/lower-right(e.g.{component_examples_of_lower_right})/lower-left(e.g.{component_examples_of_lower_left})/enclosed(e.g.{component_examples_of_enclosed})/middle(e.g.{component_examples_of_middle})/sandwiched(e.g.{component_examples_of_sandwiched})/middle-left(e.g.{component_examples_of_middle_left})/middle-right(e.g.{component_examples_of_middle_right}))
'''

prompt_cE = '''<image> Locate every component of the classical Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_C = '''<image> Locate every component of the Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_cC = '''<image> Locate every component of the classical Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_J = '''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_cJ = '''<image> Locate every component of the classical Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_JcE = '''<image> Locate every component of the Kanji (or classical Chinese character).
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

prompt_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	相対位置（left(偏。例：{component_examples_of_left})/right(旁。例：{component_examples_of_right})/above(冠。例：{component_examples_of_above})/below(脚。例：{component_examples_of_below})/surround-from-upper-left(垂。例：{component_examples_of_surround_from_upper_left})/surround-from-lower-left(繞。例：{component_examples_of_surround_from_lower_left})/full-surround(構。例：{component_examples_of_full_surround})/surround-from-above(構。例：{component_examples_of_surround_from_above})/surround-from-left(例：{component_examples_of_surround_from_left})/surround-from-upper-right(例：{component_examples_of_surround_from_upper_right})/surround-from-below(例：{component_examples_of_surround_from_below})/upper-left(左上。例：{component_examples_of_upper_left})/upper-right(右上。例：{component_examples_of_upper_right})/lower-left(左下。例：{component_examples_of_lower_left})/lower-right(右下。例：{component_examples_of_lower_right})/enclosed(構えの中の部品。例：{component_examples_of_enclosed})/middle(左右や上下の間の部品。例：{component_examples_of_middle})/sandwiched(例：{component_examples_of_sandwiched})/middle-left(例：{component_examples_of_middle_left})/middle-right(例：{component_examples_of_middle_right})))
'''

prompt_zh_TW = '''找出圖像中漢字的所有組成部分。
以 TSV 格式輸出每個找到的組成部分及其對應的直角座標。
範例：
X0	Y0	X1	Y1	組成部分	相對位置(left(偏。例如:口,亻,扌,忄,𤣩,彳,糹,釒,訁,飠,⻖,犭,衤,礻,⺬,⻊,冫,氵,麥,麦,風,支,鼠,木,糸)/right(旁。例如:刂,⻏,卩,攵,攴,口)/above(頭。例如:宀,冖,⺮,艹,癶,罒,覀,屮,爫,彐,彑,夂,文)/below(底。例如:⺗,灬,龰,夂,夊,口)/surround-from-upper-left(例如:厂,疒,尸,广,戸,虍)/surround-from-lower-left(例如:⻌,廴,走,鬼,麥,麦,風,支,爪,毛,夊,鼠,文,几)/full-surround(同時夾著文字上下或左右。例如:囗,行,衣)/surround-from-above(例如:門,几,冂,鬥)/surround-from-left(匚,匸など)/surround-from-upper-right(例如:勹,气,戈,弋)/surround-from-below(例如:凵)/upper-left(左上)/upper-right(右上)/lower-left(左下)/lower-right(右下)/enclosed(内部的部件)/middle(夾在左右或上下之間的部件))
'''

#prompt = prompt_J
#prompt = prompt_E
#prompt = prompt_cE
#prompt = prompt_C
#prompt = prompt_ja
#prompt = prompt_zh_TW
prompt = prompt_E5

simpler_prompt = f'''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (left(e.g.{component_examples_of_left})/right(e.g.{component_examples_of_right})/above(e.g.{component_examples_of_above})/below(e.g.{component_examples_of_below})/surround/upper-left(e.g.{component_examples_of_upper_left})/lower-left(e.g.{component_examples_of_lower_left})/lower-right(e.g.{component_examples_of_lower_right})/middle(e.g.{component_examples_of_middle}))
'''

component_prompt = '''<image> Run OCR for component (or structure) of the Kanji and output the result.
'''

character_prompt = '''<image> Run Kanji (or traditional Hanzi) OCR and output the result.
'''

def run_VLM (images, prompt):
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images = len(images)
    )
    
    # Generate output
    response = generate(model, processor, formatted_prompt, images,
                        max_tokens = 512, temperature=0.0,
                        verbose=False)
    print (response)
    return response.text

def detect_ids (X1, Y1, X2, Y2, Component_Text, Component_Position):
    number_of_components = len(Component_Text)
    if number_of_components == 1:
        if ( ( Component_Position[0] == 'full-surround' ) or
             ( Component_Position[0] == 'full' ) ):
            return Component_Text[0]

    elif number_of_components == 2:
        if Component_Text[1] == '四点底':
            Component_Text[1] = '灬'
        match Component_Position[0]:
            case 'left':
                if ( ( Component_Position[1] == 'right' ) or
                     ( Component_Position[1] == 'surround-from-lower-right' ) or
                     ( Component_Position[1] == 'full-surround' ) or
                     ( Component_Position[1] == 'full' ) ):
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'upper-left':
                if Component_Position[1] == 'lower-right':
                    if ( ( Component_Text[0] == '言' ) or
                         ( Component_Text[0] == '訁' ) or
                         ( Component_Text[0] == '鬲' ) ):
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '雨' ) or
                           ( Component_Text[0] == '⺮' ) or
                           ( Component_Text[1] == '儿' ) or
                           ( Component_Text[1] == '女' ) or
                           ( Component_Text[1] == '心' ) or
                           ( Component_Text[1] == '虫' ) or
                           ( X1[1] < X2[0] ) or
                           ( Y2[0] <= Y1[1] ) ):
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '夕' ) or
                           ( Component_Text[0] == '𠂇' ) ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'

                elif Component_Position[1] == 'lower-left':
                    if ( Component_Text[0] == '辶' ):
                        return f'⿺{Component_Text[1]}{Component_Text[0]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

                elif Component_Position[1] == 'upper-right':
                    if Y2[0] <= Y1[0]:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'

                elif ( ( Component_Position[1] == 'full-surround' ) or
                       ( Component_Position[1] == 'enclosed' ) ):
                    if Component_Text[1] == '儿':
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'

                elif Component_Text[1] == '口':
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'

                elif Component_Text[0] == '𠂇':
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'

                elif Component_Text[0] == '麻':
                    if ( Y2[0] > Y1[1] ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

                else:
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'lower-left':
                if Component_Position[1] == 'lower-right':
                    if ( ( {Component_Text[0]} == '攵' ) or
                         ( {Component_Text[0]} == '糸' ) or
                         ( ( abs( Y1[1] - Y1[0] ) < 5 ) and
                           ( abs( Y2[1] - Y2[0] ) < 5 ) ) ):
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                    else:
                    # if ( ( Component_Position[1] == 'upper-right' ) or
                    #      ( Component_Position[1] == 'enclosed'    ) )
                    #     if ( ( {Component_Text[0]} == '辶' ) or
                    #          ( {Component_Text[0]} == '廴' ) or
                    #          ( {Component_Text[0]} == '走' ) ):
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'

                elif ( ( Component_Position[1] == 'right' ) or
                       ( Component_Position[1] == 'upper-right' ) or
                       ( Component_Position[1] == 'full-surround' ) or
                       ( Component_Position[1] == 'full' ) or
                       ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'right':
                if Component_Position[1] == 'left':
                    return f'⿰{Component_Text[1]}{Component_Text[0]}'

            case 'above':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'full-surround' ) or
                     ( Component_Position[1] == 'full' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'upper':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'lower' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    if Component_Text[0] == '气':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'below':
                if Component_Position[1] == 'above':
                    return f'⿱{Component_Text[1]}{Component_Text[0]}'
                elif Component_Position[1] == 'surround-from-above':
                    return f'⿵{Component_Text[0]}{Component_Text[1]}'

            case 'lower':
                if Component_Position[1] == 'upper':
                    return f'⿱{Component_Text[1]}{Component_Text[0]}'
                elif Component_Position[1] == 'surround-from-above':
                    return f'⿵{Component_Text[0]}{Component_Text[1]}'

            case 'upper-right':
                if Component_Position[1] == 'lower-left':
                    if Component_Text[0] == '戈':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

                elif ( ( Component_Position[1] == 'enclosed' ) or
                       ( Component_Position[1] == 'lower-right' ) ):
                    return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'surround':
                if Component_Position[1] == 'middle':
                    if ( ( Component_Text[0] == '⻌' ) or
                         ( Component_Text[0] == '廴' ) ):
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'full-surround':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    if Component_Text[0] == '凵':
                        return f'⿶{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '几':
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '广' ) or
                           ( Component_Text[0] == '麻' ) or
                           ( Component_Text[0] == '尸' ) ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '宀':
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '門' ) or
                           ( Component_Text[0] == '冂' ) ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '戈':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿴{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-above':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'covered' ) or
                     ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    if ( ( Component_Text[0] == '虍' ) or
                         ( Component_Text[0] == '鹿' ) or
                         ( Component_Text[0] == '广' ) or
                         ( Component_Text[0] == '疒' ) or
                         ( Component_Text[0] == '尸' ) ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '宀' ) or
                           ( Component_Text[0] == '穴' ) or
                           ( Component_Text[0] == '冖' ) or
                           ( Component_Text[0] == '雨' ) ):
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-below':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    if Component_Text[0] == '皿':
                        return f'⿱{Component_Text[1]}{Component_Text[0]}'
                    else:
                        return f'⿶{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'full' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'covered' ) ):
                    if Component_Text[0] == '疒':
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '門') or
                           ( Component_Text[0] == '冂') ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '⻌' ) or
                           ( Component_Text[0] == '廴' ) ):
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'
                elif Component_Position[1] == 'right':
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-right' ) ):
                    if Component_Text[0] == '匚':
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '几':
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '勹':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-lower-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'upper-right' ) ):
                    return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿼{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-left' ) ):
                    return f'⿹{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-lower-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'upper-left' ) ):
                    return f'⿽{Component_Text[0]}{Component_Text[1]}'

    elif number_of_components == 3:
        match Component_Position[0]:
            case 'upper':
                if ( ( Component_Position[1] == 'middle' ) and
                     ( ( Component_Position[2] == 'lower' ) or
                       ( Component_Position[2] == 'below' ) ) ):
                    if Component_Text[1] == Component_Text[2]:
                        return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'lower-left' ) and
                       ( Component_Position[2] == 'lower-right' ) ):
                    return f'⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}'
                elif ( ( Component_Position[1] == 'surround-from-above' ) and
                       ( Component_Position[2] == 'middle' ) ):
                    if Component_Text[1] == '囗':
                        return f'⿱{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿱{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'

            case 'above':
                if ( ( Component_Position[1] == 'middle' ) and
                     ( ( Component_Position[2] == 'lower' ) or
                       ( Component_Position[2] == 'below' ) ) ):
                    if Component_Text[1] == Component_Text[2]:
                        return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'enclosed' ) and
                       ( Component_Position[2] == 'below' ) ):
                    return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'lower-left' ) and
                       ( Component_Position[2] == 'lower-right' ) ):
                    return f'⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}'
                elif ( ( Component_Position[1] == 'surround-from-above' ) and
                       ( Component_Position[2] == 'middle' ) ):
                    if Component_Text[1] == '囗':
                        return f'⿱{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿱{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'

            case 'upper-left':
                if ( ( Component_Position[1] == 'upper-right' ) and
                     ( ( Component_Position[2] == 'lower' ) or
                       ( Component_Position[2] == 'below' ) or
                       ( ( Component_Position[2] == 'lower-right' ) and
                         ( X1[2] < X2[0] ) )
                      ) ):
                    return f'⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'right' ) and
                       ( Component_Position[2] == 'below' ) ):
                    if ( abs(Y1[1] - Y1[0] ) < 5 ):
                        return f'⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'middle' ) and
                       ( Component_Position[2] == 'lower-right' ) ):
                    return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'middle' ) and
                       ( Component_Position[2] == 'lower-left' ) ):
                    return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'full-surround' ) and
                       ( Component_Position[2] == 'enclosed' ) ):
                    if Component_Text[1] == '門':
                        return f'⿰{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿰{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'

            case 'left':
                if ( ( ( Component_Position[1] == 'upper-right' ) or
                       ( Component_Position[1] == 'right' ) ) and
                     ( Component_Position[2] == 'lower-right' ) ):
                    return f'⿰{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                elif ( ( Component_Position[1] == 'right' ) and
                       ( Component_Position[2] == 'below' ) ):
                    return f'⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
                elif ( ( Component_Position[1] == 'full-surround' ) and
                       ( Component_Position[2] == 'enclosed' ) ):
                    if ( ( Component_Text[1] == '匚' ) or
                         ( Component_Text[1] == '匸' ) ):
                        return f'⿰{Component_Text[0]}⿷{Component_Text[1]}{Component_Text[2]}'
                    elif ( ( Component_Text[1] == '門' ) or
                           ( Component_Text[1] == '鬥' ) or
                           ( Component_Text[1] == '几' ) or
                           ( Component_Text[1] == '冂' ) or
                           ( Component_Text[1] == '𰃦' ) or
                           ( Component_Text[1] == '凡' ) or
                           ( Component_Text[1] == '齊' ) or
                           ( Component_Text[1] == '𣎆' ) or
                           ( Component_Text[1] == '𦝠' ) or
                           ( Component_Text[1] == '戌' ) ):
                        return f'⿰{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿰{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'

    elif number_of_components == 4:
        match Component_Position[0]:
            case 'upper':
                if ( ( Component_Position[1] == 'lower-left' ) and
                     ( Component_Position[2] == 'lower-right' ) and
                     ( Component_Position[3] == 'below' ) ):
                    if ( ( Component_Text[0] == Component_Text[1] ) and
                         ( Component_Text[1] == Component_Text[2] ) ):
                        return f'⿱⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                    else:
                        return f'⿳{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
            case 'above':
                if ( ( Component_Position[1] == 'lower-left' ) and
                     ( Component_Position[2] == 'lower-right' ) and
                     ( Component_Position[3] == 'below' ) ):
                    if ( ( Component_Text[0] == Component_Text[1] ) and
                         ( Component_Text[1] == Component_Text[2] ) ):
                        return f'⿱⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                    else:
                        return f'⿳{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                elif ( ( Component_Position[1] == 'above' ) and
                       ( Component_Position[2] == 'surround-from-above' ) and
                       ( Component_Position[3] == 'enclosed' ) ):
                    return f'⿳{Component_Text[0]}{Component_Text[1]}⿵{Component_Text[2]}{Component_Text[3]}'
                elif ( ( Component_Text[0] == '亠' ) and
                       ( Component_Text[1] == '口' ) and
                       ( Component_Text[2] == '冖' ) ):
                    return f'⿱⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'

            case 'upper-left':
                if ( ( Component_Position[1] == 'upper-right' ) and
                     ( Component_Position[2] == 'surround-from-above' ) and
                     ( Component_Position[3] == 'enclosed' ) ):
                    if ( Component_Text[2] == '冖' ):
                        return f'⿱⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                    else:
                        return f'⿳⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'

            case 'upper-middle':
                if ( ( Component_Position[1] == 'lower-left' ) and
                     ( Component_Position[2] == 'lower-right' ) and
                     ( Component_Position[3] == 'below' ) ):
                    return f'⿱⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'

            case 'surround-from-upper-left':
                if ( ( Component_Position[1] == 'upper-right' ) and
                     ( Component_Position[2] == 'lower-left' ) and
                     ( Component_Position[3] == 'lower-right' ) ):
                    if Component_Text[0] == '匚':
                        return f'⿷{Component_Text[0]}⿱{Component_Text[1]}⿰{Component_Text[2]}{Component_Text[3]}'
                    else:
                        return f'⿸{Component_Text[0]}⿱{Component_Text[1]}⿰{Component_Text[2]}{Component_Text[3]}'

def merge_left_and_right (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[0], X1[1])
    cy1 = min (Y1[0], Y1[1])
    cx2 = max (X2[0], X2[1])
    cy2 = max (Y2[0], Y2[1])
    orig_comp1 = Component_Text[0]
    orig_comp2 = Component_Text[1]
    del X1[1]
    del Y1[1]
    del X2[1]
    del Y2[1]
    del Component_Text[1]
    del Component_Position[1]
    X1[0] = cx1
    Y1[0] = cy1
    X2[0] = cx2
    Y2[0] = cy2
    print(f'-> ({X1[0]},{Y1[0]})-({X2[0]},{Y2[0]})')
    if ( ( (X2[0] - X1[0]) > 0 ) and
         ( (Y2[0] - Y1[0]) > 0 ) ):
        im_crop = im.crop((X1[0], Y1[0], X2[0], Y2[0]))
        comp1_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp1.png'
        im_crop.save(comp1_image_file_name)

        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        if (os.path.isfile(comp2_image_file_name)):
            os.remove(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.rename(comp3_image_file_name, comp2_image_file_name)

        comp1_response = run_VLM ([comp1_image_file_name], component_prompt)
        print (f'new component = "{comp1_response}".')
        Component_Position[0] = 'above'
        if len(comp1_response) == 1:
            Component_Text[0] = comp1_response
        else:
            Component_Text[0] = f'⿰{Component_Text[0]}{orig_comp2}'
    else:
        Component_Text[0] = f'⿰{Component_Text[0]}{orig_comp2}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_below_left_and_right (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[1], X1[2])
    cy1 = min (Y1[1], Y1[2])
    cx2 = max (X2[1], X2[2])
    cy2 = max (Y2[1], Y2[2])
    orig_comp2 = Component_Text[1]
    orig_comp3 = Component_Text[2]
    del X1[2]
    del Y1[2]
    del X2[2]
    del Y2[2]
    del Component_Text[2]
    del Component_Position[2]
    X1[1] = cx1
    Y1[1] = cy1
    X2[1] = cx2
    Y2[1] = cy2
    print(f'-> ({X1[1]},{Y1[1]})-({X2[1]},{Y2[1]})')
    if ( ( (X2[1] - X1[1]) > 0 ) and
         ( (Y2[1] - Y1[1]) > 0 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp2_response = run_VLM ([comp2_image_file_name], component_prompt)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'below'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿰{orig_comp2}{orig_comp3}'
    else:
        Component_Text[1] = f'⿰{orig_comp2}{orig_comp3}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_above_and_below (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[0], X1[1])
    cy1 = min (Y1[0], Y1[1])
    cx2 = max (X2[0], X2[1])
    cy2 = max (Y2[0], Y2[1])
    orig_comp1 = Component_Text[0]
    orig_comp2 = Component_Text[1]
    del X1[1]
    del Y1[1]
    del X2[1]
    del Y2[1]
    del Component_Text[1]
    del Component_Position[1]
    X1[0] = cx1
    Y1[0] = cy1
    X2[0] = cx2
    Y2[0] = cy2
    print(f'-> ({X1[0]},{Y1[0]})-({X2[0]},{Y2[0]})')
    if ( ( (X2[0] - X1[0]) > 0 ) and
         ( (Y2[0] - Y1[0]) > 0 ) ):
        im_crop = im.crop((X1[0], Y1[0], X2[0], Y2[0]))
        comp1_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp1.png'
        im_crop.save(comp1_image_file_name)

        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        if (os.path.isfile(comp2_image_file_name)):
            os.remove(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.rename(comp3_image_file_name, comp2_image_file_name)

        comp1_response = run_VLM ([comp1_image_file_name], component_prompt)
        print (f'new component = "{comp1_response}".')
        if len(comp1_response) == 1:
            Component_Text[0] = comp1_response
        else:
            Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
    else:
        Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_right_above_and_below (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]
    image_width, image_height = im.size

    cx1 = min (X1[1], X1[2])
    cy1 = 0 # min (Y1[1], Y1[2])
    cx2 = image_width - 1  # max (X2[1], X2[2])
    cy2 = image_height - 1 # max (Y2[1], Y2[2])
    orig_comp2 = Component_Text[1]
    orig_comp3 = Component_Text[2]
    del X1[2]
    del Y1[2]
    del X2[2]
    del Y2[2]
    del Component_Text[2]
    del Component_Position[2]
    X1[1] = cx1
    Y1[1] = cy1
    X2[1] = cx2
    Y2[1] = cy2
    print(f'-> ({X1[1]},{Y1[1]})-({X2[1]},{Y2[1]})')
    if ( ( (X2[1] - X1[1]) > 0 ) and
         ( (Y2[1] - Y1[1]) > 0 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp2_response = run_VLM ([comp2_image_file_name], component_prompt)
        print (f'new component = "{comp2_response}".')
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿱{orig_comp2}{orig_comp3}'
    else:
        Component_Text[1] = f'⿱{orig_comp2}{orig_comp3}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_enclosed_verticale3 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[1], X1[2], X1[3])
    cy1 = min (Y1[1], Y1[2], Y1[3])
    cx2 = max (X2[1], X2[2], X2[3])
    cy2 = max (Y2[1], Y2[2], Y2[3])
    orig_comp2 = Component_Text[1]
    orig_comp3 = Component_Text[2]
    orig_comp4 = Component_Text[3]
    del X1[2:4]
    del Y1[2:4]
    del X2[2:4]
    del Y2[2:4]
    del Component_Text[2:4]
    del Component_Position[2:4]
    X1[1] = cx1
    Y1[1] = cy1
    X2[1] = cx2
    Y2[1] = cy2
    print(f'-> ({X1[1]},{Y1[1]})-({X2[1]},{Y2[1]})')
    if ( ( (X2[1] - X1[1]) > 0 ) and
         ( (Y2[1] - Y1[1]) > 0 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp4_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp4.png'
        if (os.path.isfile(comp4_image_file_name)):
            os.remove(comp4_image_file_name)

        comp2_response = run_VLM ([comp2_image_file_name], component_prompt)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'enclosed'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'
    else:
        Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_below_left_and_vertical2 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[1], X1[2], X1[3])
    cy1 = min (Y1[1], Y1[2], Y1[3])
    cx2 = max (X2[1], X2[2], X2[3])
    cy2 = max (Y2[1], Y2[2], Y2[3])
    orig_comp2 = Component_Text[1]
    orig_comp3 = Component_Text[2]
    orig_comp4 = Component_Text[3]
    del X1[2:4]
    del Y1[2:4]
    del X2[2:4]
    del Y2[2:4]
    del Component_Text[2:4]
    del Component_Position[2:4]
    X1[1] = cx1
    Y1[1] = cy1
    X2[1] = cx2
    Y2[1] = cy2
    print(f'-> ({X1[1]},{Y1[1]})-({X2[1]},{Y2[1]})')
    if ( ( (X2[1] - X1[1]) > 0 ) and
         ( (Y2[1] - Y1[1]) > 0 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp4_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp4.png'
        if (os.path.isfile(comp4_image_file_name)):
            os.remove(comp4_image_file_name)

        comp2_response = run_VLM ([comp2_image_file_name], component_prompt)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'below'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿰{orig_comp2}⿱{orig_comp3}{orig_comp4}'
    else:
        Component_Text[1] = f'⿰{orig_comp2}⿱{orig_comp3}{orig_comp4}'

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    ids_file_name  = f'{OUTPUT_PATH}/{basename}_ids.txt'
    full_file_name = f'{OUTPUT_PATH}/{basename}_full.txt'
    print (image_file, prompt)
    images = [ image_file ]

    response = run_VLM (images, prompt)

    component_number = 0
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    Component_Text = []
    Component_Position = []
    with open(f'{TSV_OUTPUT_PATH}/{basename}.tsv', 'w', encoding = 'utf-8') as tsv_destfile:
        for line_match in re.findall('([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([^() \t\n\r]+?)(\(.+\))?\s+([a-z-]+)\S*\n?', response):
            x1, y1, x2, y2, line_text, comment, position = line_match
            x1 = int (x1)
            y1 = int (y1)
            x2 = int (x2)
            y2 = int (y2)
            orx1 = round ( ( x1 * image_width  ) / 1000 )
            ory1 = round ( ( y1 * image_height ) / 1000 )
            orx2 = round ( ( x2 * image_width  ) / 1000 )
            ory2 = round ( ( y2 * image_height ) / 1000 )
            # orw  = round ( ( ( x2 - x1 ) * image_width)  / 1000 )
            # orh  = round ( ( ( y2 - y1 ) * image_height) / 1000 )
            X1.append(orx1)
            Y1.append(ory1)
            X2.append(orx2)
            Y2.append(ory2)
            Component_Text.append(line_text)
            Component_Position.append(position)
            print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}')
            print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}',
                   file=tsv_destfile)
            component_number = component_number + 1
            if ( ( (orx2 - orx1) > 0 ) and
                 ( (ory2 - ory1) > 0 ) ):
                im_crop = im.crop((orx1, ory1, orx2, ory2))
                im_crop.save(f'{TSV_OUTPUT_PATH}/{basename}_comp{component_number}.png')

    if len(Component_Text) == 3:
        match Component_Position[0]:
            case 'left':
                if ( ( Component_Position[1] == 'right' ) and
                     ( Component_Position[2] == 'below' ) ):
                    if ( ( Component_Text[0] != '耳' ) or
                         ( Component_Text[1] != '口' ) ):
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                                   X1, Y1,
                                                                                                   X2, Y2,
                                                                                                   Component_Text,
                                                                                                   Component_Position,
                                                                                                   TSV_OUTPUT_PATH)
                        Component_Position[0] = 'above'
                        Component_Position[1] = 'below'

                elif ( ( ( Component_Position[1] == 'upper-right' ) or
                         ( Component_Position[1] == 'right' ) )
                       and
                       ( Component_Position[2] == 'right' ) ):
                    X1, Y1, X2, Y2, Component_Text, Component_Position = merge_right_above_and_below (image_file,
                                                                                                      X1, Y1,
                                                                                                      X2, Y2,
                                                                                                      Component_Text,
                                                                                                      Component_Position,
                                                                                                      TSV_OUTPUT_PATH)
                    Component_Position[0] = 'left'
                    Component_Position[1] = 'right'

                elif ( ( Component_Position[1] == 'enclosed' ) and
                       ( Component_Position[2] == 'below' ) ):
                    X1, Y1, X2, Y2, Component_Text, Component_Position = merge_right_above_and_below (image_file,
                                                                                                      X1, Y1,
                                                                                                      X2, Y2,
                                                                                                      Component_Text,
                                                                                                      Component_Position,
                                                                                                      TSV_OUTPUT_PATH)
                    Component_Position[0] = 'left'
                    Component_Position[1] = 'right'

                elif ( ( X2[0] <= X1[1] ) and
                       ( X2[0] <= X1[2] ) and
                       ( abs(Y1[2] - Y2[1]) < 5 ) ):
                    if ( ( Component_Position[1] != 'above' ) or
                         ( Component_Position[2] != 'below' ) ):
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_right_above_and_below (image_file,
                                                                                                          X1, Y1,
                                                                                                          X2, Y2,
                                                                                                          Component_Text,
                                                                                                          Component_Position,
                                                                                                          TSV_OUTPUT_PATH)
                        Component_Position[0] = 'left'
                        Component_Position[1] = 'right'

            case 'upper-left':
                if Component_Position[1] == 'upper-right':
                    if ( ( Component_Position[2] == 'lower-left' ) or
                         ( Component_Position[2] == 'lower-right' ) ):
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                                   X1, Y1,
                                                                                                   X2, Y2,
                                                                                                   Component_Text,
                                                                                                   Component_Position,
                                                                                                   TSV_OUTPUT_PATH)
                        Component_Position[0] = 'above'
                        Component_Position[1] = 'below'

                    elif ( ( abs(X1[1] - X1[0]) < 5 ) and
                           ( abs(X2[1] - X2[0]) < 5 ) and
                           ( Y1[1] <= Y2[0] ) ):
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                                   X1, Y1,
                                                                                                   X2, Y2,
                                                                                                   Component_Text,
                                                                                                   Component_Position,
                                                                                                   TSV_OUTPUT_PATH)
                        Component_Position[0] = 'above'
                        Component_Position[1] = 'below'

            case 'upper-right':
                if Component_Position[1] == 'full-surround':
                    if ( ( abs(X1[1] - X1[0]) < 5 ) and
                         ( abs(X2[1] - X2[0]) < 5 ) and
                         ( Y1[1] <= Y2[0] ) ):
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                                   X1, Y1,
                                                                                                   X2, Y2,
                                                                                                   Component_Text,
                                                                                                   Component_Position,
                                                                                                   TSV_OUTPUT_PATH)
                        Component_Position[0] = 'above'
                        Component_Position[1] = 'below'

            case 'surround-from-left':
                if Component_Position[1] == 'surround-from-right':
                    if Component_Position[2] == 'enclosed':
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                                   X1, Y1,
                                                                                                   X2, Y2,
                                                                                                   Component_Text,
                                                                                                   Component_Position,
                                                                                                   TSV_OUTPUT_PATH)
                        Component_Position[0] = 'above'
                        Component_Position[1] = 'below'
                        
            case 'surround-from-above':
                if Component_Position[1] == 'lower-left':
                    if Component_Position[2] == 'lower-right':
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_above_and_below (image_file,
                                                                                                    X1, Y1,
                                                                                                    X2, Y2,
                                                                                                    Component_Text,
                                                                                                    Component_Position,
                                                                                                    TSV_OUTPUT_PATH)
                        Component_Position[0] = 'surround-from-above'
                        Component_Position[1] = 'enclosed'

    elif len(Component_Text) == 4:
        print (X1, Y1, X2, Y2, Component_Text, Component_Position)
        if ( ( X1[0] == X1[1] ) and
             ( Y1[0] == Y1[1] ) and
             ( X2[0] == X2[1] ) and
             ( Y2[0] == Y2[1] ) and
             ( X1[2] == X1[3] ) and
             ( Y1[2] == Y1[3] ) and
             ( X2[2] == X2[3] ) and
             ( Y2[2] == Y2[3] ) and
             ( Component_Text[0] == Component_Text[1] ) and
             ( Component_Text[2] == Component_Text[3] ) ):
            print ('duplicate c1 = c2 and c3 = c4')
            del X1[3]
            del Y1[3]
            del X2[3]
            del Y2[3]
            del Component_Text[3]
            del Component_Position[3]
            del X1[1]
            del Y1[1]
            del X2[1]
            del Y2[1]
            del Component_Text[1]
            del Component_Position[1]
             
        elif ( ( ( Component_Position[0] == 'surround-from-upper-left' ) or
               ( Component_Position[0] == 'surround-from-lower-right' ) ) and
             ( Component_Position[1] == 'upper-left' ) and
             ( Component_Position[2] == 'upper-right' ) and
             ( Component_Position[3] == 'enclosed' ) ):
            if ( ( abs(Y1[2] - Y2[1]) < 5 ) and
                 ( abs(Y1[3] - Y2[2]) < 5 ) ):
                X1, Y1, X2, Y2, Component_Text, Component_Position = merge_enclosed_verticale3 (image_file,
                                                                                                X1, Y1,
                                                                                                X2, Y2,
                                                                                                Component_Text,
                                                                                                Component_Position,
                                                                                                TSV_OUTPUT_PATH)

        elif ( ( Component_Position[0] == 'upper-left' ) and
               ( Component_Position[1] == 'upper-right' ) and
               ( Component_Position[2] == 'lower-left' ) and
               ( Component_Position[3] == 'lower-right' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_left_and_right (image_file,
                                                                                       X1, Y1,
                                                                                       X2, Y2,
                                                                                       Component_Text,
                                                                                       Component_Position,
                                                                                       TSV_OUTPUT_PATH)
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_below_left_and_right (image_file,
                                                                                             X1, Y1,
                                                                                             X2, Y2,
                                                                                             Component_Text,
                                                                                             Component_Position,
                                                                                             TSV_OUTPUT_PATH)
            Component_Position[0] = 'above'

        elif ( ( Component_Position[0] == 'upper-left' ) and
               ( Component_Position[1] == 'lower-left' ) and
               ( Component_Position[2] == 'lower-right' ) and
               ( Component_Position[3] == 'lower-right' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_below_left_and_vertical2 (image_file,
                                                                                                 X1, Y1,
                                                                                                 X2, Y2,
                                                                                                 Component_Text,
                                                                                                 Component_Position,
                                                                                                 TSV_OUTPUT_PATH)
            Component_Position[0] = 'above'

    elif len(Component_Text) == 5:
        if ( ( X1[0] == X1[1] ) and
             ( Y1[0] == Y1[1] ) and
             ( X2[0] == X2[1] ) and
             ( Y2[0] == Y2[1] ) and
             ( Component_Text[0] == Component_Text[1] ) and
             ( X1[3] == X1[4] ) and
             ( Y1[3] == Y1[4] ) and
             ( X2[3] == X2[4] ) and
             ( Y2[3] == Y2[4] ) and
             ( Component_Text[3] == Component_Text[4] ) ):
            print ('duplicate c1 = c2 and c4 = c5')
            del X1[4]
            del Y1[4]
            del X2[4]
            del Y2[4]
            del Component_Text[4]
            del Component_Position[4]
            del X1[1]
            del Y1[1]
            del X2[1]
            del Y2[1]
            del Component_Text[1]
            del Component_Position[1]

        elif ( ( Component_Position[0] == 'upper-left' ) and
             ( Component_Position[1] == 'upper-right' ) and
             ( Component_Position[2] == 'lower-left' ) and
             ( Component_Position[3] == 'lower-right' ) and
             ( Component_Position[4] == 'below' ) ):
            cx1 = min (X1[0], X1[1], X1[2], X1[3])
            cy1 = min (Y1[0], Y1[1], Y1[2], Y1[3])
            cx2 = max (X2[0], X2[1], X2[2], X2[3])
            cy2 = max (Y2[0], Y2[1], Y2[2], Y2[3])
            orig_comp1 = Component_Text[0]
            orig_comp2 = Component_Text[1]
            orig_comp3 = Component_Text[2]
            orig_comp4 = Component_Text[3]
            del X1[0:3]
            del Y1[0:3]
            del X2[0:3]
            del Y2[0:3]
            del Component_Text[0:3]
            del Component_Position[0:3]
            X1[0] = cx1
            Y1[0] = cy1
            X2[0] = cx2
            Y2[0] = cy2
            print(f'-> ({X1[1]},{Y1[1]})-({X2[1]},{Y2[1]})')
            if ( ( (X2[0] - X1[0]) > 0 ) and
                 ( (Y2[0] - Y1[0]) > 0 ) ):
                im_crop = im.crop((X1[0], Y1[0], X2[0], Y2[0]))
                comp1_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp1.png'
                im_crop.save(comp1_image_file_name)

                comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
                if (os.path.isfile(comp2_image_file_name)):
                    os.remove(comp2_image_file_name)

                comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
                if (os.path.isfile(comp3_image_file_name)):
                    os.remove(comp3_image_file_name)

                comp4_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp4.png'
                if (os.path.isfile(comp4_image_file_name)):
                    os.remove(comp4_image_file_name)

                comp5_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp5.png'
                if (os.path.isfile(comp5_image_file_name)):
                    os.rename(comp5_image_file_name, comp2_image_file_name)

                comp1_response = run_VLM ([comp1_image_file_name], component_prompt)
                print (f'new component = "{comp1_response}".')
                Component_Position[0] = 'above'
                if len(comp1_response) == 1:
                    Component_Text[0] = comp1_response
                else:
                    Component_Text[0] = f'⿱⿰{orig_comp1}{orig_comp2}⿰{orig_comp3}{orig_comp4}'
            else:
                Component_Text[0] = f'⿱⿰{orig_comp1}{orig_comp2}⿰{orig_comp3}{orig_comp4}'

    with open(f'{OUTPUT_PATH}/{basename}.txt', 'w', encoding = 'utf-8') as destfile:
        destfile.write(response)

    with open(f'{OUTPUT_PATH}/{basename}.prompt', 'w', encoding = 'utf-8') as prompt_file:
        prompt_file.write(prompt)
    print (X1, Y1, X2, Y2, Component_Text, Component_Position)
    return X1, Y1, X2, Y2, Component_Text, Component_Position

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
    else:
        images = [ image_file ]
        char_response = run_VLM (images, character_prompt)
        print (f'character = "{char_response}".')

        X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                      prompt,
                                                                                      TSV_OUTPUT_PATH,
                                                                                      OUTPUT_PATH)
        if ( ( len(char_response) == 1 ) and
             ( any ( comp == char_response for comp in Component_Text ) ) and
             ( ( len(Component_Text) < 2 ) or
               ( Component_Text[0] != '囗' ) ) ):
            with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                print(char_response, file=full_destfile)
            return char_response
        else:
            ids = detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
            if ids:
                with open(f'{OUTPUT_PATH}/{basename}_ids.txt',
                          'w', encoding = 'utf-8') as ids_destfile:
                    print(ids, file=ids_destfile)
            elif len(Component_Text) == 1:
                if (os.path.isfile(full_file_name)):
                    os.remove(full_file_name)

                if len(char_response) == 1:
                    with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                        print(char_response, file=full_destfile)
                    return char_response
                elif Component_Position[0] == 'full-surround':
                    with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                        print(Component_Text[0], file=full_destfile)

            elif len(Component_Text) > 7:
                X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                              simpler_prompt,
                                                                                              TSV_OUTPUT_PATH,
                                                                                              OUTPUT_PATH)
                if ( ( len(char_response) == 1 ) and
                     ( any ( comp == char_response for comp in Component_Text ) ) ):
                    with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                        print(char_response, file=full_destfile)
                        return char_response
                else:
                    ids = detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
                    if ids:
                        with open(f'{OUTPUT_PATH}/{basename}_ids.txt',
                                  'w', encoding = 'utf-8') as ids_destfile:
                            print(ids, file=ids_destfile)

    return ids


proc = subprocess.run("ipfs add -- | cut -d' ' -f2", shell=True, input=prompt, stdout=PIPE, stderr=PIPE, text=True)
IPFS_CID = proc.stdout.rstrip('\r\n')

print (IPFS_CID)

OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(OUTPUT_PATH, exist_ok=True)
TSV_OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(TSV_OUTPUT_PATH, exist_ok=True)

print (OUTPUT_PATH, TSV_OUTPUT_PATH)

for image_file_name in args.image_files:
    ids = manage_OCR_for_glyph_image (image_file_name, prompt,
                                      TSV_OUTPUT_PATH, OUTPUT_PATH)
    print (f'prompt CID = {IPFS_CID}')
    print (f'{image_file_name} : {ids}\n')
