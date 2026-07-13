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
parser.add_argument('--git', help='Run git add, commit and push') 

args = parser.parse_args()


# Load the model
model_path = args.model
model_separator_pos = model_path.find('/')
model_name = model_path[model_separator_pos + 1:]

model, processor = load(model_path)
config = load_config(model_path)

component_examples_of_left = '口(叶),亻(仁),扌(持),忄(惜),𤣩(球),彳(行),釒(銀),訁(語),飠(飲),⻖(阪),犭(狼),衤(被),礻(神),⺬(祇),⻊(路),冫(冷),氵(河),支(𠚽),木(機),糸,舌,辛(辡),甫,歹,文(刘),龠,禾(秋),齒,雚,蒙,𠁣(門),𩰋(鬥),坐,了,同(𡜝),𫵖,子,孑(孫),斉,咅,仌,丁,亭,彦,矢(知),㠯(𰀥),𦰩(難),巳(𠨎),己(𠨎),刃(𱐔),彡(須),立(䇃),堇(勤),人(从),矛(敄),尚(敞),十(卄),丰(邦),田(畼),矣(欸),豆(頭),豸(豹),羊(𦍏),甬(勈),日(明),昌(𣣘),巾(幅),目(眼),鹿(𪊺),或(𠜻),褱(𣀤),侯(鄇),垔(歅),産(剷),声(殸),方(放),秝(𥣸),止(此),𡰪(辟),夢(𪇓),委(魏)'

component_examples_of_right = '刂(劍),⻏(郭),卩(叩),攵(牧),攴,口(加),乚(礼),夌,垔(煙),豈,栗,倉,冥,彔,欠,𠃛(門),𩰊(鬥),㣊,菐,犬,丣,坐,與,舁,頃,了,共(供),巽(撰),同(銅),子(㝀),齊,斉,咅,仌,互(坘),亟(極),丁,亭,彦,矢(䀢),難(儺),㠯(佀),矣(俟),尹(伊),巳(𠨎),己(𠨎),刃(仞),彡(杉),頁(頃),立(位),冉(呥),厶(私),堇(謹),人(从),从(𪻐),杀(刹),𣏂(剎),矛(䋒),𠂉(𭤨),尚(倘),衣(挔),𧘇(𠇊),几(机),支(伎),十(什),丰(仹),卯(柳),田(畑),尺(択),豆(脰),呉(誤),辡(𡁈),羊(垟),𦉰(𤤡),亡(忙),舞(儛),甬(俑),孔(吼),日(𬽪),昌(唱),𠔼(㧇),巾(𪤷),布(怖),目(相),鹿(漉),修(㹋),或(域),褱(懷),咸(减),侯(喉),四(伵),産(嵼),文(伩),士(仕),声(𠴢),方(㕫),秝(𥕆),禾(秌),亦(㑊),辛(垶),夢(懜),委(倭),鬼(魏),戎(娀),戻(涙),夬(決)'

component_examples_of_above = '亠,宀,冖(冠),⺮,艹,癶,𭼽,罒(睘),覀,⺷(義),屮(㞷),爫,彐,彑,夂(冬),文(产),䒑(屰),业(業),兴,加,卯,次,所,𣅀,立(音),㐭,𠆢,亼,亽,六,𫩠,八,⺜,𠀎,准,禾(季),𦥯,龹,𤇾,𰃮,𫇦,髟,冎,𡨄,𣦼,殸,攸(悠),氶(丞),处,丣,一,𦥑,與,頃,了,共(巷),龻,𰀉,灾,𦥔,吅,咅,亟,𦭝(蔑),難(臡),𠀐(貴),亞(惡),㇇(𡥀),𠨎(巽),刃(忍),彡(辵),攵(㣊),不(否),冉(𣅾),厶(弁),人(介),从(怂),吅(單),𱼀(䍃),矛(柔),巛(巢),𠂉(乞),尚(堂),衣(𧘉),几(殳),支(𥁈),十(古),丰(𣐇),卯(𨥫),田(界),尺(𪽗),𠂒(先),口(兄),羊(羴),龴(甬),亡(巟),甬(勇),子(孟),孔(𡵾),日(早),昌(㫯),𠔼(冡),同(𣑸),巾(𢁝),目(見),鹿(麋),修(𢟅),或(𢃤),咸(感),卄(共),四(𰉐),士(声),秝(𬓶),止(歯),亦(変),辛(㖖),辡(𰺪),夢(𲀎),𦭝(夢),𡗗(春)'

component_examples_of_below = '⺗,灬,龰,夂(夏),夊,口(古),二,儿(兄),几(亢),了,子(字),旦,丂,𰆊,八,大,犬,𬺢(具),厶(去),彡,難,菐,廾(卉),𪱙,丣,坐,一(丞),與,舁,頃,了,共(巽),巽(𦺈),同(𠀹),吅,互(𦬚),亟,丁,亭,屮(𡗡),矢(矣),難(𦍀),㠯(官),𪟊(寡),刃(𦬄),彡(㐱),立(笠),不(示),冉(再),堇(蓳),人(珡),从(𠅃),业(並),朩(杀),朮(𣏂),术(𣏂),吅(品),矛(罞),巛(𠮰),𠦒(華),禸(禺),衣(装),𧘇(衣),支(芰),十(早),丰(夆),卯(奅),田(畗),尺(𫁶),𠨎(𠨕),矣(𦮸),䒑(豆),𠂒(𭇆),豆(壴),呉(茣),羊(𦍒),龴(令),甬(𥦁),孔(芤),日(晶),昌(菖),巾(帀),布(希),目(盲),鹿(蔍),攸(峳),修(蓚),或(𦱂),咸(嵅),侯(篌),垔(𡨾),四(𣳉),厂(产),文(㞵),方(𬙙),秝(𩄞),止(正),禾(𦊜),辛(宰),罒(𦭝),委(萎),鬼(䰟),戎(茙),戻(䈆),夬(䆕)'

component_examples_of_surround_from_upper_left = '厂(厚),𠂆(𠂢),𠂋(后),厃(危),疒(病),尸(屈),广(廣),戸,虍(處),𬻉,𭤨(旗),倝(幹),产(彦),𠂇(右),麻(磨),鹿(麃),⺶,攸(修),厤(曆),𠩵(暦),耂(考),尹(君),巾(𢁟),方(房)'

component_examples_of_surround_from_lower_left = '⻌(進),廴(建),走(起),麥,麦,風,支,爪,毛,夊,鼠,文,乙,𠃊,元,克,光,是,支(𭣗),尺(𡰰),乚(匕),鬼(魅)'

component_examples_of_full_surround = '囗(國),行(街),衣(哀),井(丼),𦥑,二,㗊(器),互(𠀕),卯(卿),辡(辨),四(囧),秝(𥣲),辡(辨)'

component_examples_of_surround_from_above = '門(聞),鬥(闘),几(凧),冂(囘),𰃦(向),凡(風),齊,斉,𣎆,𦝠,戌,尺(尽),𦉰(罔),𠔼(同),咸(𱫢)'

component_examples_of_surround_from_left = '匚,匸'

component_examples_of_surround_from_upper_right = '勹,气,戈,弋,⺄'

component_examples_of_surround_from_below = '凵,𠒂,舁'

component_examples_of_enclosed = '丶(丼),口(哀),歹(夙),女(威),日(間),同(興),仌(𠕎),人(閃),㗊(𡈨),亻(亟),丰(𠙾),田(𡇍),巽(𨶷),矣(𨴱),䒑(𦉰),亡(罔),𠅇(罔),甬(𡇮),甬(𨴭),子(囝),昌(閶),巾(𡆫),布(𡇊),或(國),咸(𭍩),卄(𠕁),垔(𡇽),文(㓙),㐅(凶),方(圀),禾(囷),亦(𲘣),夬(䦑)'

component_examples_of_upper_right = '力(勉),匕(匙),㠯(𲏘),巽(選),彡(尨),𱼀(將),巛(巡),𠂉(臨),𧘇(𮞅),几(処),互(𧺳),卯(𨒖),田(𤔉),豆(逗),口(呉),羊(𨒫),甬(通),孔(𲏑),昌(𣮑),同(迵),布(𡲫),咸(𡯽),文(这),方(䢍),亦(迹),夢(𬩝),委(逶),戎(毧),戻(𩗭)'

component_examples_of_lower_right = '彡(修),其(旗),力(勝),㔾(卮),子(㞌),亟(𢉗),丁(庁),矢(疾),矣(𡱢),㠯(𢈂),立(𢨶),攵(䖍),冉(㾆),堇(厪),人(庂),巛(𠈉),夂(䖍),攵(䖍),尚(𤷛),衣(扆),支(庋),丰(𠨵),卯(㡻),共(𢈎),豆(痘),呉(虞),羊(庠),舞(𢋑),甬(𭙡),子(存),孔(𡰼),同(𢈉),巾(𡰯),布(𢇴),或(𢈿),侯(瘊),文(虔),秝(厤),止(𱤽),亦(𪊳),辛(屖),鬼(廆),夬(疦)'

component_examples_of_lower_left = '十(卂),口(句),口(命),𬺣(或),立(𣱠),丰(𫻩),田(𤰭),羊(氧),日(旬),廾(戒),方(㫄),止(𠣏),辡(𰁏)'

component_examples_of_upper_left = '土(敖),氵(柒),氵(染),叕(歠),日(猒),瓜(瓥),耳(聖),𱼀(然)'

component_examples_of_sandwiched_from_left_and_right = '矛(楙),𦰩(攤),丨(攸),亅(水),分(椕),合(搿),𡵉(微),丩(嘂),了(氶),𬼶(亟),人(臾),刂(辨)'

component_examples_of_sandwiched_from_above_and_below = '一(兴),頁(囂),日(卓),目(算),田(畫),⺣(稥),厶(窓),冖(亭),冖(夢),𠀎(𡨄),水(丞),厶(𣏋),巛(巠),豆(喜),䒑(善),𠔼(蒙),四(𧶠),鬼(褢)'

component_examples_of_inserted_from_left = '口(亟),㇇(丞)'

component_examples_of_inserted_from_right = '又(亟),品(區),矢(医),厶(鬼),王(匡),田(𭅗),巽(㔵),舞(𠥢),日(𫧍),巾(匝)'

position_prompt_en = f'position (left(e.g.{component_examples_of_left})/right(e.g.{component_examples_of_right})/above(e.g.{component_examples_of_above})/below(e.g.{component_examples_of_below})/surround-from-upper-left(e.g.{component_examples_of_surround_from_upper_left})/surround-from-lower-left(e.g.{component_examples_of_surround_from_lower_left})/full-surround(e.g.{component_examples_of_full_surround})/surround-from-above(e.g.{component_examples_of_surround_from_above})/surround-from-left(e.g.{component_examples_of_surround_from_left})/surround-from-upper-right(e.g.{component_examples_of_surround_from_upper_right})/surround-from-below(e.g.{component_examples_of_surround_from_below})/upper-left(e.g.{component_examples_of_upper_left})/upper-right(e.g.{component_examples_of_upper_right})/lower-right(e.g.{component_examples_of_lower_right})/lower-left(e.g.{component_examples_of_lower_left})/enclosed(e.g.{component_examples_of_enclosed})/sandwiched-from-left-and-right(e.g.{component_examples_of_sandwiched_from_left_and_right})/sandwiched-from-above-and-below(e.g.{component_examples_of_sandwiched_from_above_and_below})/inserted-from-left(e.g.{component_examples_of_inserted_from_left})/inserted-from-right(e.g.{component_examples_of_inserted_from_right}))'

position_prompt_en_s = f'position (left(e.g.{component_examples_of_left})/right(e.g.{component_examples_of_right})/above(e.g.{component_examples_of_above})/below(e.g.{component_examples_of_below})/surround-from-upper-left(e.g.{component_examples_of_surround_from_upper_left})/surround-from-lower-left(e.g.{component_examples_of_surround_from_lower_left})/full-surround(e.g.{component_examples_of_full_surround})/surround-from-above(e.g.{component_examples_of_surround_from_above})/surround-from-left(e.g.{component_examples_of_surround_from_left})/surround-from-upper-right(e.g.{component_examples_of_surround_from_upper_right})/surround-from-below(e.g.{component_examples_of_surround_from_below})/upper-left(e.g.{component_examples_of_upper_left})/upper-right(e.g.{component_examples_of_upper_right})/lower-right(e.g.{component_examples_of_lower_right})/lower-left(e.g.{component_examples_of_lower_left})/enclosed(e.g.{component_examples_of_enclosed})'

position_prompt_ja = f'相対位置(left(偏。例：{component_examples_of_left})/right(旁。例：{component_examples_of_right})/above(冠。例：{component_examples_of_above})/below(脚。例：{component_examples_of_below})/surround-from-upper-left(垂。例：{component_examples_of_surround_from_upper_left})/surround-from-lower-left(繞。例：{component_examples_of_surround_from_lower_left})/full-surround(構1。例：{component_examples_of_full_surround})/surround-from-above(構2。例：{component_examples_of_surround_from_above})/surround-from-left(例：{component_examples_of_surround_from_left})/surround-from-upper-right(例：{component_examples_of_surround_from_upper_right})/surround-from-below(例：{component_examples_of_surround_from_below})/upper-left(左上。例：{component_examples_of_upper_left})/upper-right(右上。例：{component_examples_of_upper_right})/lower-right(右下。例：{component_examples_of_lower_right})/lower-left(左下。例：{component_examples_of_lower_left})/enclosed(構えの中の部品。例：{component_examples_of_enclosed})/sandwiched-from-left-and-right(左右の間。例：{component_examples_of_sandwiched_from_left_and_right})/sandwiched-from-above-and-below(上下の間。例：{component_examples_of_sandwiched_from_above_and_below})/inserted-from-left(例：{component_examples_of_inserted_from_left})/inserted-from-right(例：{component_examples_of_inserted_from_right}))'

position_prompt_L2R_ja = f'相対位置(left(偏。例：{component_examples_of_left})/right(旁。例：{component_examples_of_right})/surround-from-upper-left(垂。例：{component_examples_of_surround_from_upper_left})/surround-from-lower-left(繞。例：{component_examples_of_surround_from_lower_left})/full-surround(構1。例：{component_examples_of_full_surround})/surround-from-above(構2。例：{component_examples_of_surround_from_above})/surround-from-left(例：{component_examples_of_surround_from_left})/surround-from-upper-right(例：{component_examples_of_surround_from_upper_right})/surround-from-below(例：{component_examples_of_surround_from_below})/upper-left(左上。例：{component_examples_of_upper_left})/upper-right(右上。例：{component_examples_of_upper_right})/lower-right(右下。例：{component_examples_of_lower_right})/lower-left(左下。例：{component_examples_of_lower_left})/enclosed(構えの中の部品。例：{component_examples_of_enclosed})/sandwiched-from-left-and-right(左右の間。例：{component_examples_of_sandwiched_from_left_and_right}))'

position_prompt_A2B_ja = f'相対位置(above(冠。例：{component_examples_of_above})/below(脚。例：{component_examples_of_below})/surround-from-upper-left(垂。例：{component_examples_of_surround_from_upper_left})/surround-from-lower-left(繞。例：{component_examples_of_surround_from_lower_left})/full-surround(構1。例：{component_examples_of_full_surround})/surround-from-above(構2。例：{component_examples_of_surround_from_above})/surround-from-left(例：{component_examples_of_surround_from_left})/surround-from-upper-right(例：{component_examples_of_surround_from_upper_right})/surround-from-below(例：{component_examples_of_surround_from_below})/upper-left(左上。例：{component_examples_of_upper_left})/upper-right(右上。例：{component_examples_of_upper_right})/lower-right(右下。例：{component_examples_of_lower_right})/lower-left(左下。例：{component_examples_of_lower_left})/enclosed(構えの中の部品。例：{component_examples_of_enclosed})/sandwiched-from-above-and-below(上下の間。例：{component_examples_of_sandwiched_from_above_and_below}))'

position_prompt_zh_TW = f'相對位置(left(偏。例如:{component_examples_of_left})/right(旁。例如:{component_examples_of_right})/above(頭。例如:{component_examples_of_above})/below(底。例如:{component_examples_of_below})/surround-from-upper-left(例如:{component_examples_of_surround_from_upper_left})/surround-from-lower-left(例如:{component_examples_of_surround_from_lower_left})/full-surround(同時夾著文字上下或左右。例如:{component_examples_of_full_surround})/surround-from-above(例如:{component_examples_of_surround_from_above})/surround-from-left(例如:{component_examples_of_surround_from_left})/surround-from-upper-right(例如:{component_examples_of_surround_from_upper_right})/surround-from-below(例如:{component_examples_of_surround_from_below})/upper-left(左上。例如:{component_examples_of_upper_left})/upper-right(右上。例如:{component_examples_of_upper_right})/lower-left(左下。例如:{component_examples_of_lower_left})/lower-right(右下。例如:{component_examples_of_lower_right})/enclosed(内部的部件。例如:{component_examples_of_enclosed})/sandwiched-from-left-and-right(夾在左右之間的部件。例如:{component_examples_of_sandwiched_from_left_and_right})/sandwiched-from-above-and-below(夾在上下之間的部件。例如:{component_examples_of_sandwiched_from_above_and_below})/inserted-from-left(例如：{component_examples_of_inserted_from_left})/inserted-from-right(例如：{component_examples_of_inserted_from_right}))'

position_prompt_en_p = 'position (left/right/above/below/surround-from-upper-left/surround-from-lower-left/full-surround/surround-from-above/surround-from-left/surround-from-upper-right/surround-from-below/surround-from-right/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/sandwiched-from-left-and-right/sandwiched-from-above-and-below/inserted-from-left/inserted-from-right)'

position_prompt_ja_p = '相対位置(left(偏)/right(旁)/above(冠)/below(脚)/surround-from-upper-left(垂)/surround-from-lower-left(繞)/full-surround(箱構)/surround-from-above(上構)/surround-from-left/surround-from-upper-right/surround-from-below/upper-left(左上)/upper-right(右上)/lower-left(左下)/lower-right(右下)/enclosed(構の中)/sandwiched-from-left-and-right(左右の間)/sandwiched-from-above-and-below(上下の間)/inserted-from-left/inserted-from-right)'

position_prompt_ja_p_s = '相対位置(left(偏)/right(旁)/above(冠)/below(脚)/upper-left(左上)/upper-right(右上)/lower-left(左下)/lower-right(右下)/surround(構、垂、繞など)/enclosed(構の中)/sandwiched(挟まれた)/inserted-from-left(左から差し込む)/inserted-from-right(右から差し込む))'

prompt_E6p = f'''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_E5 = f'''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en}
'''

prompt_E5s = f'''<image> Locate every component of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_s}
'''

prompt_cE = f'''<image> Locate every component of the classical Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en}
'''

prompt_C6p = f'''<image> Locate every component of the Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_C = f'''<image> Locate every component of the Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en}
'''

prompt_cC = f'''<image> Locate every component of the classical Hanzi.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en}
'''

prompt_J6p = f'''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en_p}
'''

prompt_J = f'''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	{position_prompt_en}
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
X0	Y0	X1	Y1	部品	{position_prompt_ja}
'''

prompt_L2R_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_L2R_ja}
'''

prompt_A2B_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_A2B_ja}
'''

prompt_ja6p = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_ja_p}
'''

prompt_ja6p_s = f'''画像にある漢字を構成する全ての部品を見つけてください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：
X0	Y0	X1	Y1	部品	{position_prompt_ja_p_s}
'''

prompt_zh_TW = f'''找出圖像中漢字的所有組成部分。
以 TSV 格式輸出每個找到的組成部分及其對應的直角座標。
範例：
X0	Y0	X1	Y1	組成部分	{position_prompt_zh_TW}
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

# prompt = prompt_ja6p
# retry_prompt_default = prompt_ja
# retry_prompt_L2R = prompt_L2R_ja
# retry_prompt_A2B = prompt_A2B_ja
# retry_prompt2 = prompt_E5
# retry_prompt3 = prompt_zh_TW

prompt = prompt_ja6p_s
retry_prompt_default = prompt_ja6p
retry_prompt_L2R = prompt_L2R_ja
retry_prompt_A2B = prompt_A2B_ja
retry_prompt2 = prompt_ja
retry_prompt3 = prompt_zh_TW

# prompt = prompt_E6p
# retry_prompt_default = prompt_E5
# retry_prompt_L2R = prompt_L2R_ja
# retry_prompt_A2B = prompt_A2B_ja
# retry_prompt2 = prompt_ja
# retry_prompt3 = prompt_zh_TW

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

# component_prompt = '''<image> Run OCR for component (or structure) of the Kanji and output the result.
# '''
component_prompt = '''<image> 漢字部品用 OCR を実行し、結果を文字もしくは IDS で返してください。
'''

# character_prompt = '''<image> Run Kanji (or traditional Hanzi) OCR and output the result.
# '''
character_prompt = '''<image> 漢字用 OCR を実行し、結果の文字を返してください。
'''

def run_VLM (images, prompt):
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images = len(images)
    )
    
    # Generate output
    response = generate(model, processor, formatted_prompt, images,
                        max_tokens = 1024, temperature=0.0,
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
                     ( Component_Position[1] == 'full' ) or
                     ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'above' ) ):
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
                elif ( ( Component_Position[1] == 'right' ) and
                       ( X1[0] <= X2[1] ) ):
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

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
                       ( Component_Position[1] == 'lower-right' ) or
                       ( Component_Position[1] == 'below' ) ):
                    return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'enclosed':
                if Component_Position[1] == 'surround-from-lower-left':
                    return f'⿺{Component_Text[1]}{Component_Text[0]}'

            case 'surround':
                if Component_Position[1] == 'middle':
                    if ( ( Component_Text[0] == '⻌' ) or
                         ( Component_Text[0] == '廴' ) ):
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'
                    elif ( Component_Text[0] == '門' ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
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
                    elif ( ( Component_Text[0] == '戈' ) or
                           ( Component_Text[0] == '咸' ) ):
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
                    elif ( Component_Text[0] == '食' ):
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'
                elif Component_Position[1] == 'right':
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'
                elif Component_Position[1] == 'surround-from-right':
                    if ( X2[0] <= X1[1] ):
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-right' ) or
                     ( Component_Position[1] == 'right' ) ):
                    if Component_Text[0] == '匚':
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'
                    elif ( Component_Text[0] == '囗' ):
                        return f'⿴{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '几' ) or
                           ( Component_Text[0] == '門' ) ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '勹':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                elif ( Component_Position[1] == 'below' ):
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'
                elif ( Component_Text[0] == '耂' ):
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-lower-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'upper-right' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    if ( ( Component_Text[0] == '几' ) or
                         ( Component_Text[0] == '門' ) ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif ( Component_Text[0] == '囗' ):
                        return f'⿴{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿼{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-left' ) or
                     ( Component_Position[1] == 'below' ) ):
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

                elif ( ( Component_Position[1] == 'below' ) and
                       ( Component_Position[2] == 'below' ) ):
                    if ( ( Component_Text[1] == Component_Text[2] ) or
                         ( ( Component_Text[1] == '立' ) and
                           ( Component_Text[2] == '日' ) ) ):
                        return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                    elif ( ( Component_Text[0] == '日' ) and
                           ( Component_Text[1] == '共' ) ):
                        return f'⿱⿱{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
                    elif ( ( Component_Text[0] == '⺈' ) and
                           ( Component_Text[1] == '目' ) ):
                        return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( ( Component_Position[1] == 'enclosed' ) or
                         ( Component_Position[1] == 'sandwiched-from-above-and-below' ) )
                       and
                       ( Component_Position[2] == 'below' ) ):
                    if ( ( ( Component_Text[0] == '立' ) and
                           ( ( Component_Text[1] == '日' ) or
                             ( Component_Text[1] == '曰' ) ) )
                         or
                         ( ( Component_Text[0] == '中' ) and
                           ( Component_Text[1] == '一' ) )
                         or
                         ( ( Component_Text[0] == '十' ) and
                           ( Component_Text[1] == '目' ) ) ):
                        return f'⿱⿱{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
                    elif ( ( ( Component_Text[1] == '目' ) and
                             ( Component_Text[2] == '廾' ) )
                           or
                           ( ( Component_Text[1] == '田' ) and
                             ( Component_Text[2] == '共' ) )
                           or
                           ( Component_Text[1] == Component_Text[2] ) ):
                        return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'lower-left' ) and
                       ( Component_Position[2] == 'lower-right' ) ):
                    return f'⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'left' ) and
                       ( Component_Position[2] == 'right' ) ):
                    if ( X2[0] < X1[2] ):
                        return f'⿰⿱{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'surround-from-above' ) and
                       ( ( Component_Position[2] == 'middle' ) or
                         ( Component_Position[2] == 'enclosed' ) ) ):
                    if Component_Text[1] == '囗':
                        return f'⿱{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿱{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'

                elif ( ( Component_Position[1] == 'surround-from-upper-right' ) and
                       ( ( Component_Position[2] == 'middle' ) or
                         ( Component_Position[2] == 'enclosed' ) ) ):
                    return f'⿱{Component_Text[0]}⿹{Component_Text[1]}{Component_Text[2]}'

            case 'upper-left':
                if ( ( Component_Position[1] == 'upper-right' ) and
                     ( ( Component_Position[2] == 'lower' ) or
                       ( Component_Position[2] == 'below' ) or
                       ( ( Component_Position[2] == 'lower-right' ) and
                         ( X1[2] < X2[0] ) )
                      ) ):
                    if ( ( Component_Text[0] == '耳' ) and
                         ( Component_Text[1] == '口' ) ):
                        return f'⿽⿱{Component_Text[1]}{Component_Text[2]}{Component_Text[0]}'
                    else:
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
                elif ( ( Component_Position[1] == 'above' ) and
                       ( Component_Position[2] == 'below' ) ):
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

            case 'lower-left':
                if ( ( Component_Position[1] == 'upper-right' ) and
                     ( Component_Position[2] == 'below' ) ):
                    return f'⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'

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
                elif ( ( Component_Position[1] == 'below' ) and
                       ( Component_Position[2] == 'lower-left' ) and
                       ( Component_Position[3] == 'lower-right' ) ):
                    return f'⿳{Component_Text[0]}{Component_Text[1]}⿰{Component_Text[2]}{Component_Text[3]}'
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

        print (f'component prompt = "{component_prompt}".')
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
    image_width, image_height = im.size

    cx1 = min (X1[0], X1[1])
    cy1 = min (Y1[0], Y1[1])
    cx2 = image_width - 1 # max (X2[0], X2[1])
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

def merge_enclosed_vertical3 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
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

def merge_enclosed_vertical4 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]

    cx1 = min (X1[1], X1[2], X1[3], X1[4])
    cy1 = min (Y1[1], Y1[2], Y1[3], Y1[4])
    cx2 = max (X2[1], X2[2], X2[3], X2[4])
    cy2 = max (Y2[1], Y2[2], Y2[3], Y2[4])
    orig_comp2 = Component_Text[1]
    orig_comp3 = Component_Text[2]
    orig_comp4 = Component_Text[3]
    orig_comp5 = Component_Text[4]
    del X1[2:5]
    del Y1[2:5]
    del X2[2:5]
    del Y2[2:5]
    del Component_Text[2:5]
    del Component_Position[2:5]
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

        comp5_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp5.png'
        if (os.path.isfile(comp5_image_file_name)):
            os.remove(comp5_image_file_name)

    Component_Text[1] = orig_comp2
    print (f'New {orig_comp2},{orig_comp3},{orig_comp4} -> {Component_Text[1]}')

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_below_vertical3 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH):
    im = Image.open(image_file)
    basename = os.path.splitext(os.path.basename(image_file))[0]
    image_width, image_height = im.size

    cx1 = 0 # min (X1[1], X1[2], X1[3])
    cy1 = min (Y1[1], Y1[2], Y1[3])
    cx2 = image_width - 1 # max (X2[1], X2[2], X2[3])
    cy2 = image_height - 1 # max (Y2[1], Y2[2], Y2[3])
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
        for line_match in re.findall('([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([^()（） \t\n\r]+?)(\(.+\))?\s+([a-z-上下左右]+)\S*\n?', response):
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

                elif ( ( Component_Position[1] == 'surround-from-above' ) and
                       ( Component_Position[2] == 'enclosed' ) ):
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
                    if ( ( ( Component_Position[1] != 'above' ) and
                           ( Component_Position[1] != 'upper-right' ) )
                         or
                         ( ( Component_Position[2] != 'below' ) and
                           ( Component_Position[2] != 'lower-right' ) ) ):
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
                        if ( ( Component_Text[0] == '勹' ) or
                             ( Component_Text[0] == '气' ) or
                             ( Component_Text[0] == '戈' ) or
                             ( Component_Text[0] == '弋' ) or
                             ( Component_Text[0] == '⺄' ) ):
                            Component_Position[0] = 'surround-from-upper-right'
                        else:
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

            case 'above':
                if Component_Position[1] == 'surround-from-upper-left':
                    if Component_Position[2] == 'below':
                        X1, Y1, X2, Y2, Component_Text, Component_Position = merge_above_and_below (image_file,
                                                                                                    X1, Y1,
                                                                                                    X2, Y2,
                                                                                                    Component_Text,
                                                                                                    Component_Position,
                                                                                                    TSV_OUTPUT_PATH)
                        if ( Y2[0] <= Y1[1] ):
                            Component_Position[0] = 'above'
                        else:
                            Component_Position[0] = 'surround-from-upper-left'

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
                X1, Y1, X2, Y2, Component_Text, Component_Position = merge_enclosed_vertical3 (image_file,
                                                                                               X1, Y1,
                                                                                               X2, Y2,
                                                                                               Component_Text,
                                                                                               Component_Position,
                                                                                               TSV_OUTPUT_PATH)

        elif ( ( Component_Position[0] == 'surround-from-lower-left' ) and
               ( Component_Position[1] == 'above' ) and
               ( Component_Position[2] == 'sandwiched-from-above-and-below' ) and
               ( Component_Position[3] == 'sandwiched-from-above-and-below' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_enclosed_vertical3 (image_file,
                                                                                           X1, Y1,
                                                                                           X2, Y2,
                                                                                           Component_Text,
                                                                                           Component_Position,
                                                                                           TSV_OUTPUT_PATH)

        elif ( ( Component_Position[0] == 'upper-left' ) and
               ( Component_Position[1] == 'upper-right' ) and
               ( Component_Position[2] == 'lower-left' ) and
               ( Component_Position[3] == 'lower-right' ) and
               ( ( max ( X2[0] - X1[0],
                         X2[1] - X1[1],
                         X2[2] - X1[2],
                         X2[3] - X1[3] ) > image_width / 5 ) or
                 ( max ( Y2[0] - Y1[0],
                         Y2[1] - Y1[1],
                         Y2[2] - Y1[2],
                         Y2[3] - Y1[3] ) > image_height / 5 ) ) ):
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

        elif ( ( Component_Position[0] == 'above' ) and
               ( Component_Position[1] == 'enclosed' ) and
               ( Component_Position[2] == 'sandwiched' ) and
               ( Component_Position[3] == 'below' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_below_vertical3 (image_file,
                                                                                        X1, Y1,
                                                                                        X2, Y2,
                                                                                        Component_Text,
                                                                                        Component_Position,
                                                                                        TSV_OUTPUT_PATH)
            Component_Position[0] = 'above'

        elif ( ( Component_Position[0] == 'above' ) and
               ( Component_Position[1] == 'surround-from-upper-left' ) and
               ( Component_Position[2] == 'middle' ) and
               ( Component_Position[3] == 'below' ) and
               ( ( Y1[0] == Y1[1] ) and
                 ( Y2[0] == Y2[1] ) ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_above_and_below (image_file,
                                                                                        X1, Y1,
                                                                                        X2, Y2,
                                                                                        Component_Text,
                                                                                        Component_Position,
                                                                                        TSV_OUTPUT_PATH)
            if ( Y2[0] <= Y1[1] ):
                Component_Position[0] = 'above'
            else:
                Component_Position[0] = 'surround-from-upper-left'

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

        elif ( ( Component_Position[0] == 'surround-from-left' ) and
               ( Component_Position[1] == 'enclosed' ) and
               ( Component_Position[2] == 'above' ) and
               ( Component_Position[3] == 'middle' ) and
               ( Component_Position[4] == 'below' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_enclosed_vertical4 (image_file,
                                                                                           X1, Y1,
                                                                                           X2, Y2,
                                                                                           Component_Text,
                                                                                           Component_Position,
                                                                                           TSV_OUTPUT_PATH)

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

    im = Image.open(image_file)
    image_width, image_height = im.size

    images = [ image_file ]
    print (f'character prompt = "{character_prompt}".')
    char_response = run_VLM (images, character_prompt)
    print (f'character = "{char_response}".')
    char_res = re.match ('^(.)[.,。、．]$', char_response)
    if char_res:
        char_response = char_res.group(1)
        print (f'character = "{char_response}".')

    X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                  prompt,
                                                                                  TSV_OUTPUT_PATH,
                                                                                  OUTPUT_PATH)
    retry_flag = False
    #retry_prompt = retry_prompt_default
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

        elif ( ( max ( X2[0] - X1[0], X2[1] - X1[1] ) < image_width / 5 ) or
               ( max ( Y2[0] - Y1[0], Y2[1] - Y1[1] ) < image_height / 5 ) ):
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
                             ( Component_Text[1] == '⻖' ) or
                             ( Component_Text[1] == '僉' ) or
                             ( Component_Text[1] == '攸' ) or
                             ( Component_Text[1] == '冏' ) or
                             ( Component_Text[1] == '乚' ) or
                             ( Component_Text[1] == '隹' ) or
                             ( Component_Text[1] == '夊' ) or
                             ( Component_Text[1] == '夂' ) or
                             ( Component_Text[1] == '堇' ) or
                             ( Component_Text[1] == '侯' ) ):
                            retry_flag = True
                            retry_prompt = retry_prompt_L2R
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
                         ( Component_Text[0] == '殳' ) or
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
                        retry_prompt = retry_prompt_default

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
             ( Component_Text[1] == '丷' ) and
             ( Component_Text[2] == '丷' ) ):
            retry_flag = True
            retry_prompt = retry_prompt_A2B

    elif len(Component_Text) > 7:
        # print ('Retry with simpler prompt')
        # X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
        #                                                                               simpler_prompt,
        #                                                                               TSV_OUTPUT_PATH,
        #                                                                               OUTPUT_PATH)
        retry_flag = True
        if ( Component_Position[0] == 'above' ):
            retry_prompt = retry_prompt_A2B
        elif ( Component_Position[0] == 'left' ):
            retry_prompt = retry_prompt_L2R
        else:
            retry_prompt = retry_prompt_default

    if retry_flag:
        print ('Retry')
        X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                      retry_prompt,
                                                                                      TSV_OUTPUT_PATH,
                                                                                      OUTPUT_PATH)
        if ( ( len(X1) >= 2 ) and
             ( ( max ( X2[0] - X1[0], X2[1] - X1[1] ) < image_width / 5 ) or
               ( max ( Y2[0] - Y1[0], Y2[1] - Y1[1] ) < image_height / 5 ) ) ):
            print ('Retry(2)')
            X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                          retry_prompt2,
                                                                                          TSV_OUTPUT_PATH,
                                                                                          OUTPUT_PATH)

    if ( ( len(char_response) == 1 ) and
         ( any ( comp == char_response for comp in Component_Text ) ) and
         ( ( len(Component_Text) < 2 ) or
           ( Component_Text[0] != '囗' ) ) ):
        with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
            print(char_response, file=full_destfile)
        return char_response

    
    ids = detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
    if ids:
        with open(f'{OUTPUT_PATH}/{basename}_ids.txt',
                  'w', encoding = 'utf-8') as ids_destfile:
            print(ids, file=ids_destfile)
        return ids
    else:
        print ('Retry(3)')
        X1, Y1, X2, Y2, Component_Text, Component_Position = run_OCR_for_glyph_image (image_file_name,
                                                                                      retry_prompt3,
                                                                                      TSV_OUTPUT_PATH,
                                                                                      OUTPUT_PATH)
        if ( ( len(char_response) == 1 ) and
             ( any ( comp == char_response for comp in Component_Text ) ) ):
            with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                print(char_response, file=full_destfile)
            return char_response

        elif ( ( len(X1) >= 2 ) and
               ( ( max ( X2[0] - X1[0], X2[1] - X1[1] ) < image_width / 4 ) or
                 ( max ( Y2[0] - Y1[0], Y2[1] - Y1[1] ) < image_height / 4 ) ) and
               ( len(char_response) == 1 ) ):
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
            else:
                if ( len(char_response) == 1 ):
                    with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                        print(char_response, file=full_destfile)
                    return char_response

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
    if args.git:
        subprocess.run("git pull", shell=True)
        subprocess.run(f"git add {TSV_OUTPUT_PATH}/*.txt {TSV_OUTPUT_PATH}/*.tsv", shell=True)
        subprocess.run("git pull", shell=True)
        subprocess.run(f"git commit {TSV_OUTPUT_PATH}/*.txt {TSV_OUTPUT_PATH}/*.tsv -m 'New files.'", shell=True)
        subprocess.run("git pull", shell=True)
        subprocess.run("git push origin main", shell=True)

    print (f'prompt CID = {IPFS_CID}')
    print (f'{image_file_name} : {ids}\n')
