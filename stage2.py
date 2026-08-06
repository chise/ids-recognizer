from PIL import Image
import os
import re
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import vlm_ocr


structure_examples_L2R_ja = '''
例:「劉」
----------------------------------------------------------------
10	10	980	980	劉	root	---
10	10	490	980	𨥫	left	劉
490	10	980	980	刂	right	劉
10	10	490	490	卯	above	𨥫
10	490	490	980	金	below	𨥫
----------------------------------------------------------------

例:「激」
------------------------------------------------------------
0	0	999	999	激	root	---
0	0	300	999	氵	left	激
300	0	999	999	敫	right	激
------------------------------------------------------------

例:「惰」
------------------------------------------------------------
0	0	999	999	惰	root	---
0	0	499	999	忄	left	惰
500	0	999	999	𲺋	right	惰
500	0	999	499	左	above	𲺋
500	500	999	999	月	below	𲺋
------------------------------------------------------------

例:「證」
------------------------------------------------------------
0	0	999	999	證	root	---
0	0	499	999	言	left	證
500	0	999	999	登	right	證
500	0	999	499	癶	above	登
500	500	999	999	豆	below	登
------------------------------------------------------------

例:「撤」
------------------------------------------------------------
0	0	999	999	撤	root	---
0	0	300	999	扌	left	撤
300	0	999	999	𰕎	right	撤
300	0	500	999	育	left	𰕎
500	0	999	999	攵	right	𰕎
------------------------------------------------------------

例:「難」
------------------------------------------------------------
0	0	999	999	難	root	---
0	0	499	999	𦰩	left	難
500	0	999	999	隹	right	難
------------------------------------------------------------

例:「鑚」
------------------------------------------------------------
0	0	999	999	鑚	root	---
0	0	499	999	金	left	鑚
500	0	999	999	賛	right	鑚
------------------------------------------------------------

例:「僕」
------------------------------------------------------------
0	0	999	999	僕	root	---
0	0	499	999	亻	left	僕
500	0	999	999	菐	right	僕
------------------------------------------------------------

例:「獸」
----------------------------------------------------------------
10	10	980	980	獸	root	---
10	10	490	980	嘼	left	獸
490	10	980	980	犬	right	獸
10	10	490	600	𲕟	above	嘼
10	600	490	980	口	below	嘼
----------------------------------------------------------------

例:「牖」
----------------------------------------------------------------------------
10	10	980	980	牖	root	---
10	10	490	980	片	left	牖
490	10	980	980	〓	right	牖
490	10	980	980	戸	surround-from-upper-left	〓
600	400	980	980	甫	inserted-from-lower-right	〓
----------------------------------------------------------------------------

例:「號」
----------------------------------------------------------------------------
10	10	980	980	號	root	---
10	10	490	980	号	left	號
490	10	980	980	虎	right	號
490	10	980	980	虍	surround-from-upper-left	虎
600	400	980	980	儿	inserted-from-lower-right	虎
----------------------------------------------------------------------------

例:「棣」
------------------------------------------------------------
0	0	999	999	棣	root	---
0	0	499	999	木	left	棣
500	0	999	999	隶	right	棣
------------------------------------------------------------

例:「輟」
------------------------------------------------------------
0	0	999	999	輟	root	---
0	0	499	999	車	left	輟
500	0	999	999	叕	right	輟
------------------------------------------------------------

例:「殿」
--------------------------------------------------------------------------
0	0	999	999	殿	root	---
0	0	499	999	𡱒	left	殿
500	0	999	999	殳	right	殿
0	0	499	999	尸	surround-from-upper-left	𡱒
100	100	499	999	共	inserted-from-lower-right	𡱒
500	0	499	499	几	above	殳
500	500	999	999	又	below	殳
--------------------------------------------------------------------------

例:「瀆」
--------------------------------------------------------------------------
0	0	999	999	瀆	root	---
0	0	400	999	氵	left	瀆
400	0	999	999	𧶠	right	瀆
400	0	499	299	士	above	𧶠
400	300	999	599	四	middle	𧶠
400	600	999	999	貝	below	𧶠
--------------------------------------------------------------------------

例:「膠」
----------------------------------------------------------------
10	10	980	980	膠	root	---
10	10	400	980	月	left	膠
400	10	980	980	翏	right	膠
400	10	980	490	羽	above	翏
400	490	980	980	㐱	below	翏
----------------------------------------------------------------

例:「荆」
----------------------------------------------------------------
10	10	980	980	荆	root	---
10	10	490	980	茾	left	荆
490	10	980	980	刂	right	荆
10	10	490	490	艹	above	茾
10	490	490	980	开	below	茾
----------------------------------------------------------------

例:「滕」
--------------------------------------------------------------------------
10	10	980	980	滕	root	---
10	10	980	980	𰮤	surround-from-upper-left	滕
490	490	980	980	氺	inserted-from-lower-right	滕
10	10	490	980	月	left	𰮤
490	10	490	490	龹	right	𰮤
--------------------------------------------------------------------------

例:「旅」
--------------------------------------------------------------------------
10	10	980	980	旅	root	---
10	10	980	950	𭤨	surround-from-upper-left	旅
490	490	980	980	𧘇	inserted-from-lower-right	旅
--------------------------------------------------------------------------

例:「修」
--------------------------------------------------------------------------
10	10	980	980	修	root	---
10	10	980	980	攸	surround-from-upper-left	修
490	490	980	980	彡	inserted-from-lower-right	修
10	10	490	980	〓	left	攸
490	10	490	490	攵	right-above	攸
10	10	250	980	亻	right-above	〓
250	10	490	980	丨	right-above	〓
--------------------------------------------------------------------------

例:「獄」
--------------------------------------------------------------------------
10	10	980	980	獄	root	---
10	10	980	980	犾	full-surround	獄
330	0	659	980	言	full-enclosed	獄
10	10	329	980	犭	left	犾
660	10	980	980	犬	right	犾
--------------------------------------------------------------------------
'''

structure_examples_A2B_ja = '''
例:「亭」
----------------------------------------------------------------
10	10	940	940	亭	root	---
10	10	940	570	〓	above	亭
10	570	940	940	丁	below	亭
10	10	940	290	亠	above	〓
10	290	940	550	口	enclosed	〓
10	550	940	570	冖	below	〓
----------------------------------------------------------------

例:「具」
----------------------------------------------------------------
10	10	940	940	具	root	---
10	10	940	570	目	above	具
10	570	940	940	𬺢	below	具
----------------------------------------------------------------

例:「巽」
----------------------------------------------------------------
10	10	980	980	巽	root	---
10	10	980	490	〓	above	巽
10	490	980	980	共	below	巽
10	10	490	490	巳	left	〓
490	10	980	490	巳	right	〓
----------------------------------------------------------------

例:「卓」
--------------------------------------------------------------------------
10	10	980	980	卓	root	---
10	10	980	200	⺊	above	卓
10	200	980	980	早	below	卓
--------------------------------------------------------------------------

例:「卒」
--------------------------------------------------------------------------
10	10	980	980	卒	root	---
10	10	980	490	𠅃	above	卒
10	490	980	980	十	below	卒
--------------------------------------------------------------------------

例:「嘉」
--------------------------------------------------------------------------
10	10	980	980	嘉	root	---
10	10	980	680	壴	above	嘉
10	680	980	980	加	below	嘉
--------------------------------------------------------------------------

例:「壽」
--------------------------------------------------------------------------
10	10	980	980	壽	root	---
10	10	980	400	士	above	壽
10	400	980	700	〓	middle	壽
10	700	980	980	吋	below	壽
10	400	980	480	乛	above	〓
10	480	980	620	工	middle	〓
10	620	980	700	一	below	〓
--------------------------------------------------------------------------

例:「駕」
--------------------------------------------------------------------------
10	10	980	980	駕	root	---
10	10	980	400	加	above	駕
10	400	980	700	馬	middle	駕
--------------------------------------------------------------------------

例:「黃」
--------------------------------------------------------------------------
10	10	980	980	黃	root	---
10	10	980	980	〓	full-surround	黃
10	500	980	880	由	full-enclosed	黃
10	10	980	400	廿	above	〓
10	400	980	980	〓	below	〓
10	400	980	500	一	below	〓
10	880	980	980	八	below	〓
--------------------------------------------------------------------------

例:「夢」
----------------------------------------------------------------
10	10	940	940	夢	root	---
10	10	940	570	〓	above	夢
10	570	940	940	夕	below	夢
10	10	940	490	𦭝	above	〓
10	490	940	570	冖	below	〓
----------------------------------------------------------------

例:「奡」
--------------------------------------------------------------------------
10	10	980	980	奡	root	---
10	10	980	700	𦣻	above	奡
10	700	980	980	夰	below	奡
--------------------------------------------------------------------------

例:「寒」
--------------------------------------------------------------------------
10	10	980	980	寒	root	---
10	10	980	700	𡨄	above	寒
10	700	980	980	冫	below	寒
--------------------------------------------------------------------------

例:「專」
--------------------------------------------------------------------------
10	10	980	980	專	root	---
10	10	980	700	叀	above	專
10	700	980	980	寸	below	專
--------------------------------------------------------------------------

例:「掌」
--------------------------------------------------------------------------
10	10	980	980	掌	root	---
10	10	980	600	𫩠	above	掌
10	600	980	980	手	below	掌
--------------------------------------------------------------------------

例:「磬」
--------------------------------------------------------------------------
10	10	980	980	磬	root	---
10	10	980	600	殸	above	磬
10	600	980	980	石	below	磬
--------------------------------------------------------------------------

例:「榮」
--------------------------------------------------------------------------
10	10	980	980	榮	root	---
10	10	980	600	𤇾	above	榮
10	600	980	980	木	below	榮
10	10	980	500	炏	above	𤇾
10	500	980	600	冖	below	𤇾
10	10	490	500	火	left	炏
490	10	490	500	火	right	炏
--------------------------------------------------------------------------

例:「燕」
--------------------------------------------------------------------------
10	10	980	980	燕	root	---
10	10	980	700	〓	above	燕
10	700	980	980	灬	below	燕
10	10	980	300	廿	above	〓
10	280	980	700	北	full-surround	〓
110	300	880	680	口	full-enclosed	〓
--------------------------------------------------------------------------

例:「皐」
--------------------------------------------------------------------------
10	10	980	980	皐	root	---
10	10	980	490	白	above	皐
10	490	980	980	𠦂	below	皐
--------------------------------------------------------------------------

例:「丞」
----------------------------------------------------------------
10	10	940	940	丞	root	---
10	10	940	700	氶	above	丞
10	700	940	940	一	below	丞
----------------------------------------------------------------

例:「互」
----------------------------------------------------------------
10	10	980	980	互	root	---
10	10	980	200	一	above	互
10	200	980	980	彑	below	互
----------------------------------------------------------------

例:「靑」
--------------------------------------------------------------------------
10	10	980	980	靑	root	---
10	10	980	490	龶	above	靑
10	490	980	980	円	below	靑
--------------------------------------------------------------------------

例:「責」
--------------------------------------------------------------------------
10	10	980	980	靑	root	---
10	10	980	490	龶	above	責
10	490	980	980	貝	below	責
--------------------------------------------------------------------------

例:「審」
--------------------------------------------------------------------------
10	10	980	980	靑	root	---
10	10	980	490	宷	above	審
10	490	980	980	田	below	審
--------------------------------------------------------------------------

例:「菲」
--------------------------------------------------------------------------
10	10	980	980	菲	root	---
10	10	980	200	艹	above	菲
10	200	980	980	非	below	菲
--------------------------------------------------------------------------

例:「賔」
--------------------------------------------------------------------------
10	10	980	980	賔	root	---
10	10	980	200	宀	above	賔
10	200	980	980	𮙸	below	賔
--------------------------------------------------------------------------

例:「犂」
--------------------------------------------------------------------------
10	10	980	980	犂	root	---
10	10	980	490	𥝢	above	犂
10	490	980	980	牛	below	犂
10	10	490	490	禾	left	𥝢
490	10	980	490	𠚣	right	𥝢
--------------------------------------------------------------------------

例:「盪」
--------------------------------------------------------------------------
10	10	980	980	盪	root	---
10	10	980	490	湯	above	盪
10	490	980	980	皿	below	盪
10	10	490	490	氵	left	湯
490	10	980	490	昜	right	湯
--------------------------------------------------------------------------

例:「監」
--------------------------------------------------------------------------
10	10	980	980	監	root	---
10	10	980	490	〓	above	監
10	490	980	980	皿	below	監
10	10	490	490	臣	left	〓
490	10	980	490	〓	right	〓
490	10	980	290	𠂉	above	〓
490	290	980	490	一	below	〓
--------------------------------------------------------------------------

例:「繫」
--------------------------------------------------------------------------
10	10	980	980	繫	root	---
10	10	980	490	𣪠	above	繫
10	490	980	980	糸	below	繫
10	10	490	490	𨊥	left	𣪠
490	10	980	490	殳	right	𣪠
--------------------------------------------------------------------------

例:「亟」
--------------------------------------------------------------------------
10	10	980	980	亟	root	---
10	10	980	980	〓	surround-from-center	亟
60	60	400	920	口	inserted-from-left	亟
540	60	920	920	又	inserted-from-right	亟
10	10	980	980	二	full-surround	〓
60	60	920	920	亻	full-enclosed	〓
--------------------------------------------------------------------------

例:「梁」
--------------------------------------------------------------------------
10	10	980	980	梁	root	---
10	10	980	600	〓	above	梁
10	600	980	980	木	below	梁
10	10	400	600	氵	left	〓
400	10	980	600	刅	right	〓
--------------------------------------------------------------------------

例:「寶」
--------------------------------------------------------------------------
10	10	980	980	寶	root	---
10	10	980	980	𡪓	surround-from-left	寶
490	300	980	500	缶	inserted-from-right	寶
--------------------------------------------------------------------------

例:「布」
--------------------------------------------------------------------------
10	10	980	980	布	root	---
10	10	980	600	𠂇	surround-from-upper-left	布
200	300	980	980	巾	inserted-from-lower-right	布
--------------------------------------------------------------------------

例:「襄」
--------------------------------------------------------------------------
10	10	980	980	襄	root	---
10	10	980	980	衣	full-surround	襄
20	100	970	880	〓	full-enclosed	襄
20	100	970	400	吅	above	〓
20	400	970	880	𠀎	below	〓
--------------------------------------------------------------------------
'''

L2R_prompt_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
[注意] 闕字や異体字に注意してください。また、単純な字は無理に分解しなくても大丈夫です。
また、各部品や文字はなるべく正字化せず、一番似た字体の文字を使って出力してください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：

X0	Y0	X1	Y1	親	root	---
X0	Y0	X1	Y1	部品1	相対位置	親
X0	Y0	X1	Y1	部品2	相対位置	親
X0	Y0	X1	Y1	部品3	相対位置	部品2
X0	Y0	X1	Y1	部品4	相対位置	部品2

{structure_examples_L2R_ja}
'''

A2B_prompt_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
[注意] 闕字や異体字に注意してください。また、単純な字は無理に分解しなくても大丈夫です。
また、各部品や文字はなるべく正字化せず、一番似た字体の文字を使って出力してください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：

X0	Y0	X1	Y1	親	root	---
X0	Y0	X1	Y1	部品1	相対位置	親
X0	Y0	X1	Y1	部品2	相対位置	親
X0	Y0	X1	Y1	部品3	相対位置	部品2
X0	Y0	X1	Y1	部品4	相対位置	部品2

{structure_examples_A2B_ja}
'''

prompt_ja = f'''画像にある漢字を構成する全ての部品を見つけてください。
[注意] 闕字や異体字に注意してください。また、単純な字は無理に分解しなくても大丈夫です。
また、各部品や文字はなるべく正字化せず、一番似た字体の文字を使って出力してください。
見つかった各部品は矩形座標とともに下記のような TSV 形式で出力してください：

X0	Y0	X1	Y1	親	root	---
X0	Y0	X1	Y1	部品1	相対位置	親
X0	Y0	X1	Y1	部品2	相対位置	親
X0	Y0	X1	Y1	部品3	相対位置	部品2
X0	Y0	X1	Y1	部品4	相対位置	部品2

{structure_examples_L2R_ja}

{structure_examples_A2B_ja}

例:「匵」
-------------------------------------------------------------------
10	10	980	980	匵	root	---
10	10	980	980	匚	surround-from-left	匵
180	100	820	920	𧶠	inserted-from-right	匵
-------------------------------------------------------------------

例:「凶」
------------------------------------------------------------------
0	0	1000	1000	凶	root	---
0	0	1000	1000	凵	surround-from-below	凶
100	0	900	900	㐅	inserted-from-above	凶
------------------------------------------------------------------

例:「勉」
--------------------------------------------------------------------------
10	10	980	980	勉	root	---
10	10	980	980	免	surround-from-lower-left	勉
490	20	980	900	力	inserted-from-upper-right	勉
--------------------------------------------------------------------------

例:「向」
--------------------------------------------------------------------------
10	10	980	980	向	root	---
10	10	980	980	𰃦	surround-from-above	向
90	200	900	900	口	inserted-from-below	向
--------------------------------------------------------------------------

例:「開」
------------------------------------------------------------------
10	10	980	980	開	root	---
10	10	980	980	門	surround-from-above	開
90	200	900	980	开	inserted-from-below	開
------------------------------------------------------------------

例:「后」
--------------------------------------------------------------------------
10	10	980	980	后	root	---
10	10	980	980	𠂋	surround-from-upper-left	后
490	490	980	980	口	inserted-from-lower-right	后
--------------------------------------------------------------------------

例:「履」
--------------------------------------------------------------------------
10	10	980	980	履	root	---
10	10	980	980	尸	surround-from-upper-left	履
490	490	980	980	復	inserted-from-lower-right	履
--------------------------------------------------------------------------

例:「應」
--------------------------------------------------------------------------
10	10	980	980	應	root	---
10	10	980	950	䧹	surround-from-upper-left	應
490	490	980	980	心	inserted-from-lower-right	應
--------------------------------------------------------------------------

例:「戎」
--------------------------------------------------------------------------
10	10	980	980	戎	root	---
10	10	980	950	戈	surround-from-upper-right	戎
10	100	800	980	十	inserted-from-lower-left	戎
--------------------------------------------------------------------------

例:「式」
--------------------------------------------------------------------------
10	10	980	980	式	root	---
10	10	980	950	弋	surround-from-upper-right	式
10	100	800	980	工	inserted-from-lower-left	式
--------------------------------------------------------------------------

例:「圉」
--------------------------------------------------------------------------
10	10	980	980	圉	root	---
10	10	980	980	囗	full-surround	圉
60	60	920	920	幸	full-enclosed	圉
--------------------------------------------------------------------------

例:「匿」
-------------------------------------------------------------------
10	10	980	980	匿	root	---
10	10	980	980	匚	surround-from-left	匿
180	100	820	920	若	inserted-from-right	匿
-------------------------------------------------------------------
'''


prompt = prompt_ja


def detect_ids (X1, Y1, X2, Y2, Component_Text, Component_Position, Mother):
    number_of_components = len(Component_Text)
    if ( ( number_of_components >= 3 ) and
         ( Component_Position[0] == 'root' ) ):
        match Component_Position[1]:
            case 'left':
                if ( ( Component_Position[2] == 'right' ) and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( number_of_components >= 5 ):
                        if ( ( ( Component_Text[1] == '〓' ) or
                               ( len(Component_Text[1]) >= 2 ) )
                             and
                             ( Component_Text[1] == Mother[3] ) and
                             ( Component_Text[1] == Mother[4] ) ):
                            if ( ( Component_Position[3] == 'above' ) and
                                 ( Component_Position[4] == 'below' ) ):
                                return f'⿰⿱{Component_Text[3]}{Component_Text[4]}{Component_Text[2]}'
                            elif ( ( Component_Position[3] == 'surround-from-upper-left' ) and
                                   ( ( Component_Position[4] == 'enclosed-from-upper-left' ) or
                                     ( Component_Position[4] == 'inserted-from-lower-right' ) ) ):
                                return f'⿰⿸{Component_Text[3]}{Component_Text[4]}{Component_Text[2]}'
                            else:
                                return f'⿰{Component_Text[1]}{Component_Text[2]}'
                        elif ( ( ( ( Component_Text[2] == '〓' ) or
                                   ( len(Component_Text[2]) >= 2 ) )
                                 or
                                 ( ( Component_Text[2] == '青' ) and
                                   ( Component_Text[4] == '円' ) ) )
                               and
                               ( Component_Text[2] == Mother[3] ) and
                               ( Component_Text[2] == Mother[4] ) ):
                            if ( ( Component_Position[3] == 'above' ) and
                                 ( Component_Position[4] == 'below' ) ):
                                return f'⿰{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                            elif ( ( Component_Position[3] == 'surround-from-upper-left' ) and
                                   ( ( Component_Position[4] == 'enclosed-from-upper-left' ) or
                                     ( Component_Position[4] == 'inserted-from-lower-right' ) ) ):
                                return f'⿰{Component_Text[1]}⿸{Component_Text[3]}{Component_Text[4]}'
                            else:
                                return f'⿰{Component_Text[1]}{Component_Text[2]}'
                        else:
                            return f'⿰{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿰{Component_Text[1]}{Component_Text[2]}'

            case 'above':
                if ( ( Component_Position[2] == 'below' ) and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( ( ( Component_Text[1] == '〓' ) or
                           ( len(Component_Text[1]) >= 2 ) )
                         and
                         ( number_of_components >= 5 ) and
                         ( Component_Text[1] == Mother[3] ) and
                         ( Component_Text[1] == Mother[4] ) ):
                        if Component_Position[3] == 'above':
                            if ( ( number_of_components >= 6 ) and
                                 ( Component_Text[1] == Mother[5] ) ):
                                if ( Component_Position[4] == 'full-surround' ):
                                    return f'⿱⿱{Component_Text[3]}⿴{Component_Text[4]}{Component_Text[5]}{Component_Text[2]}'
                                elif ( ( Component_Position[4] == 'enclosed' ) and
                                       ( Component_Position[5] == 'below' ) ):
                                    return f'⿱⿳{Component_Text[3]}{Component_Text[4]}{Component_Text[5]}{Component_Text[2]}'
                                elif ( ( Component_Position[4] == 'middle' ) and
                                       ( Component_Position[5] == 'below' ) ):
                                    return f'⿱⿳{Component_Text[3]}{Component_Text[4]}{Component_Text[5]}{Component_Text[2]}'
                            else:
                                return f'⿱⿱{Component_Text[3]}{Component_Text[4]}{Component_Text[2]}'
                        elif ( ( number_of_components >= 7 ) and
                               ( ( Component_Text[4] == '〓' ) or
                                 ( len(Component_Text[4]) >= 2 ) )
                               and
                               ( Component_Position[5] == 'above' ) and
                               ( Component_Position[6] == 'below' ) and
                               ( Component_Text[4] == Mother[5] ) and
                               ( Component_Text[4] == Mother[6] ) ):
                            return f'⿱⿰{Component_Text[3]}⿱{Component_Text[5]}{Component_Text[6]}{Component_Text[2]}'
                        else:
                            return f'⿱⿰{Component_Text[3]}{Component_Text[4]}{Component_Text[2]}'
                    elif ( ( ( Component_Text[2] == '〓' ) or
                             ( len(Component_Text[2]) >= 2 ) or
                             ( Component_Text[2] == '弄' ) )
                           and
                           ( number_of_components >= 5 ) and
                           ( Component_Text[2] == Mother[3] ) and
                           ( Component_Text[2] == Mother[4] ) ):
                        if ( Component_Position[3] == 'left' ):
                            return f'⿱{Component_Text[1]}⿰{Component_Text[3]}{Component_Text[4]}'
                        else:
                            return f'⿱{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                    else:
                        if ( Component_Text[1] == 'ナ' ):
                            return f'⿸𠂇{Component_Text[2]}'
                        elif ( Component_Text[1] == '𠂇' ):
                            return f'⿸{Component_Text[1]}{Component_Text[2]}'
                        if ( Component_Text[1] == '气'):
                            return f'⿹{Component_Text[1]}{Component_Text[2]}'
                        else:
                            return f'⿱{Component_Text[1]}{Component_Text[2]}'
                elif ( ( Component_Position[2] == 'middle' ) and
                       ( number_of_components >= 4 ) and
                       ( Component_Text[0] == Mother[1] ) and
                       ( Component_Text[0] == Mother[2] ) and
                       ( Component_Text[0] == Mother[3] ) ):
                    if ( ( ( Component_Text[2] == '〓' ) or
                           ( len(Component_Text[2]) >= 2 ) )
                         and
                         ( number_of_components >= 7 ) and
                         ( Component_Text[2] == Mother[4] ) and
                         ( Component_Text[2] == Mother[5] ) and
                         ( Component_Text[2] == Mother[6] ) ):
                        if ( ( Component_Position[4] == 'enclosed' ) and
                             ( Component_Position[5] == 'left' ) and
                             ( Component_Position[6] == 'right' ) ):
                            return f'⿳{Component_Text[1]}⿲{Component_Text[4]}{Component_Text[5]}{Component_Text[6]}{Component_Text[3]}'
                        else:
                            return f'⿳{Component_Text[1]}⿳{Component_Text[4]}{Component_Text[5]}{Component_Text[6]}{Component_Text[3]}'
                    else:
                        return f'⿳{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'

            case 'surround-from-upper-left':
                if ( ( ( Component_Position[2] == 'enclosed-from-upper-left' ) or
                       ( Component_Position[2] == 'inserted-from-lower-right' ) )
                     and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( ( ( Component_Text[2] == '〓' ) or
                           ( len(Component_Text[2]) >= 2 ) )
                         and
                         ( number_of_components >= 5 ) and
                         ( Component_Text[2] == Mother[3] ) and
                         ( Component_Text[2] == Mother[4] ) ):
                        if ( Component_Position[3] == 'above' ):
                            return f'⿸{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                        else:
                            return f'⿸{Component_Text[1]}{Component_Text[2]}'
                    else:
                        if ( Component_Text[1] == '勹' ):
                            return f'⿹{Component_Text[1]}{Component_Text[2]}'
                        else:
                            return f'⿸{Component_Text[1]}{Component_Text[2]}'

            case 'surround-from-lower-left':
                if ( ( ( Component_Position[2] == 'enclosed-from-lower-left' ) or
                       ( Component_Position[2] == 'inserted-from-upper-right' ) )
                     and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    return f'⿺{Component_Text[1]}{Component_Text[2]}'

            case 'surround-from-left':
                if ( ( ( Component_Position[2] == 'enclosed-from-left' ) or
                       ( Component_Position[2] == 'inserted-from-right' ) )
                     and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( Component_Text[1] == '門' ):
                        return f'⿵{Component_Text[1]}{Component_Text[2]}'
                    elif ( Component_Text[1] == '勹' ):
                        return f'⿹{Component_Text[1]}{Component_Text[2]}'
                    elif ( Component_Text[1] == '辶' ):
                        return f'⿺{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿷{Component_Text[1]}{Component_Text[2]}'

            case 'surround-from-above':
                if ( ( ( Component_Position[2] == 'enclosed-from-above' ) or
                       ( Component_Position[2] == 'inserted-from-below' ) )
                     and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( ( number_of_components >= 4 ) and
                         ( Component_Text[0] == Mother[3] ) and
                         ( Component_Position[3] == 'below' ) ):
                        return f'⿱⿵{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                    elif ( ( number_of_components >= 5 ) and
                           ( ( Component_Text[2] == '〓' ) or
                             ( len(Component_Text[2]) >= 2 ) )
                           and
                           ( Component_Text[2] == Mother[3] ) and
                           ( Component_Text[2] == Mother[4] ) ):
                        if ( ( number_of_components >= 6 ) and
                             ( ( Component_Text[2] == '〓' ) or
                               ( len(Component_Text[2]) >= 2 ) )
                             and
                             ( Component_Text[2] == Mother[5] ) and
                             ( Component_Position[3] == 'left' ) and
                             ( Component_Position[4] == 'right' ) and
                             ( Component_Position[5] == 'below' ) ):
                            if ( Component_Text[1] == '宀' ):
                                return f'⿱{Component_Text[1]}⿱⿰{Component_Text[3]}{Component_Text[4]}{Component_Text[5]}'
                            else:
                                return f'⿵{Component_Text[1]}⿱⿰{Component_Text[3]}{Component_Text[4]}{Component_Text[5]}'
                        elif ( ( Component_Position[3] == 'above' ) and
                               ( Component_Position[4] == 'below' ) ):
                            if ( Component_Text[1] == '宀' ):
                                return f'⿱{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                            else:
                                return f'⿵{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                    elif ( Component_Text[1] == '宀' ):
                        return f'⿱{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿵{Component_Text[1]}{Component_Text[2]}'

            case 'surround-from-upper-right':
                if ( ( ( Component_Position[2] == 'enclosed-from-upper-right' ) or
                       ( Component_Position[2] == 'inserted-from-lower-left' ) )
                     and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( Component_Text[1] == '戌' ):
                        return f'⿵{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿹{Component_Text[1]}{Component_Text[2]}'

            case 'full-surround':
                if ( ( Component_Position[2] == 'full-enclosed' ) and
                     ( Component_Text[0] == Mother[1] ) and
                     ( Component_Text[0] == Mother[2] ) ):
                    if ( ( ( Component_Text[2] == '〓' ) or
                           ( len(Component_Text[2]) >= 2 ) ) and
                         ( len(Component_Text) >= 5 ) and
                         ( Component_Text[2] == Mother[3] ) and
                         ( Component_Text[2] == Mother[4] ) ):
                        if ( Component_Position[3] == 'above' ):
                            return f'⿴{Component_Text[1]}⿱{Component_Text[3]}{Component_Text[4]}'
                        else:
                            return f'⿴{Component_Text[1]}{Component_Text[2]}'
                    else:
                        return f'⿴{Component_Text[1]}{Component_Text[2]}'

            case 'surround-from-center':
                if ( ( len(Component_Text) >= 4 ) and
                     ( Component_Position[2] == 'inserted-from-left' ) and
                     ( Component_Position[3] == 'inserted-from-right' ) ):
                    if ( ( ( Component_Text[1] == '〓' ) or
                           ( len(Component_Text[1]) >= 2 ) ) and
                         ( Component_Text[1] == Mother[4] ) and
                         ( Component_Text[1] == Mother[5] ) ):
                        if ( Component_Position[4] == 'full-surround' ):
                            return f'&U-i001+2FFB;⿴{Component_Text[4]}{Component_Text[5]}⿰{Component_Text[2]}{Component_Text[3]}'
                    else:                    
                        return f'&U-i001+2FFB;⿴{Component_Text[1]}⿰{Component_Text[2]}{Component_Text[3]}'

    elif ( number_of_components >= 2 ):
        match Component_Position[0]:
            case 'left':
                if ( ( Component_Position[1] == 'right' ) and
                     ( Mother[0] == Mother[1] ) ):
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

    if ( ( number_of_components >= 1 ) and
         ( Component_Position[0] == 'root' ) ):
        return Component_Text[0]

def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH, model, processor, config):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    ids_file_name  = f'{OUTPUT_PATH}/{basename}_ids.txt'
    full_file_name = f'{OUTPUT_PATH}/{basename}_full.txt'
    #print (image_file, prompt)
    print (image_file)
    images = [ image_file ]

    response = vlm_ocr.run_VLM (images, prompt, model, processor, config)

    component_number = 0
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    Component_Text = []
    Component_Position = []
    Mother = []
    with open(f'{TSV_OUTPUT_PATH}/{basename}.tsv', 'w', encoding = 'utf-8') as tsv_destfile:
        for line_match in re.findall('([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([^()（） \t\n\r]+?)(\(.+\))?\s+([^()（） \t\n\r]+?)\s+([^()（） \t\n\r]+?)\n?', response):
            x1, y1, x2, y2, line_text, comment, position, mother = line_match
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
            Mother.append(mother)
            print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}	{mother}')
            print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}	{mother}',
                   file=tsv_destfile)
            component_number = component_number + 1
            if ( ( (orx2 - orx1) > 0 ) and
                 ( (ory2 - ory1) > 0 ) ):
                im_crop = im.crop((orx1, ory1, orx2, ory2))
                im_crop.save(f'{TSV_OUTPUT_PATH}/{basename}_comp{component_number}.png')

    with open(f'{OUTPUT_PATH}/{basename}.txt', 'w', encoding = 'utf-8') as destfile:
        destfile.write(response)

    with open(f'{OUTPUT_PATH}/{basename}.prompt', 'w', encoding = 'utf-8') as prompt_file:
        prompt_file.write(prompt)
    print (X1, Y1, X2, Y2, Component_Text, Component_Position, Mother)
    return X1, Y1, X2, Y2, Component_Text, Component_Position, Mother
