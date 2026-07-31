from PIL import Image
import os
import re
import mlx.core as mx
import vlm_ocr

# component_prompt = '''<image> Run OCR for component (or structure) of the Kanji and output the result.
# '''
component_prompt = '''<image> 漢字部品用 OCR を実行し、結果を文字もしくは IDS で返してください。
'''

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
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'

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

            case 'relative':
                if ( Component_Position[1] == 'relative' ):
                    if ( ( X2[0] <= X1[1] ) and
                         ( abs(Y1[1] - Y1[0]) <= 5 ) and
                         ( abs(Y2[1] - Y2[0]) <= 5 ) ):
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Y2[0] <= Y1[1] ) and
                           ( abs(X1[1] - X1[0]) <= 5 ) and
                           ( abs(X2[1] - X2[0]) <= 5 ) ):
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

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
                         ( ( Component_Text[0] == '一' ) and
                           ( Component_Text[1] == '厶' ) )
                         or
                         ( ( Component_Text[0] == '十' ) and
                           ( Component_Text[1] == '目' ) ) ):
                        return f'⿱⿱{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
                    elif ( ( ( Component_Text[1] == '目' ) and
                             ( Component_Text[2] == '廾' ) )
                           or
                           ( ( Component_Text[1] == '目' ) and
                             ( Component_Text[2] == '八' ) )
                           or
                           ( ( Component_Text[1] == '田' ) and
                             ( Component_Text[2] == '共' ) )
                           or
                           ( ( Component_Text[1] == '二' ) and
                             ( Component_Text[2] == '厶' ) )
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

            case 'surround':
                if ( ( Component_Position[1] == 'enclosed' ) and
                     ( Component_Position[2] == 'enclosed' ) and
                     ( Component_Text[0] == '囗' ) ):
                    return f'⿴{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'

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
                    if ( ( Component_Text[0] == '艹' ) and
                         ( Component_Text[2] == '十' ) and
                         ( Component_Text[3] == '十' ) ):
                        return f'⿴⿱{Component_Text[0]}⿰{Component_Text[2]}{Component_Text[3]}{Component_Text[1]}'
                    else:
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

            case 'surround':
                if ( ( Component_Position[1] == 'enclosed' ) and
                     ( Component_Position[2] == 'enclosed' ) and
                     ( Component_Position[3] == 'enclosed' ) and
                     ( Y2[1] <= Y1[2] ) and
                     ( Y2[2] <= Y1[3] ) ):
                    if Component_Text[0] == '匚':
                        return f'⿷{Component_Text[0]}⿳{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
                    else:
                        return f'⿴{Component_Text[0]}⿳{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'

def merge_left_and_right (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
    im = Image.open(image_file)
    image_width, image_height = im.size
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
    if ( ( (X2[0] - X1[0]) > image_width / 3 ) and
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
        comp1_response = vlm_ocr.run_VLM ([comp1_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp1_response}".')
        Component_Position[0] = 'above'
        if len(comp1_response) == 1:
            Component_Text[0] = comp1_response
        else:
            Component_Text[0] = f'⿰{Component_Text[0]}{orig_comp2}'
    else:
        Component_Text[0] = f'⿰{Component_Text[0]}{orig_comp2}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_below_left_and_right (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
    im = Image.open(image_file)
    image_width, image_height = im.size
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
    if ( ( (X2[1] - X1[1]) > image_width / 3 ) and
         ( (Y2[1] - Y1[1]) > 0 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp2_response = vlm_ocr.run_VLM ([comp2_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'below'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿰{orig_comp2}{orig_comp3}'
    else:
        Component_Text[1] = f'⿰{orig_comp2}{orig_comp3}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_above_and_below (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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
         ( (Y2[0] - Y1[0]) > image_height / 3 ) ):
        im_crop = im.crop((X1[0], Y1[0], X2[0], Y2[0]))
        comp1_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp1.png'
        im_crop.save(comp1_image_file_name)

        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        if (os.path.isfile(comp2_image_file_name)):
            os.remove(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.rename(comp3_image_file_name, comp2_image_file_name)

        comp1_response = vlm_ocr.run_VLM ([comp1_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp1_response}".')
        if len(comp1_response) == 1:
            Component_Text[0] = comp1_response
        else:
            Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
    else:
        Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_right_above_and_below (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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
         ( (Y2[1] - Y1[1]) > image_height / 3 ) ):
        im_crop = im.crop((X1[1], Y1[1], X2[1], Y2[1]))
        comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
        im_crop.save(comp2_image_file_name)

        comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
        if (os.path.isfile(comp3_image_file_name)):
            os.remove(comp3_image_file_name)

        comp2_response = vlm_ocr.run_VLM ([comp2_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp2_response}".')
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿱{orig_comp2}{orig_comp3}'
    else:
        Component_Text[1] = f'⿱{orig_comp2}{orig_comp3}'
    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_enclosed_vertical3 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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

        comp2_response = vlm_ocr.run_VLM ([comp2_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'enclosed'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'
    else:
        Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_enclosed_vertical4 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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

def merge_below_vertical3 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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

        comp2_response = vlm_ocr.run_VLM ([comp2_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'below'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'
    else:
        Component_Text[1] = f'⿳{orig_comp2}{orig_comp3}{orig_comp4}'

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def merge_below_left_and_vertical2 (image_file, X1, Y1, X2, Y2, Component_Text, Component_Position, TSV_OUTPUT_PATH, model, processor, config):
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

        comp2_response = vlm_ocr.run_VLM ([comp2_image_file_name], component_prompt, model, processor, config)
        print (f'new component = "{comp2_response}".')
        Component_Position[1] = 'below'
        if len(comp2_response) == 1:
            Component_Text[1] = comp2_response
        else:
            Component_Text[1] = f'⿰{orig_comp2}⿱{orig_comp3}{orig_comp4}'
    else:
        Component_Text[1] = f'⿰{orig_comp2}⿱{orig_comp3}{orig_comp4}'

    return X1, Y1, X2, Y2, Component_Text, Component_Position

def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH, model, processor, config):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    ids_file_name  = f'{OUTPUT_PATH}/{basename}_ids.txt'
    full_file_name = f'{OUTPUT_PATH}/{basename}_full.txt'
    print (image_file, prompt)
    images = [ image_file ]

    response = vlm_ocr.run_VLM (images, prompt, model, processor, config)

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
                                                                                                   TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                      TSV_OUTPUT_PATH, model, processor, config)
                    Component_Position[0] = 'left'
                    Component_Position[1] = 'right'

                elif ( ( Component_Position[1] == 'enclosed' ) and
                       ( Component_Position[2] == 'below' ) ):
                    X1, Y1, X2, Y2, Component_Text, Component_Position = merge_right_above_and_below (image_file,
                                                                                                      X1, Y1,
                                                                                                      X2, Y2,
                                                                                                      Component_Text,
                                                                                                      Component_Position,
                                                                                                      TSV_OUTPUT_PATH, model, processor, config)
                    Component_Position[0] = 'left'
                    Component_Position[1] = 'right'

                elif ( ( Component_Position[1] == 'surround-from-above' ) and
                       ( Component_Position[2] == 'enclosed' ) ):
                    X1, Y1, X2, Y2, Component_Text, Component_Position = merge_right_above_and_below (image_file,
                                                                                                      X1, Y1,
                                                                                                      X2, Y2,
                                                                                                      Component_Text,
                                                                                                      Component_Position,
                                                                                                      TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                          TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                   TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                   TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                   TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                   TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                    TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                    TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                               TSV_OUTPUT_PATH, model, processor, config)

        elif ( ( Component_Position[0] == 'surround-from-lower-left' ) and
               ( Component_Position[1] == 'above' ) and
               ( Component_Position[2] == 'sandwiched-from-above-and-below' ) and
               ( Component_Position[3] == 'sandwiched-from-above-and-below' ) ):
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_enclosed_vertical3 (image_file,
                                                                                           X1, Y1,
                                                                                           X2, Y2,
                                                                                           Component_Text,
                                                                                           Component_Position,
                                                                                           TSV_OUTPUT_PATH, model, processor, config)

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
                                                                                       TSV_OUTPUT_PATH, model, processor, config)
            X1, Y1, X2, Y2, Component_Text, Component_Position = merge_below_left_and_right (image_file,
                                                                                             X1, Y1,
                                                                                             X2, Y2,
                                                                                             Component_Text,
                                                                                             Component_Position,
                                                                                             TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                                 TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                        TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                        TSV_OUTPUT_PATH, model, processor, config)
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
                                                                                           TSV_OUTPUT_PATH, model, processor, config)

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

                comp1_response = vlm_ocr.run_VLM ([comp1_image_file_name], component_prompt, model, processor, config)
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
