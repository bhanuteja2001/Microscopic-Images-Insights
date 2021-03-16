from reportlab.lib import colors

from reportlab.platypus import Table
from reportlab.platypus import Image
from reportlab.platypus import TableStyle

from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# #################>
# Our code imports
from styles import styles

# width = MT_WIDTH and height = 5% of MT_HEIGHT
def genFooterTable(width, height):
    
    widthList = [
        width * 60 / 100,   # column 1 - 60% of width
        width * 0.4 / 100,  # column 2 - 0.4% of width
        width * 39.5 / 100, # column 3 - 39.5% of width
    ]

    heightList = [
        height * 4 / 100,   # row 1 - 4% of height
        height * 96 / 100   # row 2 - 96% of height
    ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Left footer text - will be located at column at index 0, row at index 1
    leftText = 'GISS - Super Secret Survey'
    leftTextParaStyle = ParagraphStyle(
        name = 'leftText',
        fontName = 'Times-Roman',
        fontSize = 12,
        alignment = TA_CENTER, # horizontal alignment. NOTE: Vertical alignment is set on table style!   
    )

    leftPara = Paragraph(leftText, leftTextParaStyle)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Right footer text - will be located at column at index 2, row at index 1 
    # ~~~~~~~~~~~~~~~~
    paragraphsList = [] # useful to insert multiple paragraphs in one table cell.

    # Line 1
    rightText1 = 'House nº384 Sir Housenberg Street'
    rightText1ParaStyle = ParagraphStyle(
        name = 'rightText1',
        fontName = 'Times-Roman',
        fontSize = 12,
    )

    right1Para = Paragraph(rightText1, rightText1ParaStyle)

    paragraphsList.append(right1Para)

    # Line 2
    rightText2 = 'New Welinbom'
    rightText2ParaStyle = ParagraphStyle(
        name = 'rightText2',
        fontName = 'Times-Roman',
        fontSize = 12,
    )

    right2Para = Paragraph(rightText2, rightText2ParaStyle)

    paragraphsList.append(right2Para)

    # Line 3
    rightText3 = '38274-BT EU'
    rightText3ParaStyle = ParagraphStyle(
        name = 'rightText3',
        fontName = 'Times-Roman',
        fontSize = 12,
    )

    right3Para = Paragraph(rightText3, rightText3ParaStyle)

    paragraphsList.append(right3Para)

    # ~~~~~~~~~~~~~~~~
    # table
    res = Table([
        ['','',''],                     # row 1, cols 1, 2, 3
        [leftPara,'',paragraphsList]    # row 2, cols 1, 2, 3
    ],
    widthList, heightList)

    res.setStyle([
        #styles.GRID_RED_ALL,

        styles.LEFT_PADDING_ZERO_ALL,
        styles.BOTTOM_PADDING_ZERO_ALL,

        # all columns, row at index 0 
        ('BACKGROUND', (0,0), (-1,0), colors.black),    # draw horizontal black line in footer
        # col at index 1, row at index 1                
        ('BACKGROUND', (1,1), (1,1), colors.black),     # draw vertical black line in footer 

        # col at index 0, row at index 1 
        ('VALIGN', (0,1), (0,1), 'CENTER'), #vertical alignment. NOTE: Horizontal alignment is set on paragraph style!

        # col at index 2, row at index 1 
        ('VALIGN', (2,1), (2,1), 'TOP'),
        ('ALIGN', (2,1), (2,1), 'LEFT'),
        ('LEFTPADDING', (2,1), (2,1), 5),
    ])

    return res