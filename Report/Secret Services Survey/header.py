from reportlab.lib import colors

from reportlab.platypus import Table
from reportlab.platypus import Image

# #################>
# Our code imports
from styles import styles
from utils import utils
from resources import resrc

# width = MT_WIDTH and height = 15% of MT_HEIGHT
def genHeaderTable(width, height):
    
    heightList = [
        height * 90 / 100, # row 1 - 90% of height
        height * 10 / 100  # row 2 - 10% of height
    ]

    res = Table([
        [genHeaderTopLineTable(width, heightList[0])],  # row 1
        ['Document number: 4821']                       # row 2
    ],
    width, heightList
    )

    res.setStyle([
        
        styles.LEFT_PADDING_ZERO_ALL,
        styles.BOTTOM_PADDING_ZERO_ALL,

        # column at index 0, row at index 1
        ('FONTSIZE', (0,1), (0,1), 8),
        ('FONTNAME', (0,1), (0,1), 'Courier-Bold'),
        ('LEFTPADDING', (0,1), (0,1), 15),
    ])

    return res

# height = 90% of MT_HEIGHT
def genHeaderTopLineTable(width, height):

    widthList = [
        width * 15 / 100,   # column 1 - 15% of width
        width * 85 / 100    # column 2 - 85% of width
    ]

    gissPic = Image(resrc.GISS_PIC_PATH)
    gissPic.drawWidth = widthList[0]
    gissPic.drawHeight = height * 80 / 100

    res = Table([
        [
            gissPic,                                    # col 1
            genHeaderTLRightTable(widthList[1], height) # col 2
        ]
    ],
    widthList, height
    )

    res.setStyle([
        
        #styles.GRID_RED_ALL, # useful for debug

        styles.LEFT_PADDING_ZERO_ALL, # from styles.py
        styles.BOTTOM_PADDING_ZERO_ALL,
    ])

    return res

from reportlab.graphics.shapes import Drawing
from reportlab.graphics.shapes import String
# Header Top Line Right Table
def genHeaderTLRightTable(width, height):

    heightList = [
        height * 35 / 100, # row 1 - 35% of height
        height * 65 / 100  # row 2 - 65% of height
    ]

    titleDraw = Drawing()

    title = 'Secret Services Survey'

    font = 'MainTitleFont'
    fontSize = 30

    titleShadow = String(
        0, 0, # start point x = 0, y = 0 -> relative to table cell
        title, 
        fontName=font,
        fontSize=fontSize,
        fillColor=colors.black
    )
    titleDraw.add(titleShadow)
    
    titleFront = String(
        0, 0, # start point x = 0, y = 0 -> relative to table cell
        title, 
        fontName=font,
        fontSize=fontSize,
        fillColor=colors.red
    )
    titleDraw.add(titleFront)

    # center the text
    rowCenter = width / 2
    
    titleWidth = utils.getStringWidth(titleShadow) # from utils.py
    titleCenter = titleWidth / 2
    
    x = rowCenter - titleCenter
    x = x - x / 4 # since it's an imported font the getStringWidth doesn't returns a perfect match of string width!

    titleFront.x = x 
    titleShadow.x = x + 2 # shadow will be 2 points ahead to the right
    titleShadow.y = titleShadow.y - 2

    res = Table([
        [titleDraw],                                            # row 1
        [genHeaderTLRSubtitlePicTable(width, heightList[1])]    # row 2
    ], 
    width, heightList
    )

    res.setStyle([
        #styles.GRID_RED_ALL, # useful for debug

        styles.LEFT_PADDING_ZERO_ALL, # from styles.py
        styles.BOTTOM_PADDING_ZERO_ALL,
    ])

    return res

# Header Top Line Right Subtitle Pic Table
def genHeaderTLRSubtitlePicTable(width, height):

    widthList = [
        width * 90 / 100, # column 1 - 90% of width
        width * 10 / 100  # column 2 - 10% of width
    ]

    jcPic = Image(resrc.BABY_JC_PIC_PATH) # resrc - from resources.py
    jcPic.drawWidth = widthList[1] * 160 / 100  # 160% to overflow
    jcPic.drawHeight = height * 115 / 100       # 115% to overflow 

    res = Table([
        [
            genHeaderTLRSubtitleTable(widthList[0], height),# col 1
            jcPic                                           # col 2
        ],
    ], 
    widthList, height
    )

    res.setStyle([
        #styles.GRID_RED_ALL, # useful for debug

        styles.LEFT_PADDING_ZERO_ALL, # from styles.py
        styles.BOTTOM_PADDING_ZERO_ALL,

        # column at index 1, row at index 0
        ('LEFTPADDING', (1,0),(1,0), -(jcPic.drawWidth - widthList[1])),
        ('BOTTOMPADDING', (1,0),(1,0), -(jcPic.drawHeight - height + 5)),
    ])

    return res

# Header Top Line Right Subtitle Table
def genHeaderTLRSubtitleTable(width, height):

    heightList = [
        height * 50 / 100,  # row 1 - 50% of height
        height * 50 / 100   # row 2 - 50% of height
    ]

    res = Table([
        ['Case study'],                                             # row 1
        ['"Joker Cat an unexplained danger never seen before..."']  # row 2
    ], 
    width, heightList
    )

    res.setStyle([
        #styles.GRID_RED_ALL,

        styles.LEFT_PADDING_ZERO_ALL,
        styles.BOTTOM_PADDING_ZERO_ALL,

        # start at column at index 0, row at index 0 
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),

        # column at index 0, row at index 0 - 1st row    
        ('FONTSIZE', (0,0), (0,0), 26),
        ('BOTTOMPADDING', (0,0), (0,0), 20),

        # column at index 0, row at index 1 - 2nd row
        ('FONTSIZE', (0,1), (0,1), 13),
        ('BOTTOMPADDING', (0,1), (0,1), 15),
    ])

    return res