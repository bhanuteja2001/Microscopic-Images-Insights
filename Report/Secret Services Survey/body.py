from reportlab.platypus import Table
from reportlab.platypus import TableStyle

from reportlab.platypus import Paragraph
from reportlab.lib.styles import (
    ParagraphStyle, 
    getSampleStyleSheet
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus.flowables import Spacer

from datetime import datetime as dt

# #################>
# Our code imports
from styles import styles # styles.py

# width = MT_WIDTH and height = 80% of MT_HEIGHT
def genBodyTable(width, height):
    
    paragraphsList = [] # useful to insert multiple paragraphs in one table cell.
    paraStyles = getSampleStyleSheet() # returns a list with some reportlab pre sets of paragraph styles.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # title
    # example:
    # 1) a - anchor link (example 1) - see survey.py line 29 for destination
    # NOTE: anchor have 2 components, the link and the destination!
    title = '''
    <a href="#surveyAnchorEXAMPLE1" color="blue">
        Survey about Joker Cat
    </a>
    '''

    titleStyle = paraStyles['Heading1']
    # NOTE: since we are dealing with objects...be careful when change properties and reuse the object later!
    titleStyle.alignment = TA_CENTER # from reportlab.lib.enums

    titlePara = Paragraph(title, titleStyle)
    paragraphsList.append(titlePara)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # text 1
    # examples:
    # 1) a - anchor link (example 2) - see line 137 for destination
    # NOTE: anchor have 2 components, the link and the destination!
    # 2) u - underline
    # 3) b - bold
    # 4) i - italic
    text1 = '''
    This 
    <a href="#surveyAnchorEXAMPLE2" color="blue">
        survey
    </a> 
    <u>aims to study the impact</u> of 
    <b><i>Joker Cat</i></b> 
    on our society as well as on our lives. 
    What we have to change and what we need to do to save ourselves!
    '''

    text1Style = paraStyles['Normal']
    # NOTE: since we are dealing with objects...be careful when change properties and reuse the object later!
    text1Style.alignment = TA_CENTER # from reportlab.lib.enums
    text1Style.fontSize = 14
    text1Style.leading = 16 # space between lines in the SAME paragraph!

    text1Para = Paragraph(text1, text1Style)
    paragraphsList.append(text1Para) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 20)
    paragraphsList.append(space)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # title 2
    title2 = 'No need to panic!'

    title2Style = paraStyles['Heading2']
    # NOTE: since we are dealing with objects...be careful when change properties and reuse the object later!
    title2Style.alignment = TA_CENTER # from reportlab.lib.enums
    title2Style.fontSize = 16

    title2Para = Paragraph(title2, title2Style)
    paragraphsList.append(title2Para)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 20)
    paragraphsList.append(space)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # text 2
    # examples:
    # 1) b - bold
    # 2) font
    # 3) i - italic
    text2 = '''
    The <b>G</b>lobal <b>I</b>ntelligence <b>S</b>ecret 
    <b>S</b>ervices - <b>GISS</b> - are here to fight against 
    <font color="red"><i><b>JC</b></i></font>.
    '''

    text2Style = ParagraphStyle(
        name = 'Justify',
        parent = paraStyles['Normal'],
        #fontSize = 14, # not needed since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 20, # space between lines in the SAME paragraph!
        alignment = TA_JUSTIFY # from reportlab.lib.enums
    )

    text2Para = Paragraph(text2, text2Style)
    paragraphsList.append(text2Para) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 20)
    paragraphsList.append(space)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # text 3
    # examples:
    # 1) a - anchor destination - see line 41 for link
    # NOTE: anchor have 2 components, the link and the destination!
    # 2) br/ - new line
    # 3) font
    # 4) b - bold
    # 5) super - superscript
    # 6) strike
    # 7) sub - subscript
    # 8) u - underline
    # 9) link - hyperlink
    text3 = ''' 
<a name='surveyAnchorEXAMPLE2' />
Joker Cat just can't stop doing this, destroying sofas and 
making people suffer... Bites people's hands and 
scratches everything he can put his hands on! 
<br/>So GISS is doing an investigation on how to 
<font size="25"><b>stop</b></font> this 
<super>
    little
</super>
 guy!
The time has come to rally to a new manner! 
Let no human deny the perils of our time...
the tides of 
a<strike>n un<strike color='green' offset='3' width='20'><strike color='black' offset='3' width='1'>winnable
        </strike>
    </strike>
</strike> 
war are upon us! 
In unity lies strength! 
From this day forward let no human make 
<sub>
    war 
</sub>
upon another human! 
<u color='red'>We shall 
<link href='https://youtu.be/B3OCXBL4Hxs' color='blue'>
    win 
</link>
through no matter the cost!</u>
    '''   

    text3Style = ParagraphStyle(
        name = 'FirstLine',
        parent = paraStyles['Normal'],
        #fontSize = 14, # not needed since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 20, # space between lines in the SAME paragraph!
        firstLineIndent = 20, # the first line in paragraph will be 20 points ahead.
        alignment = TA_JUSTIFY # from reportlab.lib.enums
    )

    text3Para = Paragraph(text3, text3Style)
    paragraphsList.append(text3Para) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 120)
    paragraphsList.append(space)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # text4
    date = dt.now() # dt from datetime import datetime as dt
    text4 = date.strftime(f'GISS - %#d %B, %Y') # <day number> <month name>, <year as four number digits>

    text4Style = ParagraphStyle(
        name = 'date',
        parent = paraStyles['Normal'],
        fontSize = 12, # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 20, # space between lines in the SAME paragraph!
        firstLineIndent = width * 70 / 100, # the first line in paragraph will be 20 points ahead.
        alignment = TA_LEFT, # from reportlab.lib.enums
    )

    text4Para = Paragraph(text4, text4Style)
    paragraphsList.append(text4Para)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 25)
    paragraphsList.append(space)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Signatures                                           |

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# example 1) without using a table
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # underscore line
    underscore1 = '_______________________'

    underscore1ParaStyle = ParagraphStyle(
        name = 'under',
        parent = paraStyles['Normal'],
        fontSize = 12, # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        alignment = TA_LEFT, # from reportlab.lib.enums
        firstLineIndent = width * 67 / 100, # NOTE: We have to align manually!
    )

    underscore1Para = Paragraph(underscore1, underscore1ParaStyle)
    paragraphsList.append(underscore1Para)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GISS boss name
    bossName = 'Dr Marco Reeves'

    bossNameParaStyle = ParagraphStyle(
        name = 'bossName',
        parent = paraStyles['Normal'],
        fontSize = 12, # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        alignment = TA_LEFT, # from reportlab.lib.enums
        firstLineIndent = width * 73 / 100, # NOTE: We have to align manually!
    )

    bossNamePara = Paragraph(bossName, bossNameParaStyle)
    paragraphsList.append(bossNamePara)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GISS boss title
    bossTitle = 'GISS Major President'

    bossTitleParaStyle = ParagraphStyle(
        name = 'bossTitle',
        parent = paraStyles['Normal'],
        fontSize = 10, # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        alignment = TA_LEFT, # from reportlab.lib.enums
        firstLineIndent = width * 72.5 / 100, # NOTE: We have to align manually!
    )

    bossTitlePara = Paragraph(bossTitle, bossTitleParaStyle)
    paragraphsList.append(bossTitlePara)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # space between texts
    space = Spacer(1, 25)
    paragraphsList.append(space)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# example 2) using a table
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # underscore line
    underscore2 = '_______________________'

    # NOTE: this paragraph will center automatically on the table 
    # because the object paraStyles['Normal'] was changed 
    # at line 59 and we are reusing it!
    underscore2ParaStyle = ParagraphStyle(
        name = 'under2',
        parent = paraStyles['Normal'], 
        fontSize = 12,  # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 0,    # NEEDED since paragraph style 'Normal' have a leading of 12!
        #alignment = TA_CENTER, # NOT NEEDED since paragraph style 'Normal' was changed to be centered!
    )

    underscore2Para = Paragraph(underscore2, underscore2ParaStyle)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GISS subdirector name
    subdirName = 'Dr Ambrose Dias'

    subdirNameParaStyle = ParagraphStyle(
        name = 'subdirName',
        parent = paraStyles['Normal'],
        fontSize = 12,  # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 0,    # NEEDED since paragraph style 'Normal' have a leading of 12!
        #alignment = TA_CENTER, # NOT NEEDED since paragraph style 'Normal' was changed to be centered!
    )

    subdirNamePara = Paragraph(subdirName, subdirNameParaStyle)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GISS subdirector title
    subdirTitle = 'GISS Subdirector'

    subdirTitleParaStyle = ParagraphStyle(
        name = 'bossTitle',
        parent = paraStyles['Normal'],
        fontSize = 10,  # NEEDED since paragraph style 'Normal' was changed before to have a font size of 14!
        leading = 0,    # NEEDED since paragraph style 'Normal' have a leading of 12!
        #alignment = TA_CENTER, # NOT NEEDED since paragraph style 'Normal' was changed to be centered!
    )

    subdirTitlePara = Paragraph(subdirTitle, subdirTitleParaStyle)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The table
    widthList = [
        width * 65 / 100,   # column 1 - 65% of width
        width * 35 / 100    # column 2 - 35% of width
    ]

    table = Table([
        ['', underscore2Para],  # row 1, columns 1 and 2
        ['', subdirNamePara],   # row 2, columns 1 and 2
        ['', subdirTitlePara]   # row 3, columns 1 and 2
    ],
    widthList
    )
    table.setStyle([
        #styles.GRID_RED_ALL,

        styles.LEFT_PADDING_ZERO_ALL,
        styles.BOTTOM_PADDING_ZERO_ALL,   
    ])

    paragraphsList.append(table) # append table with GISS Subdirector signature data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body Table
    res = Table([[paragraphsList]], width, height)
    res.setStyle([
        #styles.GRID_RED_ALL,

        styles.BOTTOM_PADDING_ZERO_ALL,
        styles.LEFT_PADDING_ZERO_ALL,

        # start at column at index 0, row at index 0
        ('VALIGN', (0,0), (-1,-1), 'TOP'), # vertical alignment - this makes the content of the table flow from top to bottom!
    ])
    
    return res