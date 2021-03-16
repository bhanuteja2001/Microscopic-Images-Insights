from reportlab.platypus import Table
from reportlab.platypus import TableStyle

from reportlab.pdfbase import pdfform
from reportlab.lib import colors
from reportlab.pdfbase.acroform import RadioGroup
from reportlab.lib.utils import ImageReader

from reportlab.platypus import Paragraph
from reportlab.lib.styles import (
    ParagraphStyle, 
    getSampleStyleSheet
)

# #################>
# Our code imports
from styles import styles # styles.py
from resources import resrc # resources.py

# x = xStart and y = yStart positions to draw in canvas
def genSurveyForms(xStart, yStart, canvas):

    posX = xStart
    posY = yStart

    # Example:
    # 1) a - anchor destination - see body.py line 30 for link
    # NOTE: anchor have 2 components, the link and the destination!
    styles = getSampleStyleSheet()
    anchorDestination = Paragraph(
        "<a name='surveyAnchorEXAMPLE1' />", # see body.py line 30
        styles['Normal']
    )
    anchorDestination.wrap(
        canvas.pagewidth, 
        canvas.pageheight
    )
    anchorDestination.drawOn(canvas, posX, posY)

    canvas.setFont('FBIFont', 60)
    canvas.drawCentredString(
        canvas.pagewidth / 2, # text will be centred relativelly to this point
        posY, 
        'Joker Cat Survey'
    )

    posY -= 50

    canvas.setFont('Times-Roman', 16)
    form = canvas.acroForm # canvas has only one form
    
    # Name and age text boxes + gender drop down list
    posX, posY = genIDForm(canvas, posX, posY)

    # radio buttons
    posY -= 50
    posX, posY = genQuestion1(canvas, posX, posY)

    # we have to use ImageReader instead of Image to add this to a canvas
    jcPic = ImageReader(resrc.MAD_JC_PIC_PATH)
    canvas.drawImage(
        jcPic, 
        posX + 300, posY + 50, 
        100, 100
    )

    # check boxes
    posY -= 30
    posX, posY = genQuestion2(canvas, posX, posY)

    # reportlab don't have push buttons implemented!
    # work around for submit button
    buttonW = 58
    buttonH = 20

    posY -= 30
    posX = canvas.pagewidth - buttonW - xStart # since xStart is equals to MARGIN_LEFT and MARGIN_RIGHT is equals to MARGIN_LEFT

    form.textfield(
        value = 'Submit', # text that appears in text box
        x = posX, y = posY,
        width = buttonW, height = buttonH, borderStyle='solid',
        fontSize = 15, fillColor = colors.white,
        name = 'buttonSubmit', # useful so we can get this from javascript
    )

# Name and age text boxes + gender drop down list
def genIDForm(canvas, posX, posY):

    form = canvas.acroForm

    newPosX = posX
    newPosY = posY

    canvas.drawString(newPosX, newPosY + 6, 'Name:')

    newPosX += 45
    form.textfield(
        x = newPosX, y = newPosY,
        width = 80, height = 20, borderStyle='underlined',
        fontSize = 12, fillColor = colors.white,
        name = 'nameText'
    )

    newPosX += 80 + 20
    canvas.drawString(newPosX, newPosY + 6, 'Age:')

    newPosX += 35
    form.textfield(
        x = newPosX, y = newPosY,
        width = 40, height = 20, borderStyle='underlined',
        fontSize = 12, fillColor = colors.white,
        name = 'ageText'
    )

    newPosX += 40 + 20
    canvas.drawString(newPosX, newPosY + 6, 'Gender:')

    newPosX += 55
    options = [('Male', 'M'), ('Female', 'F')]
    form.choice(
        name = 'genderChoice',
        value = 'M',
        options = options,
        x = newPosX, y = newPosY, width = 70, height = 20,
        borderStyle = 'solid', borderWidth = 1,
        forceBorder = True,
        fillColor = colors.white,
    )

    return posX, newPosY - 6

# radio buttons
def genQuestion1(canvas, posX, posY):
    
    form = canvas.acroForm

    newPosX = posX
    newPosY = posY

    canvas.drawString(
        newPosX, 
        newPosY, 
        'Which of the following best describes your feelings about JC?'
    )
    newPosY -= 30

    q1op1 = 'I don\'t care'
    q1op2 = 'I\'m a little concerned'
    q1op3 = 'I have to be alert'
    q1op4 = 'I don\'t sleep at night'
    q1op5 = 'I\'m completely terrified'

    # Group name is important so we can group properly every option.
    groupName = 'q1'
    radioGroup = RadioGroup(groupName)

    newPosY = drawRadioOption(canvas, groupName, 'q1op1', newPosX, newPosY, q1op1)
    newPosY = drawRadioOption(canvas, groupName, 'q1op2', newPosX, newPosY, q1op2)
    newPosY = drawRadioOption(canvas, groupName, 'q1op3', newPosX, newPosY, q1op3)
    newPosY = drawRadioOption(canvas, groupName, 'q1op4', newPosX, newPosY, q1op4)
    newPosY = drawRadioOption(canvas, groupName, 'q1op5', newPosX, newPosY, q1op5)
    
    return newPosX, newPosY

# check boxes
def genQuestion2(canvas, posX, posY):

    form = canvas.acroForm

    newPosX = posX
    newPosY = posY

    canvas.drawString(
        newPosX, 
        newPosY, 
        'What actions do you think are most effective in the fight against JC?'
    )
    newPosY -= 30

    q2op1 = 'Spray him with water'
    q2op2 = 'Yell at him'
    q2op3 = 'Give him his favorite food'
    q2op4 = 'Give him a new toy'

    posYTemp = newPosY
    newPosY = drawCheckBoxOption(canvas, newPosX, newPosY, 'q2op1', 'option 1', q2op1)
    newPosY = drawCheckBoxOption(canvas, newPosX, newPosY, 'q2op2', 'option 2', q2op2)
    # to have:
    # 1) option 1           3) option 3
    # 2) option 2           4) option 4
    newPosX += 200
    newPosY = drawCheckBoxOption(canvas, newPosX, posYTemp, 'q2op3', 'option 3', q2op3)
    newPosY = drawCheckBoxOption(canvas, newPosX, newPosY, 'q2op4', 'option 4', q2op4)
    newPosX = posX

    return newPosX, newPosY

# radioGroupName - so we can group radio buttons together
# radioBtnName - so we can access them individualy in javascript
def drawRadioOption(canvas, radioGroupName, radioBtnName, posX, posY, optionText):

    form = canvas.acroForm

    newPosX = posX
    newPosY = posY

    form.radio(
        name = radioGroupName,  # to group with radio group object
        value = radioBtnName,   # the name to access each radio button in javascript
        selected = False,       # initial value   
        x = newPosX, y = newPosY, 
        buttonStyle = 'circle',
        shape = 'circle',
        fillColor = colors.pink, 
        textColor = colors.blue, 
        borderColor = colors.white
    )

    posX += 25
    canvas.drawString(posX, posY + 4, optionText)    

    return posY - 30

def drawCheckBoxOption(canvas, posX, posY, name, tooltip, optionText):

    form = canvas.acroForm

    form.checkbox(
        name=name,          # the name to access checkbox in javascript
        tooltip=tooltip,
        checked=False,      # initial value
        x = posX, y = posY,
        buttonStyle = 'diamond',
        borderStyle = 'bevelled',
        borderWidth = 1,
        borderColor = colors.red,
        fillColor = colors.green,
        textColor = colors.blue,
        forceBorder = True
    )

    posX += 25
    canvas.drawString(posX, posY + 4, optionText)

    return posY - 30
    