# Other videos:
# Generate PDF with Python - Reportlab: https://youtu.be/ZDR7-iSuwkQ
# Generate PDF with Python - Reportlab - Create Table: https://youtu.be/B3OCXBL4Hxs
# Generate PDF with Python - Reportlab - Create Table - Part 2: https://youtu.be/r--iZCQbxzE
# Generate PDF with Python - Reportlab - Create Charts: https://youtu.be/FcZ9wTGmMrw
# Generate PDF with Python - Reportlab - Joker Cat Report demo: https://youtu.be/SDeOPyhO26M
# Generate PDF with Python - Reportlab - Secret Services Survey - Preview: https://youtu.be/AviWgFXFTls
# 
# Run Python Script from Excel VBA: https://youtu.be/Z4SC53VZh-w
# Run Python Script from Excel VBA - Part 2: https://youtu.be/4Z9via5_q9k
# Run Python Script from Excel VBA - Part 3.1: https://youtu.be/PoEnWr6c1cM
# Run Python Script from Excel VBA - Part 3.2: https://youtu.be/Tkk0aedRyU4
# 
# Basic Python KeyLogger: https://youtu.be/AS4PnrWv-f4
# Convert .py into .exe: https://youtu.be/CftCQYNb7B4
# Image to Text with Python - pytesseract: https://youtu.be/4DrCIVS5U3Y
# Speech to Text with Python: https://youtu.be/If2HJ23zP2U
# Weather Forecast with Python: https://youtu.be/O9G4vBsiV40
# Search Movie with Python - IMDbPY: https://youtu.be/vzOdCPV7zvs
# Generate and Read QR Code with Python: https://youtu.be/2QK942FPCw0
# Run JavaScript from Python: https://youtu.be/ByjpBvpPp8Q
# Run Python in Browser - Brython: https://youtu.be/dFNXwq5kmNk
# Hide Text in Image with Python - Stegano: https://youtu.be/IhXbJfLCst0
# HTML to PDF with Python: https://youtu.be/m3u3oLgDcJI
# Web Scraping with Python - BeautifulSoup: https://youtu.be/Jnn2kIqPH7o
# Generate Excel with Python - OpenPyXL: https://youtu.be/KNdqnIpl2UE
# Translate Text with Python - googletrans: https://youtu.be/yRFkI8miPHA
# Convert Python 2 to Python 3 Code - 2to3: https://youtu.be/t0v4F396_nc
# Face Detection with Python - OpenCV: https://youtu.be/FeUAmWZ7Clw
# Run Python Script in LibreOffice: https://youtu.be/3Ef_ordyWQs
# Generate Excel with Python - xlwings: https://youtu.be/sGvMLmLOH5g
# RESTful Web Service - Hello World - Java Spring: https://youtu.be/RXkLlq8YxeM
# 
# ASP.NET Web API intro and Hello World - https://youtu.be/RJpBix5s3BY
# ASP.NET Web API - Controllers - https://youtu.be/nlFVQddx1wc
# ASP.NET Web API - Routing - https://youtu.be/2GvX8puS2is
# ASP.NET Web API - Parameter Binding - https://youtu.be/LuFoVFeEBPw
# ASP.NET Web API - File Uploading - https://youtu.be/MzDQALXH-SI
# ASP.NET Web API - Client Console App - https://youtu.be/Ry45EkZQ2CM
# ASP.NET Web API - BSON - https://youtu.be/BXnfCMcERaw
# ASP.NET Web API - Action Results - https://youtu.be/W80Vflesu8s
# ASP.NET Web API - Content Negotiation - https://youtu.be/vdKF1WTYVq0
# ASP.NET Web API - Media Type Formatters - https://youtu.be/SGeudFDlvCI
# ASP.NET Web API - Action Filters - https://youtu.be/x7kZJf416J4
# ASP.NET Web API - Model Validation - https://youtu.be/sN3qYlxr_vA
# ASP.NET Web API - Save File in SQL Server with ADO.NET - https://youtu.be/LixIRqhzUNQ
# ASP.NET Web API - File Downloading - https://youtu.be/bS_bXsT0kq4
# ASP.NET Web API - Get File from SQL Server with ADO.NET - https://youtu.be/zxPVmGpX07I
# ASP.NET Web API - Save File in SQL Server with Entity Framework - https://youtu.be/JYPaOO_fCug
# ASP.NET Web API - Get File from SQL Server with Entity Framework - https://youtu.be/QxOeuuosgH4
# ASP.NET Web API - Exception Filters - https://youtu.be/1x0s0K3w3Z4
# 
# Playlists:
# Python Pandas: https://www.youtube.com/playlist?list=PLOGAj7tCqHx_c5uWrZX4ykdujODcqczmQ
# Python and SQL Server: https://www.youtube.com/playlist?list=PLOGAj7tCqHx9Add6MWzl_5Wbix9V1OjSx
# Numpy Exercises: https://www.youtube.com/playlist?list=PLOGAj7tCqHx9eQjST2RV-_Py3EJHqRq0C
# Swift: https://www.youtube.com/playlist?list=PLOGAj7tCqHx9C08vyhSMciLtkMSPiirYr
# JavaScript: https://www.youtube.com/playlist?list=PLOGAj7tCqHx_grLMl0A0yC8Ts_ErJMJft
# c#: https://www.youtube.com/playlist?list=PLOGAj7tCqHx9H5dGNA4TGkmjKGOfiR4gk
# Java: https://www.youtube.com/playlist?list=PLOGAj7tCqHx-ey9xikbXOfGdbvcOielRw

from reportlab.lib.pagesizes import A4

from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import Color

from reportlab.platypus import PageBreak

# #################>
# Our code imports
from header import genHeaderTable
from body import genBodyTable
from footer import genFooterTable
from styles import styles
from utils import RotatedParagraph
from survey import genSurveyForms

# ##########################
# Register fonts
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

pdfmetrics.registerFont(
    TTFont('MainTitleFont', 'fonts\Depressionist_3_Revisited_2010.ttf')
)
pdfmetrics.registerFont(
    TTFont('FBIFont', 'fonts\FBI Old Report-Regular.ttf')
)

# #################
# Create PDF
pdf = canvas.Canvas('report.pdf')
# Set the size of a A4 page
pdf.pagewidth = A4[0]
pdf.pageheight = A4[1]

pdf.setTitle('Global Intelligence Secret Services Survey')

# #################
# document margins
MARGIN_LEFT = MARGIN_RIGHT = 40
MARGIN_TOP = MARGIN_BOTTOM = 20

# mainTable sizes
MT_WIDTH = A4[0] - (MARGIN_LEFT + MARGIN_RIGHT)
MT_HEIGHT = A4[1] - (MARGIN_TOP + MARGIN_BOTTOM)

# Header table height
HT_HEIGHT = MT_HEIGHT * 15 / 100 # 15% of MT_HEIGHT
# Body table height
BT_HEIGHT = MT_HEIGHT * 80 / 100 # 80% of MT_HEIGHT
# Footer table height
FT_HEIGHT = MT_HEIGHT * 5 / 100  # 5% of MT_HEIGHT

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Page 1
# For a better understanding of this code please see:
# Generate PDF with Python - Reportlab - Create Table - Part 2: https://youtu.be/r--iZCQbxzE
mainTablePage1 = Table(
    [   # Tables inside tables
        [genHeaderTable(MT_WIDTH, HT_HEIGHT)], # from header.py
        [genBodyTable(MT_WIDTH, BT_HEIGHT)],   # from body.py
        [genFooterTable(MT_WIDTH, FT_HEIGHT)], # from footer.py
    ],
    MT_WIDTH,
    [HT_HEIGHT, BT_HEIGHT, FT_HEIGHT]
)

# For a better understanding of this code please see:
# Generate PDF with Python - Reportlab - Create Table: https://youtu.be/B3OCXBL4Hxs
# Generate PDF with Python - Reportlab - Create Table - Part 2: https://youtu.be/r--iZCQbxzE
mainTablePage1.setStyle([
    #styles.GRID_BLACK_ALL, # useful for debug

    styles.LEFT_PADDING_ZERO_ALL, # from styles.py
    styles.BOTTOM_PADDING_ZERO_ALL,
])

mainTablePage1.wrapOn(pdf, 0, 0)
mainTablePage1.drawOn(pdf, MARGIN_LEFT, MARGIN_BOTTOM)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Confidential Text
def drawPageWaterMark():
    textConfidential = 'Confidential'

    textConfidentialParaStyle = ParagraphStyle(
        name = 'Confidential',
        fontSize = 200,
        fontName = 'FBIFont',
        leading = 0,
        textColor = Color(0, 0, 0, alpha = 0.4) # alpha to be transparent
    )

    # from utils import RotatedParagraph
    textConfidentialPara = RotatedParagraph(
        textConfidential, 
        textConfidentialParaStyle,
        -50
    )

    maxWidth = pdf.pagewidth
    maxHeight = pdf.pageheight

    textConfidentialPara.wrap(maxWidth, maxHeight)
    textConfidentialPara.drawOn(
        pdf, 
        maxWidth / 2, 
        maxHeight
    )

# confidential text at page 1
drawPageWaterMark()

# showPage -> This makes a 'page break'
# from documentation: 
# def showPage(self) - Close the current page and possibly start on a new page.
pdf.showPage()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Page 2
# confidential text at page 2
drawPageWaterMark()

# Generate the content for the 2nd page
genSurveyForms(
    MARGIN_LEFT, 
    MT_HEIGHT - MARGIN_TOP, 
    pdf
)

# save the document
pdf.save()