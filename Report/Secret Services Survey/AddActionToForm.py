from pdfrw.objects.pdfdict import PdfDict
from pdfrw.objects.pdfname import PdfName
from pdfrw import PdfReader, PdfWriter

def addAction():
    fileName = 'report.pdf'

    pdf_writer = PdfWriter()
    pdf_reader = PdfReader(fileName)

    # JavaScript to be appended to PDF document.
    # To learn more please check: 
    # "Developing Acrobat Applications Using JavaScript"
    # and "JavaScript for Acrobat API Reference"
    js = """

    // genIDForm
    var name = this.getField("nameText").value;
    var age = this.getField("ageText").value;
    var gender = this.getField("genderChoice").value;

    // genQuestion1
    var radioGroupSelectedVal = this.getField("q1").value;

    // genQuestion2
    var q2op1 = this.getField("q2op1").value;
    var q2op2 = this.getField("q2op2").value;
    var q2op3 = this.getField("q2op3").value;
    var q2op4 = this.getField("q2op4").value;
    
    var fieldsData = '';
    fieldsData += "Name: " + name + " Age: " + age + " Gender: " + gender + "\\n";
    fieldsData += "Feeling about JC: " + radioGroupSelectedVal + "\\n";
    fieldsData += "1) " + q2op1 + ", 2) " + q2op2 + ", 3) " + q2op3 + ", 4) " + q2op4; 
    app.alert(fieldsData);
    """    

    # PDF document - Page 2
    last = pdf_reader.pages[1]

    # Note: We have just one form in the entire pdf!
    # Annots are form fields
    for field in last.Annots:
        # Each field is compound with dictionaries inside of dictionaries inside of dictionaries...inside of dictionaries.
        # buttonSubmit is the name we gave to the last text box - see survey.py line 64
        if (field.get('/T') == '(buttonSubmit)'):
            # AA - (Additional-Actions dictionary) 
            #       Acrobat js api reference suggests this is where we should insert hidden actions - like javascript.
            #       For more please check: JavaScript for Acrobat API Reference
            #       Site: https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/js_api_reference.pdf
            #       Page: 303 or search for AA
            # D - An action that shall be performed when the mouse button is pressed inside the annotation's (field) active area.
            #       For more please check: Document management - Portable document format - part 1: PDF 1.7
            #       Site: https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf
            #       Page: 423 or search for "Entries in an annotation's additional-actions dictionary"
            field.update(PdfDict(AA=PdfDict(D=make_js_action(js))))
            break

    # Make a copy of the original file
    for page in pdf_reader.pages:
        pdf_writer.addpage(page)

    pdf_writer.write('reportChanged.pdf')


def make_js_action(js):

    action = PdfDict()

    # S - The type of action that this dictionary describes.
    #       For more please check: Document management - Portable document format - part 1: PDF 1.7
    #       Site: https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf
    #       Page: 436 or search for "Additional entries specific to a rendition action"
    action.S = PdfName.JavaScript
    # JS - A text string or stream containing a JavaScript script that shall be executed when the action is triggered.
    #       For more please check: Document management - Portable document format - part 1: PDF 1.7
    #       Site: https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf
    #       Page: 436 or search for "Additional entries specific to a rendition action"
    action.JS = js

    return action

addAction()