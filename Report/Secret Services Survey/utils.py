from reportlab.platypus import Paragraph

class utils:
    # Gets width of a string
    # Note: Don't work perfectly with some imported fonts!
    def getStringWidth(text):

        # returns:
        # (x0, y0, x1, y1)
        # x0 and y0 start point
        # x1 and y1 end point
        bounds = text.getBounds()
        xPosStart = bounds[0]
        xPosEnd = bounds[2]
        res = xPosEnd - xPosStart

        return res

# Extends Paragraph class
class RotatedParagraph(Paragraph):

    def __init__(self, text, style, degrees):
        self.deg = degrees
        Paragraph.__init__(self, text, style)

    # Override draw method
    def draw(self):
        self.canv.rotate(self.deg)
        Paragraph.draw(self) 