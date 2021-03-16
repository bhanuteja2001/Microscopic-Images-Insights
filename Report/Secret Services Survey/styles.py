from reportlab.lib import colors

class styles: 
    GRID_BLACK_ALL = (
        'GRID', (0,0), (-1,-1), 1, colors.black
    )
    GRID_RED_ALL = (
        'GRID', (0,0), (-1,-1), 1, colors.red
    )    

    LEFT_PADDING_ZERO_ALL = (
        'LEFTPADDING',(0,0),(-1,-1),0
    )
    
    BOTTOM_PADDING_ZERO_ALL = (
        'BOTTOMPADDING',(0,0),(-1,-1), 0
    )

    TEXT_ALIGN_CENTER_ALL = (
        'ALIGN',(0,0),(-1,-1),'CENTER'
    )






