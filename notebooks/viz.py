import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

AX_FACECOLOR = '#E3DCD0'

# Style and useful function definitions.
def titlebox(
    ax, text, color, bgcolor=None, size=8, boxsize=0.1, pad=0.05, loc=10, **kwargs
):
    """Sets a colored box about the title with the width of the plot"""
    boxsize=str(boxsize * 100)  + '%'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size=boxsize, pad=pad)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    plt.setp(cax.spines.values(), color=color)
    if bgcolor != None:
        cax.set_facecolor(bgcolor)
    else:
        cax.set_facecolor("white")
    at = AnchoredText(text, loc=loc, frameon=False, prop=dict(size=size, color=color))
    cax.add_artist(at)


def ylabelbox(ax, text, color, bgcolor=None, size=6, boxsize=0.1, pad=0.05, **kwargs):
    """Sets a colored box about the title with the width of the plot"""
    boxsize=str(boxsize * 100)  + '%'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size=boxsize, pad=pad)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.spines["top"].set_visible(True)
    cax.spines["right"].set_visible(True)
    plt.setp(cax.spines.values(), color=color)
    if bgcolor != None:
        cax.set_facecolor(bgcolor)
    else:
        cax.set_facecolor("white")

    at = AnchoredText(
        text,
        loc=10,
        frameon=False,
        prop=dict(rotation="vertical", size=size, color=color),
    )
    cax.add_artist(at)

def color_palette():
    """
    Returns a dictionary of the PBOC color palette
    """
    return {'green': '#7AA974', 'light_green': '#BFD598',
            'pale_green': '#DCECCB', 'yellow': '#EAC264',
            'light_yellow': '#F3DAA9', 'pale_yellow': '#FFEDCE',
            'blue': '#738FC1', 'light_blue': '#A9BFE3',
            'pale_blue': '#C9D7EE', 'red': '#D56C55', 'light_red': '#E8B19D',
            'pale_red': '#F1D4C9', 'purple': '#AB85AC',
            'light_purple': '#D4C2D9', 'dark_green':'#7E9D90', 'dark_brown':'#905426',
            'dark_blue': '#535D87', 'dark_grey': '#363737', 'dark_purple': '#887191'}

