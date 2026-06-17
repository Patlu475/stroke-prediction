"""Publication-quality figure styling for journal submission."""

import matplotlib.pyplot as plt
from contextlib import contextmanager

from config import RESULTS_DIR

PALETTE = {
    'blue':   '#4477AA',
    'orange': '#EE6677',
    'green':  '#228833',
    'gray':   '#BBBBBB',
    'purple': '#AA3377',
    'cyan':   '#66CCEE',
}
PALETTE_LIST = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'],
                PALETTE['purple'], PALETTE['cyan']]

SINGLE_COL_WIDTH = 3.5   # inches (journal single column)
DOUBLE_COL_WIDTH = 7.0   # inches (journal double column)
FIG_DPI = 300

_STYLE_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': FIG_DPI,
    'savefig.dpi': FIG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
}


@contextmanager
def paper_style():
    """Context manager that applies publication styling to all figures inside it."""
    with plt.style.context('default'):
        old = {k: plt.rcParams.get(k) for k in _STYLE_PARAMS}
        plt.rcParams.update(_STYLE_PARAMS)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is not None:
                    plt.rcParams[k] = v


def save_fig(fig, name):
    """Save figure as both PNG (raster) and PDF (vector) in results dir."""
    fig.savefig(f'{RESULTS_DIR}/{name}.png', dpi=FIG_DPI)
    fig.savefig(f'{RESULTS_DIR}/{name}.pdf')
    plt.close(fig)
