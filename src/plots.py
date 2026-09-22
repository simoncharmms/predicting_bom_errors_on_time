""" 
This script stores functions for plotting.
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------

import matplotlib as plt
# import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import ticker
from matplotlib import font_manager

### ---------------------------------------------------------------------------
### Font options, import BMW fonts.
### ---------------------------------------------------------------------------

font_dirs = ['path/to/font/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams['font.family'] = 'Comic Sans'


start_time = time.time()
#next_script = 'akt_onehotencoding.py'

### ---------------------------------------------------------------------------
### Graph options.
### ---------------------------------------------------------------------------

DPI = 300
plt.rc('savefig', dpi=DPI)
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['figure.figsize'] = 6.4, 4.8  # Default.
plt.rcParams['xtick.major.pad'] = 5 # Default: 3.
plt.rcParams['ytick.major.pad'] = 5 # Default: 3.

# Set color scheme.
NRCL_COLOR = "tab:blue"
BDQV_COLOR = "tab:red"

# Set title text color to dark gray (https://material.io/color) not black.
TITLE_COLOR = 'black'
plt.rcParams['text.color'] = TITLE_COLOR

# Axis titles and tick marks are medium gray.
AXIS_COLOR = 'black'
plt.rcParams['axes.labelcolor'] = AXIS_COLOR
plt.rcParams['xtick.color'] = AXIS_COLOR
plt.rcParams['ytick.color'] = AXIS_COLOR

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 


### ---------------------------------------------------------------------------
### Figure font.
### ---------------------------------------------------------------------------

fontsize = 11
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

### ---------------------------------------------------------------------------
### Functions.
### ---------------------------------------------------------------------------

def plot(*args, **kwargs):
     """An abridged version of plt.plot()."""
     ax = plt.gca()
     return ax.plot(*args, **kwargs)

def gca(**kwargs):
    """Get the current Axes of the current Figure."""
    return plt.gcf().gca(**kwargs)

def add_titlebox(ax, text):
    ax.text(.55, .8, text,
        horizontalalignment='center',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.6),
        fontsize=fontsize)
    return ax

def xaxis (ax, xaxis):
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    return ax

def yaxis (ax, xaxis):
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    return ax

# fig, ax = plt.subplots()
# type(ax)

# fig, _ = plt.subplots()
# type(fig)

### ---------------------------------------------------------------------------
### Fancy dendrogram with max_d. 
### ---------------------------------------------------------------------------

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    
    ddata = dendrogram(*args, **kwargs)
    
    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#os.chdir(project_path)
#os.system(next_script)
