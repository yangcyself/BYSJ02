import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
import seaborn as sns
import matplotlib.gridspec as gspec


matplotlib.rcParams['text.usetex'] = True
# matplotlib.rc('font',family='serif', serif=['Palatino'])
matplotlib.rc('font',family='times', serif=['Palatino'])
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

sns.set_style('white')
# plt.rcParams['figure.figsize'] = [10, 8]
def set_style():
    sns.set(font='serif', font_scale=1.4)
    
   # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})
    
    plt.rcParams.update({'font.size': 21})
    plt.rc('axes', titlesize=25)     # fontsize of the axes title
    plt.rc('axes', labelsize=23)  

legendParam = {"frameon":True,"framealpha" : 1,"edgecolor":"0","fontsize":23}

set_style()