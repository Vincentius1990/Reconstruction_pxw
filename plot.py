import torch
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

def visualize_heatmap(x, y, heat_list, heat_pre_list):
    '''

    '''

    heat = np.concatenate(heat_list, axis=1)
    heat_pre = np.concatenate(heat_pre_list, axis=1)
    heat_error = heat_pre - heat
    plt.figure(figsize=(50, 10))
    plt.contourf(x, y, heat, levels=50, cmap=matplotlib.cm.coolwarm)
    plt.savefig('figure/heat.png', bbox_inches='tight', pad_inches=0)
    plt.figure(figsize=(50, 10))
    plt.contourf(x, y, heat_pre, levels=50, cmap=matplotlib.cm.coolwarm)
    plt.savefig('figure/heat_pre.png', bbox_inches='tight', pad_inches=0)
    plt.figure(figsize=(52, 10))
    plt.contourf(x, y, heat_error, levels=50, cmap=matplotlib.cm.coolwarm)
    plt.colorbar(fraction=0.01, pad=0.005)
    plt.savefig('figure/heat_err.png', bbox_inches='tight', pad_inches=0)