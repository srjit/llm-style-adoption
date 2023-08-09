import numpy as np

import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Gill Sans"
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
label_size = 8
matplotlib.rcParams['xtick.labelsize'] = label_size

import seaborn as sns

__author__ = "Sreejith Sreekumar"
__email__ = "ssreejith@protonmail.com"
__version__ = "0.0.1"



def y_fmt(tick_val, pos):

    '''
    This function makes the y-axis ticker convert huge numbers to T, B, M, K
    rounding them up to x billions, millions etc.
    '''
    if tick_val >= 0:
        if tick_val >= 1000000000000:
            val = int(tick_val)/1000000000000
            return '{:.1f} T'.format(val)
        if tick_val >= 1000000000:
            val = int(tick_val)/1000000000
            return '{:.1f} B'.format(val)
        if tick_val >= 1000000:
            val = int(tick_val)/1000000
            return '{:.1f} M'.format(val)
        if tick_val > 1000:
            val = int(tick_val) / 1000
            return '{:.1f} K'.format(val)
    else:
        if abs(tick_val) >= 1000000000000:
            val = int(abs(tick_val))/1000000000000
            return '-{:.1f} T'.format(val)
        if abs(tick_val) >= 1000000000:
            val = int(abs(tick_val))/1000000000
            return '-{:.1f} B'.format(val)
        if abs(tick_val) >= 1000000:
            val = int(abs(tick_val))/1000000
            return '-{:.1f} M'.format(val)
        if abs(tick_val) >= 1000:
            val = int(abs(tick_val)) / 1000
            return '-{:.1f} K'.format(val)
        else:
            return tick_val

y_format = tkr.FuncFormatter(y_fmt)


def plot(X_,
         Y_,
         ax=None,
         title="",
         xlabel="",
         ylabel="",
         format_y=True,
         make_x_string=False,
         **kwargs):

    '''

    Generic plotting function for Exploratory Data analysis

    :param list X\_: List/Series of x-coordinates
    :param list Y\_: List/Series of y-coordinates
    :param Axis ax: Matplotlib axis to plot
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param boolean format_y: Set y-axis labels to a readable format
                             (If it has huge numbers or such)
    :param str make_x_string: Make the ticks on x axis a string. This is better
    for dates, FY-Quarters, etc.
    :param dict **kwargs: Additional arguments for the plot
    '''
    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='k', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='k', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='#6c757d')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_axisbelow(True)
    
    if format_y:
        ax.yaxis.set_major_formatter(y_format)

    if make_x_string:
        X_ = [str(x) for x in X_]
        
    #ax.set_xticklabels(X_, rotation=90, ha="right", fontsize=11)

    return ax.plot(X_,Y_, alpha=0.6, **kwargs)



def bar(X_,
         Y_,
         ax=None,
         title="",
         xlabel="",
         ylabel="",
         format_y=True,
         make_x_string=False,
         **kwargs):

    '''

    Bar graph with formatting

    :param list X\_: List/Series of x-coordinates
    :param list Y\_: List/Series of y-coordinates
    :param Axis ax: Matplotlib axis to plot
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param boolean format_y: Set y-axis labels to a readable format
                             (If it has huge numbers or such)
    :param str make_x_string: Make the ticks on x axis a string. This is better
    for dates, FY-Quarters, etc.
    :param dict **kwargs: Additional arguments for the plot
    '''
    
    
    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='k', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='k', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='#6c757d')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_axisbelow(True)
    
    if format_y:
        ax.yaxis.set_major_formatter(y_format)

    if make_x_string:
        X_ = [str(x) for x in X_]
        
    ax.set_xticklabels(X_, rotation=90, ha="right", fontsize=11)

    return ax.bar(X_,Y_, alpha=0.6, **kwargs)



def scatter_plot(X_,
                 Y_,
                 ax=None,
                 title="",
                 xlabel="",
                 ylabel="",
                 format_y=True,
                 make_x_string=True,
                 **kwargs):

    '''

    Create a scatter plot between X and Y variables

    :param list X\_: List/Series of x-coordinates
    :param list Y\_: List/Series of y-coordinates
    :param Axis ax: Matplotlib axis to plot
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param dict **kwargs: Additional arguments for the plot
    '''
    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='darkred', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='darkred', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_axisbelow(True)

    if format_y:
        ax.yaxis.set_major_formatter(y_format)
    if make_x_string:
        X_ = [str(x) for x in X_]

    return ax.scatter(X_,Y_, alpha=0.6, **kwargs)


def plot_histogram(X_,
                   ax=None,
                   bins=10,
                   title="",
                   xlabel="",
                   ylabel="",
                   set_x_format=True,
                   kde=False,
                   **kwargs):

    '''

    Plot histogram of a distribution

    :param list X\_: List/Series of x-coordinates
    :param Axis ax: Matplotlib axis to plot
    :param bins bins: Number of bins in the histogram
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param dict **kwargs: Additional arguments for the plot
    '''
    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='darkred', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='darkred', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='darkred')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_axisbelow(True)

    if set_x_format:
        ax.xaxis.set_major_formatter(y_format)

    if kde:
        ax_tmp = ax.twinx()
        sns.kdeplot(data=X_, ax=ax_tmp, color="k", ls="--")
        ax_tmp.set_ylabel("Density",
                          color='darkred',
                          fontsize=15,
                          fontweight="bold")

    return ax.hist(X_, bins=bins, alpha=0.6,  **kwargs)


def plot_pdf_and_cdf(X_,
                     ax=None,
                     bins=10,
                     title="",
                     xlabel="",
                     ylabel="",
                     **kwargs):

    '''

    Plot PDF and CDF of a distribution.
    Note: Please do not pass color as a kwargs option. 

    :param list X\_: List/Series of x-coordinates
    :param Axis ax: Matplotlib axis to plot
    :param bins bins: Number of bins in the histogram
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param dict **kwargs: Additional arguments for the plot
    '''

    count, bins_count = np.histogram(X_, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='darkred', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='darkred', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='darkred')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    ax.plot(bins_count[1:], pdf, color="red", label="PDF", **kwargs)
    ax.legend()
    return ax.plot(bins_count[1:], cdf, label="CDF", **kwargs)


def plot_grouped_bars(df,
                      ax,
                      title,
                      xlabel,
                      ylabel,
                      index_column,
                      bar_cols,
                      colors,
                      format_y=True):

    '''
    Plot a grouped bar plot (Pivoted Table)

    :param Dataframe df: Pivoted dataframe that needs to be plotted
    :param Axis ax: Matplotlib axis to plot
    :param str title: Title of the plot
    :param str xlabel: X-axis label
    :param str ylabel: Y-axis label
    :param str index_column: Name of the column that needs to be the x-axis ticks
    :param List bar_cols: List of columns (which will have a bar) in each group
    :param List colors: List of colors for each column in a group
    
    '''
    
    
    pos = list(range(len(df[index_column]))) 
    width = 0.25 

    ax.yaxis.set_major_formatter(y_format)


    for j, col in enumerate(bar_cols):
        
        plt.bar([p + width*j for p in pos], 
            df[col], 
            width, 
            alpha=0.5, 
            color=colors[j], 
            label=col
            ) 


    ax.set_title(title, fontsize=20, color='darkred')
    ax.set_xlabel(xlabel, color='darkred', fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, color='darkred', fontsize=15, fontweight="bold")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='darkred')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(df[index_column])

    if format_y:
        ax.yaxis.set_major_formatter(y_format)
    
    ax.legend()
    return ax