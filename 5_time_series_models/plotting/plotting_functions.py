''' Functions for model results plotting. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from matplotlib.patches import Patch
from matplotlib import colors


def visualise_cv(folds: int,
                 y: pd.Series,
                 lw: int = 10,
                 skip_first: bool = False,
                 **kwargs) -> plt.axis:
    ''' Draws a plot for indices of a time series cross validation.
    
    Args:
        folds (int): number of CV folds
        y (pd.Series): pandas object as long as the dataset
        lw (int): line width for the plot 
        **kwargs: additional keyword arguments for the function
            sklearn.model_selection.TimeSeriesSplit()
            
    Returns:
        Matplotlib axis object
    '''
    _, ax = plt.subplots()
    train_color, test_color = 'royalblue', 'darkorange'
    
    if skip_first:
        folds += 1

    tss = TimeSeriesSplit(n_splits=folds, **kwargs).split(y)
    for fold, (train_index, test_index) in enumerate(tss):
        
        if skip_first and fold == 0:
            continue
        
        # Fill in indices with the train/test groups
        indices = np.empty(len(y))
        indices[:] = 2
        indices[test_index] = 1
        indices[train_index] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [fold + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=colors.ListedColormap([train_color, test_color, 'white']),
            norm=colors.BoundaryNorm([0,1,2], 2),
        )

    if skip_first:
        ticks = np.arange(1, folds)
        tick_labels = ticks
        spacing = 0.8
    else:
        ticks = np.arange(0, folds)
        tick_labels = ticks + 1
        spacing = -0.2

    # Formatting
    ax.set(
        yticks=ticks + 0.5,
        yticklabels=tick_labels,
        xlabel='Index',
        ylabel='CV Fold',
        ylim=[folds + 0.2, spacing],
        xlim=[0, len(y)],
    )
    ax.legend(
        [Patch(color=train_color), Patch(color=test_color)],
        ['Train set', 'Test set'],
        loc='upper right',
    )
    return ax


def create_barplot_with_variance_whiskers(dataframe: pd.DataFrame,
                                          title: str,
                                          whiskers: bool = True,
                                          ylabel: str = 'Mean',
                                          xlabels_angle: int = 0,
                                          axis: str = 'columns',
                                          dtype: str = 'numeric',
                                          figsize: tuple = (8, 5)):
    ''' Plots barplot for either columns or rows of a given dataframe.
    
    Args:
        dataframe (pd.DatFrame): Input data
        title (str): Plot title
        whiskers (bool): Whether a barplot with IQR whiskers should be plotted
            or without
        ylabel (str): Y-axis label for the plot, defaults to `Mean`
        xlabels_angle (int, optional): Degrees of angle for the xticklabels.
            Defaults to 0 degrees, i.e. vertical position.
        axis (str, optional): Either `columns` or `rows`, defaults to `columns`
        dtype (str, optional): Either `numeric` or `percent`, defaults to `numeric`. In
            the case of `percent`, the percentage symbol is removed from all 
            entries of the dataframe and object dtype is converted to float.
        figsize (tuple, optional): Matplotlib figsize, defaults to (8, 5)
        
    Returns:
        Matplotlib axis object
    '''
    # Convert percent values to numeric if dtype is 'percent'
    if dtype == 'percent':
        dataframe = dataframe.replace({'%': '', '\s+': ''}, regex=True).astype(float)

    # Create figure and axis
    _, ax = plt.subplots(figsize=figsize)
        
    # Plot bars
    if axis == 'columns':
        ax.bar(dataframe.columns, dataframe.mean(), alpha=0.5, label='Mean')
        # Optionally plot whiskers
        if whiskers:
            ax.vlines(dataframe.columns,
                      np.min(dataframe.values, axis=0),
                      np.max(dataframe.values, axis=0),
                      color='black',
                      label='Min/max range')
    elif axis == 'rows':
        ax.bar(dataframe.index, dataframe.mean(axis=1), alpha=0.5, label='Mean')
        # Optionally plot whiskers
        if whiskers:
            ax.vlines(dataframe.index,
                      np.min(dataframe.values, axis=1),
                      np.max(dataframe.values, axis=1),
                      color='black',
                      label='Min/max range')
    else:
        raise ValueError("Invalid axis. Choose 'columns' or 'rows'.")
    
    # Add title and y-label
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    
    # Rotate x-axis labels for readibility
    plt.xticks(rotation=xlabels_angle, ha='right')
    
    return ax
