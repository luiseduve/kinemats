import os
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from .plot_confusion_matrix import *

def plot_histogram_class_balance(data_labels, plot_titles = None, figsize=(8,6), save_path = None):
    """
    Returns a histogram with class imbalance from a list of pandas Series containing class labels.

    :param data_labels: List of Pandas Series containing true labels.
    :return: Plot
    :rtype: matplotlib fig
    """

    num_plots = len(data_labels)

    fig, axes = plt.subplots(1, num_plots, facecolor='w', edgecolor='k', sharex=False, sharey=False, figsize=figsize)

    for i in range(num_plots):
        ax = axes[i] if num_plots > 1 else axes

        # Count data for histogram
        y = data_labels[i].value_counts()
        y.sort_index(inplace=True)

        ax.bar(y.index,y, width=0.75, color=['g'])
        ax.set_xticks(ticks = range(1,len(y.index)+1))
        ax.set_xticklabels(list(y.index))
        ax.set(xlabel="Label", ylabel="Count")

        colname = '' if plot_titles is None else plot_titles[i]
        ax.set_title(str('Class balance: '+ colname))
    
    #plt.suptitle("Mean and Std. Original Movement Data")
    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    fig.tight_layout()

def plot_mean_std_ts(data:dict, data_labels:pd.Series, rolling_window_size = 5, axes_labels = ["X","Y","Z"], colors = ['r','g','b'], figsize=(10,20), save_path = None):
    """
    Returns a plot with mean +- std of time series organized
    in rows per class label, and columns per axis of movement.

    :param data: Dictionary with keys equal to `axes_labels`. Each dict
    contains a pandas DataFrame with time series, one per row
    :param data_labels: Pandas Series containing true labels for each ts.
    :return: Plot
    :rtype: matplotlib fig
    """

    classes = list(set(data_labels.tolist()))
    num_classes = len(classes)
    num_axes = len(axes_labels)

    fig, axes = plt.subplots(num_classes, num_axes, sharex=True, sharey=True, figsize=figsize)

    for i in range(num_classes):
        for j in range(num_axes):
            t_class = classes[i]
            t_axis = axes_labels[j]

            # Filter the time series corresponding to each class
            filtered_index = data_labels[data_labels==t_class].index.values
            data_temp = pd.DataFrame(data[j, filtered_index,:])

            # Moving window with average to reduce noise.
            # Calculates mean and std among all time series for the specific class
            mean_ts = data_temp.rolling(rolling_window_size).mean().mean()
            sd_ts   = 2*data_temp.rolling(rolling_window_size).mean().std()

            # Limits to fill are in the plot
            low_line  = (mean_ts - sd_ts)
            high_line = (mean_ts + sd_ts)
        
            axes[i, j].plot(mean_ts, colors[j], linewidth=2)
            axes[i, j].fill_between(mean_ts.index, low_line, high_line, color=colors[j], alpha=0.2)

            # Set column labels
            if(i == 0):
                axes[i,j].set_title(str('TS Mean and Std with rolling window=' + str(rolling_window_size) + ' - Dim:' + t_axis), fontsize=10)
            # Set row labels
            if(j == 0):
                axes[i,j].set_ylabel(str('Class:' + str(t_class)))
    
    #plt.suptitle("Mean and Std. Original Movement Data")
    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    fig.tight_layout()


def plot_confusion_matrix(df_crosstab, title = 'Confusion Matrix', cbarlabel = 'Confusion Matrix', save_path=None):
    """ Returns a plot with the heatmap inferred from a pandas df result of crosstab function
    """
    cm = df_crosstab.to_numpy()
    headers_rows = df_crosstab.index.to_list()
    headers_cols = df_crosstab.columns.to_list()

    fig, ax = plt.subplots()
    im, cbar = heatmap(cm, headers_rows, headers_cols, ax=ax,
                    cmap="YlGn", cbarlabel=cbarlabel, vmin=0, vmax=100)
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    ax.set_title(title)

    fig.tight_layout()

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

def plot_mean_std_ts(data:np.ndarray, data_labels:np.ndarray, rolling_window_size = 5, axes_labels = ["X","Y","Z"], colors = ['r','g','b'], figsize=(10,20), save_path = None):
    """
    Returns a plot with mean +- std of time series organized
    in rows per class label, and columns per axis of movement.

    :param data: Numpy array with first dimension equal to time series index, second dimension values, and third dimension axes.
    :param data_labels: 1D numpy array with labels of each time series.
    :return: Plot
    :rtype: matplotlib fig
    """

    # Force 3D array even if it is 2D array. For compatibility with functions
    if(data.ndim == 2):
        data = np.expand_dims(data, axis=2)

    num_ts      = data.shape[0]
    length_ts   = data.shape[1]
    dim_ts      = data.shape[2]

    classes = np.unique(data_labels)
    num_classes = len(classes)
    num_axes = len(axes_labels)

    if dim_ts != num_axes or dim_ts != len(colors):
        print("Please set `axes_labels` and `colors` with a list of same length than dimensions of one time series")
        return 1

    fig, axes = plt.subplots(num_classes, num_axes, sharex=True, sharey=True, figsize=figsize)

    for i in range(num_classes):
        for j in range(num_axes):
            t_class = classes[i]
            t_axis = axes_labels[j]

            # Filter the time series corresponding to each class
            filtered_index = np.where(data_labels==t_class)[0].tolist()
            data_temp = pd.DataFrame(data[filtered_index, :, j])

            # Moving window with average to reduce noise.
            # Calculates mean and std among all time series for the specific class
            mean_ts = data_temp.rolling(rolling_window_size).mean().mean()
            sd_ts   = 2*data_temp.rolling(rolling_window_size).mean().std()

            # Limits to fill are in the plot
            low_line  = (mean_ts - sd_ts)
            high_line = (mean_ts + sd_ts)
        
            # Avoid error when there is only one dimensional TS
            idx_axes = tuple([i,j]) if(dim_ts>1) else tuple([i])

            # Plots
            axes[idx_axes].plot(mean_ts, colors[j], linewidth=2)
            axes[idx_axes].fill_between(mean_ts.index, low_line, high_line, color=colors[j], alpha=0.2)

            # Set column labels
            if(i == 0):
                axes[idx_axes].set_title(str('$\mu \pm \sigma$ | Window=' + str(rolling_window_size) + ' | Dim:' + t_axis), fontsize=10)
            # Set row labels
            if(j == 0):
                axes[idx_axes].set_ylabel(str('Class:' + str(t_class)))
    
    #plt.suptitle("Mean and Std. Original Movement Data")
    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    fig.tight_layout()
    return


def plot_violin_mc(dataframe,  x_colname="dataset", y_colname="accuracy", hue_colname=None, \
                    suptitle="", title = "", x_ticklabels = None, y_lim = None, \
                    n_rows = 1, n_cols = 1,figsize=(5,8), save_path = None,
                    boxplot_instead_violin = False):
    """
    Uses seaborn to return a violin plot from a dataframe.
    Parameters are passed directly to seaborn.violinplot()
    """

    import seaborn as sns

    # Plot, one axes per N_Neighbors
    rows_plot = n_rows
    cols_plot = n_cols #len(N_NEIGHBORS)
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(figsize[0]*cols_plot, figsize[1]*rows_plot), sharex=False, sharey=False)
    try:
        axes = axes.T.flatten()
    except Exception as e: # Only one axis
        axes = [axes]

    # Plot accuracies
    for ax_idx in range(len(axes)):
        ax = axes[ax_idx]

        # Plot per axis
        data = dataframe

        if(boxplot_instead_violin):
            sns.boxplot(ax=ax, data=data, x=x_colname, y=y_colname, hue=hue_colname, linewidth=0.5)
        else:
            sns.violinplot(ax=ax, data=data, x=x_colname, y=y_colname, hue=hue_colname, linewidth=0.5)
        
        # Texts
        # ax.set_xticks(np.arange(len(DATASETS_LIST)))
        if(x_ticklabels is not None):
            ax.set_xticklabels(x_ticklabels, rotation=0)
        if(y_lim is not None):
            ax.set_ylim(tuple(y_lim))
        ax.set(title=title, xlabel=x_colname, ylabel=y_colname)
        ax.grid(True)

    fig.suptitle(suptitle)
    fig.subplots_adjust(bottom=0.15, wspace=0.05)

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')


def plot_violin_mc_filter(dataframe,  x_colname="dataset", y_colname="accuracy", hue_colname=None, subplots_filter_colname=None, \
                    suptitle="", title = "", x_ticklabels = None, y_lim = None, \
                    n_rows = 1, n_cols = 1, figsize=(5,8), save_path = None, boxplot_instead_violin = False):
    """
    Generates multiple plots like plot_violin_mc but over different axes,
    depending on the unique() values in the column named `subplots_filter_colname`
    """

    import seaborn as sns

    # Plot, one axes per N_Neighbors
    rows_plot = n_rows
    cols_plot = n_cols #len(N_NEIGHBORS)

    # In case multiple plots need to be created, based on a filter
    categories = [None]
    if(subplots_filter_colname is not None):
        categories = dataframe[subplots_filter_colname].unique()
        rows_plot = int(np.ceil(categories.size/cols_plot))

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(figsize[0]*cols_plot, figsize[1]*rows_plot), sharex=False, sharey=False)
    try:
        axes = axes.T.flatten()
    except Exception as e: # Only one axis
        axes = [axes]

    # Plot accuracies
    for ax_idx in range(len(categories)):
        ax = axes[ax_idx]

        # Plot per axis
        data = None
        if( categories[ax_idx] == None ): # Only one value, there is no filter
            data = dataframe
        else:   # Filter the dataframe
            data = dataframe[ ( dataframe[subplots_filter_colname] == categories[ax_idx] )  ]
            title = str(subplots_filter_colname + "=" + categories[ax_idx])

        if(boxplot_instead_violin):
            sns.boxplot(ax=ax, data=data, x=x_colname, y=y_colname, hue=hue_colname, linewidth=0.5)
        else:
            parts = sns.violinplot(ax=ax, data=data, x=x_colname, y=y_colname, hue=hue_colname, linewidth=0.5)
        
        # Texts
        # ax.set_xticks(np.arange(len(DATASETS_LIST)))
        if(x_ticklabels is not None):
            ax.set_xticklabels(x_ticklabels, rotation=0)
        if(y_lim is not None):
            ax.set_ylim(tuple(y_lim))
        ax.set(title=title, xlabel=x_colname, ylabel=y_colname)
        ax.grid(True)

    fig.suptitle(suptitle)
    fig.subplots_adjust(bottom=0.15, wspace=0.05)

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')



def plot_PAA(data, data_paa, centers, vlines, axis_idx=1, axis_dim=0, ts_index=[0], axes_labels = ["X","Y","Z"], ylabel="", suptitle="Piecewise Aggregate Approximation (PAA)", figsize=(5,8), save_path = None):
    """
    Returns a plot overlapping original and PAA of a time series. Default values of axis_idx and
        axis_dim are set to provide compatibility with previous experiments that assumed dimensions in axis=0.

    :param data: Numpy array with length of array in `axis_dim` equal to length of `axes_labels`.
    :param data_paa: numpy array with summarized data
    :param centers: Values corresponding to the centers of each segment.
    :param vlines: Values corresponding to the borders of the segments to plot vertical lines.
    :param axis_idx: Which axis from the numpy array corresponds to iteration over different TS.
    :param axis_dim: Which axis from the numpy array corresponds to dimensions of the same TS (the remaining axis is TIME dimension)
    :param ts_index: Array of indices to plot, one ts per row
    :return: Plot
    :rtype: matplotlib fig
    """

    # Match the time_series_array to 3D even if it has lower dimensions

    if (data.ndim is not data_paa.ndim):
        raise ValueError("data and data_paa should have same dimension")

    dimensions = data.ndim
    if(dimensions==3):
        pass
    elif(dimensions == 2):
        data = np.expand_dims(data, axis=2)
        data_paa = np.expand_dims(data_paa, axis=2)
        dimensions = data.ndim
    elif(dimensions == 1):
        data = np.expand_dims(data, axis=1)
        data = np.expand_dims(data, axis=2)
        data_paa = np.expand_dims(data_paa, axis=1)
        data_paa = np.expand_dims(data_paa, axis=2)
        dimensions = data.ndim
    else:
        raise ValueError('Invalid number of dimensions. Max 3 dimensions in the array')
        return

    num_ts = 0
    try:
        num_ts = len(ts_index)
    except TypeError:
        num_ts = 1

    num_axes = data.shape[axis_dim]
    fig, axes = plt.subplots(num_ts, num_axes, sharex=True, sharey=True, figsize=figsize)
    
    for i in range(num_ts):
        for j in range(num_axes):
            if num_ts > 1:
                ax = axes[i,j]
            elif num_axes > 1:
                ax = axes[j]
            else:
                ax = axes

            # Dynamic allocation of indices
            idx       = [slice(None)]*dimensions
            idx[axis_idx] = ts_index[i]
            idx[axis_dim] = j
            idx = tuple(idx)

            # Plot
            ax.plot(data[idx], 'b-', label="Original")
            ax.plot(centers,  data_paa[idx], 'r--', label="PAA")
            for line_pos in vlines:
                ax.axvline(x = line_pos, linewidth=0.7, linestyle = '--', color='k', alpha=0.3)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Sample index')
            ax.set_title('Example of TS index %d - Axis %s'%(ts_index[i],axes_labels[j]))
            ax.legend()
    fig.suptitle(suptitle)

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')


def plot_APCA(data, data_apca, indices, ts_index=[0], axes_labels = ["X","Y","Z"], ylabel="",
                 suptitle="Adaptive Piecewise Constant Approximation (APCA)", figsize=(5,8), save_path = None):
    """
    Returns a plot with original and APCA of a time series index existing in data.
    :param data: Numpy array with first dimension equal to length of `axes_labels`.
    :param ts_index: Array of indexes to plot, one ts per row
    :return: Plot
    :rtype: matplotlib fig
    """

    truncated_data = True
    if(data_apca.shape == data.shape):
        #Data was not truncated
        truncated_data = False

    num_ts = 0
    try:
        num_ts = len(ts_index)
    except TypeError:
        num_ts = 1

    num_axes = data.shape[0]
    fig, axes = plt.subplots(num_ts, num_axes, sharex=True, sharey=True, figsize=figsize)
    
    for i in range(num_ts):
        for j in range(num_axes):
            if num_ts > 1:
                ax = axes[i,j]
            elif num_axes > 1:
                ax = axes[j]
            else:
                ax = axes

            # In APCA, the index is different per time series
            index = [0]+indices[j,ts_index[i],:].tolist() # Append a zero at the beginning to show the

            ax.plot(data[j,ts_index[i],:], 'b-', label="Original")
            # APCA
            if (truncated_data):
                ax.plot(index, [data_apca[j,ts_index[i],0]]+data_apca[j,ts_index[i],:].tolist(), 'r--', label="APCA")
            else: # Original frame
                ax.plot(data_apca[j,ts_index[i],:], 'r--', label="APCA")

            for line_pos in index:
                ax.axvline(x = line_pos, linewidth=0.7, linestyle = '--', color='k', alpha=0.3)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Sample index')
            ax.set_title('Example of TS index %d - Axis %s'%(ts_index[i],axes_labels[j]))
            ax.legend()
    fig.suptitle(suptitle)

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')



def plot_SAX(data, data_sax, alphabet, sax_breakpoints, index, vlines=None, hlines=None, ts_index=[0], axes_labels = ["X","Y","Z"], annotate=True,
                ylabel="", suptitle="Symbolic Aggregate approXimation (SAX)", figsize=(5,8), save_path = None):
    """
    Returns a plot with original and SAX representaiton of a time series index existing in data.
    :param data: Numpy array with first dimension equal to length of `axes_labels`.
    :param ts_index: Array of indexes to plot, one ts per row
    :return: Plot
    :rtype: matplotlib fig
    """

    num_ts = 0
    try:
        num_ts = len(ts_index)
    except TypeError:
        num_ts = 1

    # Each time series has different time length (For example when input is APCA and PAA is not performed)
    variable_vlines = False
    if(vlines.ndim > 1):
        variable_vlines = True

    num_axes = data.shape[0]
    fig, axes = plt.subplots(num_ts, num_axes, sharex=True, sharey=True, figsize=figsize)
    
    for i in range(num_ts):
        for j in range(num_axes):
            if num_ts > 1:
                ax = axes[i,j]
            elif num_axes > 1:
                ax = axes[j]
            else:
                ax = axes

            ax.plot(data[j,ts_index[i],:], 'b-', label="Original")
            
            points = data_sax[j,ts_index[i],:]
            y_axis = [ sax_breakpoints[i] for i in points]
            ax.plot(index, y_axis, 'r--', label="SAX")

            # Put text labels on each sample
            if annotate:
                num_points = len(points)
                for k in range(num_points):
                    ax.annotate(alphabet[points[k]], xy=(index[k],y_axis[k]))

            # Show vertical lines
            if vlines is not None:
                vertical_lines = vlines
                if variable_vlines:
                    vertical_lines = vlines[i,j,:]
                # Draw each vertical line
                for line_pos in vertical_lines:
                    ax.axvline(x = line_pos, linewidth=0.7, linestyle = '--', color='k', alpha=0.3)
            
            # Show horizontal lines
            if hlines is not None:
                for line_pos in hlines:
                    ax.axhline(y = line_pos, linewidth=0.7, linestyle = '--', color='k', alpha=0.3)

            ax.set_ylabel(ylabel)
            ax.set_xlabel('Sample index')
            ax.set_title('Example of TS index %d - Axis %s'%(ts_index[i],axes_labels[j]))
            ax.legend()
    fig.suptitle(suptitle)

    if(save_path is not None):
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=400, bbox_inches='tight')