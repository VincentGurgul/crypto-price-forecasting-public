''' Functions to lag all time series in a pandas dataframe. '''

import pandas as pd

def shift_timeseries_by_lags(df, lags, lag_label='lag'):
    return pd.concat([
        shift_timeseries_and_create_multiindex_column(df, lag,
                                                      lag_label=lag_label)
        for lag in lags], axis=1)

def shift_timeseries_and_create_multiindex_column(
        dataframe, lag, lag_label='lag'):
    return (dataframe.shift(lag)
                     .pipe(append_level_to_columns_of_dataframe,
                           str(lag), lag_label))
    
def append_level_to_columns_of_dataframe(
        dataframe, new_level, name_of_new_level, inplace=False):
    ''' Given a (possibly MultiIndex) DataFrame, append labels to the column
    labels and assign this new level a name.

    Args:
        dataframe : a pandas DataFrame with an Index or MultiIndex columns

        new_level : scalar, or arraylike of length equal to the number of columns
        in `dataframe`
            The labels to put on the columns. If scalar, it is broadcast into a
            list of length equal to the number of columns in `dataframe`.

        name_of_new_level : str
            The label to give the new level.

        inplace : bool, optional, default: False
            Whether to modify `dataframe` in place or to return a copy
            that is modified.

    Returns:
        dataframe_with_new_columns : pandas DataFrame with MultiIndex columns
            The original `dataframe` with new columns that have the given `level`
            appended to each column label.
    '''
    old_columns = dataframe.columns

    if not hasattr(new_level, '__len__') or isinstance(new_level, str):
        new_level = [new_level] * dataframe.shape[1]

    if isinstance(dataframe.columns, pd.MultiIndex):
        new_columns = pd.MultiIndex.from_arrays(
            old_columns.levels + [new_level],
            names=(old_columns.names + [name_of_new_level]))
    elif isinstance(dataframe.columns, pd.Index):
        new_columns = pd.MultiIndex.from_arrays(
            [old_columns] + [new_level],
            names=([old_columns.name] + [name_of_new_level]))

    if inplace:
        dataframe.columns = new_columns
        return dataframe
    else:
        copy_dataframe = dataframe.copy()
        copy_dataframe.columns = new_columns
        return copy_dataframe
