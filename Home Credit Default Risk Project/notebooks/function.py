import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def missing_values_table(df):
    '''
    Function to calculate missing values by column
    :param df: dataframe
    :return: missing values table
    '''
    missing_values = []
    cols = []
    percentages = []
    for col in df.columns:
        missing_values.append(df[col].null_count())
        cols.append(col)
        percentages.append(df[col].null_count() / df.shape[0] * 100)

    data = []
    data.append(pl.Series('Missing Values', missing_values))
    data.append(pl.Series('cols', cols))
    data.append(pl.Series('percentages', percentages))

    mis_val_table = pl.DataFrame(data)

    mis_val_table = mis_val_table.sort('Missing Values', descending=True)

    return mis_val_table


def plot_bar(dataframe, column_name):
    '''
    Function to plot bar chart
    :param dataframe: dataframe
    :param column_name: column name
    :return: bar chart
    '''
    pandas_df = dataframe.to_pandas()
    plt.figure(figsize=(8, 5))
    plt.bar(pandas_df[column_name].value_counts().index,
            pandas_df[column_name].value_counts())
    plt.title(f'Distribution of {column_name} column')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.show()


def plot_bar_xticks(dataframe, column_name):
    '''
    Function to plot bar chart with xticks
    :param dataframe: dataframe
    :param column_name: column name
    :return: bar chart
    '''
    pandas_df = dataframe.to_pandas()
    plt.figure(figsize=(8, 5))
    plt.bar(pandas_df[column_name].value_counts().index,
            pandas_df[column_name].value_counts())
    plt.title(f'Distribution of {column_name} column')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


def plot_hist(dataframe, column_name):
    '''
    Function to plot histogram
    :param dataframe: dataframe
    :param column_name: column name
    :return: histogram
    '''
    pandas_df = dataframe.to_pandas()
    plt.figure(figsize=(8, 5))
    sns.distplot(pandas_df[column_name], bins=100, kde=True)
    plt.title(f'Distribution of {column_name} column')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.show()


def plot_distribution_comp(dataframe, var, nrow=2):
    '''
    Function to plot distribution comparison
    :param dataframe: dataframe
    :param var: variable
    :param nrow: number of rows
    :return: distribution comparison
    '''
    i = 0
    t1 = dataframe.loc[dataframe['TARGET'] != 0]
    t0 = dataframe.loc[dataframe['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.legend()
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


def reduce_memory_usage_pl(df, name):
    """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
    """
    print(
        f"Memory usage of dataframe {name} is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    Numeric_Float_types = [pl.Float32, pl.Float64]
    for col in df.columns:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))
            else:
                pass
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    print(
        f"Memory usage of dataframe {name} became {round(df.estimated_size('mb'), 2)} MB")
    return df


def plot_horizontal_bar(dataframe, column_name):
    pandas_df = dataframe.to_pandas()
    plt.figure(figsize=(8, 5))
    plt.barh(pandas_df[column_name].value_counts().index,
             pandas_df[column_name].value_counts())
    plt.title(f'Distribution of {column_name} column')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.gca().invert_yaxis()
    plt.show()


def outlier_check(dataframe):
    '''
    Function to check outliers
    :param dataframe: dataframe
    :return: outlier
    '''
    dataframe = dataframe.to_pandas()
    for col in dataframe.columns:
        if dataframe[col].dtype != 'object' and dataframe[col].dtype != 'str' and dataframe[col].dtype != 'bool' and dataframe[col].dtype != 'category':
            if col != 'SK_ID_CURR' and col != 'TARGET' and col != 'SK_ID_BUREAU' and col != 'SK_ID_PREV':
                print(f'Outlier in {col} column')
                sns.boxplot(dataframe[col])
                plt.show()


def remove_outlier(dataframe):
    # identify outliers with interquartile range
    dataframe = dataframe.to_pandas()
    for col in dataframe.columns:
        if dataframe[col].dtype != 'object' and dataframe[col].dtype != 'str' and dataframe[col].dtype != 'bool' and dataframe[col].dtype != 'category':
            if col != 'SK_ID_CURR' and col != 'TARGET' and col != 'SK_ID_BUREAU' and col != 'SK_ID_PREV':
                q25, q75 = np.percentile(dataframe[col], 25), np.percentile(
                    dataframe[col], 75)
                iqr = q75 - q25
                # calculate the outlier cutoff
                cut_off = iqr * 1.5
                lower, upper = q25 - cut_off, q75 + cut_off
                # identify outliers
                outliers = [x for x in dataframe[col]
                            if x < lower or x > upper]
                print(f'Number of outliers in {col} column is {len(outliers)}')
                # remove outliers
                outliers_removed = [x for x in dataframe[col]
                                    if x >= lower and x <= upper]
                dataframe = dataframe[dataframe[col].isin(outliers_removed)]
                print(f'Outlier removed dataframe shape: {dataframe.shape}')
                print(f'Outlier removed dataframe shape: {dataframe.shape}')
                print('----------------------------------------------')

    return pl.from_pandas(dataframe)


def aggregate_and_remove(df, column_names):
    '''
    Function to aggregate and remove columns
    :param df: dataframe
    :param column_names: column names
    :return: dataframe
    '''
    agg_ops = []

    for col in column_names:
        agg_op = pl.mean(col).alias(f'{col}_MEAN')
        agg_ops.append(agg_op)

    df_agg = df.groupby('SK_ID_CURR').agg(agg_ops)

    for col in column_names:
        if col in df_agg.columns:
            df_agg = df_agg.drop(col)

    return df_agg
