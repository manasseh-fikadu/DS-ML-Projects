import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


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
