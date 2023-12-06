import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import t
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px


def outlier_check(df, col):
    """
    This function takes a dataframe and a column name as input.
    It returns the IQR, lower bound and upper bound for the column.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return iqr, lower_bound, upper_bound


def plot_distributions(stroke, features):
    """
    This function takes a dataframe and a list of features as input.
    It plots the distribution of the features for the stroke and no stroke groups.
    """
    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor('#fafafa')

    for feature in features:
        plt.subplot(221)
        sns.set_style("dark")
        plt.title(' '.join([word.capitalize()
                  for word in features[0].split('_')]) + " vs Stroke", size=15)
        sns.kdeplot(stroke.query('stroke == 1')[
                    feature], color='#c91010', shade=True, label='Had a stroke', alpha=0.5)
        sns.kdeplot(stroke.query('stroke == 0')[
                    feature], color='#1092c9', shade=True, label="Didn't have a stroke", alpha=0.5)
        plt.grid(color='gray', linestyle=':',
                 axis='x', zorder=0, dashes=(1, 7))
        plt.ylabel(features[0])
        plt.xlabel('')
        plt.yticks([])
        plt.legend(loc='upper left')

    plt.show()
    print(stroke.groupby('stroke')[features].mean().T)


def countplot_percentages(dataframe, column_name):
    """
    This function takes a dataframe and a column name as input.
    It plots the countplot of the column and annotates the percentages.
    """
    counts = dataframe[column_name].value_counts(normalize=True).reset_index()
    counts.columns = [column_name, 'percentage']
    counts['percentage'] = counts['percentage'] * 100

    fig = px.bar(counts, x=column_name, y='percentage',
                 text='percentage', title=f'{column_name} value counts',
                 labels={'percentage': 'Percentage', column_name: column_name.capitalize()})
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8,
                      uniformtext_mode='hide', height=500, width=700)
    fig.update_layout(margin=dict(l=50, r=50, b=100, t=100, pad=4))

    fig.show('svg')


def plot_dist(df, column):
    """
    This function takes a dataframe and a column name as input.
    It plots the distribution of the column.
    """
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    fig.add_trace(go.Histogram(
        x=df[column],
        name=column,
        marker_color='blue',
        opacity=0.75,
    ))

    fig.update_layout(
        title_text=' '.join([word.capitalize()
                             for word in column.split('_')]) + " Distribution",  # title of plot
        xaxis_title_text=column,
        yaxis_title_text='Count',
        bargap=0.2,
        bargroupgap=0.1
    )

    fig.show('svg')


def tabulate_percentage(dataframe, column_name):
    """
    This function takes a dataframe and a column name as input.
    It generates a table of the column values and their stroke percentages.
    """
    table = []
    for value in dataframe[column_name].unique():
        if isinstance(value, str):
            query = f"{column_name} == '{value}'"
        else:
            query = f"{column_name} == {value}"
        stroke_percentage = round(dataframe.query(query)[
                                  'stroke'].mean() * 100, 2)
        table.append([str(value), stroke_percentage])

    header = dict(values=[column_name.capitalize(), 'Stroke Percentage (%)'])
    cells = dict(values=list(zip(*table)))
    trace = go.Table(header=header, cells=cells)

    layout = go.Layout(
        title=f'Stroke Percentage by {column_name.capitalize()}')
    fig = go.Figure(data=[trace], layout=layout)

    fig.update_layout(margin=dict(l=10, r=10, b=10, t=50, pad=4))
    fig.update_layout(
        autosize=False,
        width=500,
        height=200,
    )

    fig.show('svg')


def t_and_z_test(population1, population2):
    '''
    This function takes in two populations and returns the t-test result.

    Parameters:
    population1: the first population
    population2: the second population

    Returns:
    t_test : the t-test result
    '''
    t_test = ttest_ind(population1, population2, nan_policy='omit')
    return t_test


def reject_null(p_value, alpha):
    '''
    This function takes in a p_value and alpha and returns whether to reject the null hypothesis.

    Parameters:
    p_value (float): the p_value
    alpha (float): the alpha

    Returns:
    reject (boolean): whether to reject the null hypothesis
    '''
    if p_value < alpha:
        reject = True
    else:
        reject = False
    return reject


def confidence_interval(data, confidence=0.95):
    '''
    This function takes in a data and confidence level and returns the confidence interval.

    Parameters:
    data (array): the data
    confidence (float): the confidence level

    Returns:
    confidence_interval (tuple): the confidence interval
    '''
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    stderr = np.std(data, ddof=1)/np.sqrt(n)
    margin_of_error = stderr * t.ppf((1 + confidence) / 2.0, n - 1)
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval


models = defaultdict(list)


def fit_and_evaluate_resampled(model, X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled, params={}):
    """
    This function takes a model, training and testing data and parameters as input.
    It fits the model and prints the classification report, confusion matrix and ROC AUC score.
    """
    model = model(**params).fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_resampled)
    models[model.__class__.__name__].append(
        (model, roc_auc_score(y_test_resampled, y_pred)))
    print(classification_report(y_test_resampled, y_pred))
    plot_confusion_matrix(y_test_resampled, y_pred)
    print('ROC AUC Score: ', roc_auc_score(y_test_resampled, y_pred))
    plot_roc_curve(model, X_test_resampled, y_test_resampled)


def plot_confusion_matrix(y_test, y_pred):
    """
    This function takes the actual and predicted values as input.
    It plots the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(model, X_test, y_test_resampled):
    """
    This function takes the model and testing data as input.
    It plots the ROC curve.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_resampled, y_pred_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def random_search_resampled(model, params, X_train_resampled, y_train_resampled):
    """
    This function takes a model, parameters and training data as input.
    It performs a random search and returns the best parameters.
    """
    random = RandomizedSearchCV(
        model, params, cv=5, scoring='roc_auc', n_jobs=-1)
    random.fit(X_train_resampled, y_train_resampled)
    print('Best Parameters: ', random.best_params_)
    print('Best Score: ', random.best_score_)
    return random.best_params_


def get_model_with_highest_roc_score(model_dict):
    """
    This function takes a dictionary of models and their scores as input.
    It returns the model with the highest ROC AUC score.
    """
    best_model = None
    best_roc_score = 0
    for model_name, model_scores in model_dict.items():
        for model, roc_score in model_scores:
            if roc_score > best_roc_score:
                best_model = model
                best_roc_score = roc_score
    return best_model, best_roc_score
