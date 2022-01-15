"""
Training and analaysing machine learning models for predicting customer churn.
Author: Wadda du Toit
Date: Jan 2022
"""


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3):
        print("\nBelow is the first 5 rows of the dataframe: \n", data_frame.head())
        print("\nThe dataframe has " + str(data_frame.shape[0]) + " rows and "
              + str(data_frame.shape[1]) + " columns.")
        print(
            "\nBelow is the total zero values in the dataframe: \n",
            data_frame.isnull().sum())
        print("\nBelow is a summary of the dataset: \n", data_frame.describe())

        plt.figure(figsize=(17, 7))
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        data_frame['Churn'].hist()
        plt.title("Churn distribution", fontsize=17)
        plt.tight_layout()
        plt.savefig('./images/eda/churn_distribution.png')
        plt.show()
        plt.figure(figsize=(17, 7))
        data_frame['Customer_Age'].hist()
        plt.title("Age distribution", fontsize=17)
        plt.tight_layout()
        plt.savefig('./images/eda/age_distribution.png')
        plt.show()
        plt.figure(figsize=(17, 7))
        data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title("Maritial status distribution", fontsize=17)
        plt.tight_layout()
        plt.savefig('./images/eda/maritial_status_distribution.png')
        plt.show()
        plt.figure(figsize=(17, 7))
        sns.distplot(data_frame['Total_Trans_Ct'])
        plt.title("Total transaction count distribution", fontsize=17)
        plt.tight_layout()
        plt.savefig('./images/eda/Total_transaction_count_distribution.png')
        plt.show()
        plt.figure(figsize=(17, 7))
        sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title("Correlation heatmap", fontsize=17)
        plt.tight_layout()
        plt.savefig('./images/eda/correlation_heatmap.png')
        plt.show()


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used 
                      for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        temp_lst = []
        temp_groups = data_frame.groupby(category).mean()['Churn']
        for val in data_frame[category]:
            temp_lst.append(temp_groups.loc[val])
        data_frame[category + "_Churn"] = temp_lst
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used 
                        for naming variables or index y column]

    output:
              training_data: X training data
              testing_data: X testing data
              training_labels: y training data
              y_test: y testing data
    '''
    labels = data_frame['Churn']
    response_data = pd.DataFrame()
    response_data[response] = data_frame[response]
    training_data, testing_data, training_labels, y_test = train_test_split(
        response_data, labels, test_size=0.3, random_state=42)
    return training_data, testing_data, training_labels, y_test, response_data


def train_models(training_data, testing_data, training_labels, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              training_data: X training data
              testing_data: X testing data
              training_labels: y training data
              y_test: y testing data
    output:
              None
    '''
    print("Training LogisticalRegression")
    lrc_model = LogisticRegression()
    lrc_model.fit(training_data, training_labels)
    training_labels_preds_lrc = lrc_model.predict(training_data)
    y_test_preds_lr = lrc_model.predict(testing_data)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc_model, testing_data, y_test, ax=ax, alpha=0.8)
    plt.title("Linear regression ROC curve", fontsize=15)
    plt.tight_layout()
    plt.savefig('./images/results/linear_regression_ROC_curve.png')
    plt.show()

    print("Training Random Forests")
    rfc_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    rfcCrossVal = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv=5)
    rfcCrossVal.fit(training_data, training_labels)
    training_labels_preds_rf = rfcCrossVal.best_estimator_.predict(training_data)
    y_test_preds_rf = rfcCrossVal.best_estimator_.predict(testing_data)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        rfcCrossVal.best_estimator_,
        testing_data,
        y_test,
        ax=ax,
        alpha=0.8)
    plt.title("Random forests ROC curve", fontsize=15)
    plt.tight_layout()
    plt.savefig('./images/results/random_forests_ROC_curve.png')
    plt.show()

    return training_labels_preds_lrc, y_test_preds_lr, lrc_model, training_labels_preds_rf,\
         y_test_preds_rf, rfcCrossVal


def train_randomForest(training_data, testing_data, training_labels, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              training_data: X training data
              testing_data: X testing data
              training_labels: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(training_data, training_labels)

    training_labels_preds_rf = cv_rfc.best_estimator_.predict(training_data)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(testing_data)

    return training_labels_preds_rf, y_test_preds_rf, rfc


def classification_report_images(firstModel,
                                 secondModel,
                                 training_labels,
                                 y_test,
                                 training_labels_preds_lr,
                                 training_labels_preds_rf,
                                 y_test_preds_lr,
                                 y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            training_labels: training response values
            y_test:  test response values
            training_labels_preds_lr: training predictions from logistic regression
            training_labels_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                training_labels, training_labels_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/results/random_forests_report.png')
    plt.show()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                training_labels, training_labels_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/results/logistical_regression_report.png')
    plt.show()

    joblib.dump(secondModel.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(firstModel, './models/logistic_model.pkl')


def feature_importance_plot(model, data, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    fig = shap.summary_plot(
        shap_values,
        testing_data,
        plot_type="bar",
        show=False,
        plot_size=(
            15,
            5))
    plt.savefig(output_pth + './shap_summary_plot.png')
    plt.tight_layout()
    plt.show()

    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(data.shape[1]), importances[indices])
    plt.xticks(range(data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth + 'feature_importance_plot.png')
    plt.show()

if __name__ == "__main__":
    data_frame = import_data("./data/bank_data.csv")
    perform_eda(data_frame)
    data_frame = encoder_helper(data_frame, [
                        'Gender',
                        'Education_Level',
                        'Marital_Status',
                        'Income_Category',
                        'Card_Category'
                        ])
    training_data, testing_data, training_labels, y_test, response_data = perform_feature_engineering(
        data_frame, 
        ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'])
    training_labels_preds_lr, y_test_preds_lr, lrc, training_labels_preds_rf, y_test_preds_rf, rfc = train_models(
        training_data, testing_data, training_labels, y_test)
    print("Training done")
    feature_importance_plot(rfc, response_data, testing_data, './images/eda/')
    classification_report_images(rfc,
                                 lrc,
                                 training_labels,
                                 y_test,
                                 training_labels_preds_lr,
                                 training_labels_preds_rf,
                                 y_test_preds_lr,
                                 y_test_preds_rf)
