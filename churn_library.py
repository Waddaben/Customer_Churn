# library doc string


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
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3):
        print("\nBelow is the first 5 rows of the dataframe: \n", df.head())
        print("\nThe dataframe has " + str(df.shape[0]) + " rows and "
              + str(df.shape[1]) + " columns.")
        print(
            "\nBelow is the total zero values in the dataframe: \n",
            df.isnull().sum())
        print("\nBelow is a summary of the dataset: \n", df.describe())

        plt.figure(figsize=(17, 7))
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        df['Churn'].hist()
        plt.title("Churn distribution", fontsize=17)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(17, 7))
        df['Customer_Age'].hist()
        plt.title("Age distribution", fontsize=17)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(17, 7))
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title("Maritial status distribution", fontsize=17)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(17, 7))
        sns.distplot(df['Total_Trans_Ct'])
        plt.title("Total transaction count distribution", fontsize=17)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(17, 7))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title("Correlation heatmap", fontsize=17)
        plt.tight_layout()
        plt.show()


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        temp_lst = []
        temp_groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            temp_lst.append(temp_groups.loc[val])
        df[category + "_Churn"] = temp_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    labels = df['Churn']
    data = pd.DataFrame()
    data[response] = df[response]
    return train_test_split(data, labels, test_size=0.3, random_state=42)

def train_logisticalRegression(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_lr,y_test_preds_lr

def train_randomForest(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
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
    cv_rfc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    return y_train_preds_rf,y_test_preds_rf


def classification_report_images(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') 
    plt.axis('off')
    plt.show()
    plt.savefig('./images/results/random_forests_report.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') 
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.show()
    plt.savefig('./images/results/logiestical_regression_report.png')




def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass





if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    df = encoder_helper(df, [
                        'Gender',
                        'Education_Level',
                        'Marital_Status',
                        'Income_Category',
                        'Card_Category'
                        ])

    X_train, X_test, y_train, y_test = perform_feature_engineering(df, ['Customer_Age', 'Dependent_count', 'Months_on_book',
                                                                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                                                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                                                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                                                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                                                        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                                                                        'Income_Category_Churn', 'Card_Category_Churn'])

    y_train_preds_lr,y_test_preds_lr = train_logisticalRegression(X_train, X_test, y_train, y_test)
    y_train_preds_rf,y_test_preds_rf = train_randomForest(X_train, X_test, y_train, y_test)

    classification_report_images(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    """
    feature_importance_plot(model, X_data, output_pth)
    train_models(X_train, X_test, y_train, y_test)"""
