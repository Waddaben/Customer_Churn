"""
File for testing and logging the successes and errors for the 'churn_library.py' file.
Author: Wadda du Toit
Date: Jan 2022
"""
import logging
import churn_library as cls



logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data, file location: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info("Testing import_data, rows and columns exists: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The data frame doesn't appear to have rows and columns")
        raise err

    return data_frame


def test_eda(perform_eda, data_frame):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data_frame)
        logging.info("Testing perform_eda, input is correct format: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: input is not a dataframe")
        raise err

    try:
        perform_eda(data_frame)
        logging.info("Testing perform_eda, save location: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: save location wasn't found")
        raise err


def test_encoder_helper(encoder_helper, data_frame, category_lst):
    '''
    test encoder helper
    '''
    try:
        encoder_helper(data_frame, category_lst)
        logging.info(
            "Testing encoder_helper, data frame is correct format: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Testing encoder_helper: dataframe is in incorrect format")
        raise err

    try:
        encoder_helper(data_frame, category_lst)
        logging.info(
            "Testing encoder_helper, category list correct format: SUCCESS")
    except TypeError as err:
        logging.error("Testing encoder_helper: category list must be a list")
        raise err

    try:
        encoder_helper(data_frame, category_lst)
        logging.info("Testing encoder_helper, 'Churn' column found: SUCCESS")
    except KeyError as err:
        logging.error("Testing encoder_helper: Churn' column not found")
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering,
        data_frame,
        response):
    '''
    test perform_feature_engineering
    '''
    try:
        training_data, testing_data, training_labels, testing_labels, response_data = perform_feature_engineering(
            data_frame, response)
        logging.info(
            "Testing perform_feature_engineering, keys found in dataframe: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering, keys not found in dataframe")
        raise err

    try:
        training_data, testing_data, training_labels, testing_labels, response_data = perform_feature_engineering(
            data_frame, response)
        logging.info(
            "Testing perform_feature_engineering, data frame is correct format: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Testing perform_feature_engineering: dataframe is in incorrect format")
        raise err

    try:
        training_data, testing_data, training_labels, testing_labels, response_data = perform_feature_engineering(
            data_frame, response)
        logging.info(
            "Testing perform_feature_engineering, category list correct format: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing perform_feature_engineering: category list must be a list")
        raise err

    try:
        len(training_data)
        len(testing_data)
        len(training_labels)
        len(testing_labels)
        logging.info(
            "Testing perform_feature_engineering, training and testing data and labels are in correct format: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing perform_feature_engineering: training and testing data and labels are not in correct format")
        raise err

    try:
        response_data.head()
        logging.info(
            "Testing perform_feature_engineering, response data is in correct format: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Testing perform_feature_engineering: response data is not correct format")
        raise err

    try:
        assert response_data.shape[0] > 0
        assert response_data.shape[1] > 0
        logging.info(
            "Testing perform_feature_engineering, data frame has rows and columns: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The data frame doesn't appear to have rows and columns")
        raise err

    return training_data, testing_data, training_labels, testing_labels, response_data


def test_train_models(
        train_models,
        training_data,
        testing_data,
        training_labels,
        testing_labels):
    '''
    test train_models
    '''
    try:
        training_labels_preds_lr, testing_labels_preds_lr, lrc_model, training_labels_preds_rf, testing_labels_preds_rf, rfc_model = train_models(
            training_data, testing_data, training_labels, testing_labels)
        len(training_labels_preds_lr)
        len(testing_labels_preds_lr)
        len(training_labels_preds_rf)
        len(testing_labels_preds_rf)
        logging.info(
            "Testing test_train_models, training and testing label predictions are in correct format: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing test_train_models: training and testing label predictions are not correct format")
        raise err

    try:
        lrc_model.predict(training_data)
        rfc_model.predict(training_data)
        logging.info(
            "Testing test_train_models, input data is in correct format: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing test_train_models: input data is not in correct format")
        raise err

    try:
        lrc_model.predict(training_data)
        rfc_model.predict(training_data)
        logging.info(
            "Testing test_train_models, machine learning models are in correct format: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Testing test_train_models: machine learning models are not in correct format")
        raise err


if __name__ == "__main__":
    data_frame = test_import(cls.import_data)
    test_eda(cls.perform_eda, data_frame)
    test_encoder_helper(cls.encoder_helper, data_frame, [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ])

    training_data, testing_data, training_labels, testing_labels, response_data = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        data_frame,
        ['Customer_Age', 'Dependent_count', 'Months_on_book',
         'Total_Relationship_Count', 'Months_Inactive_12_mon',
         'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
         'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
         'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
         'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
         'Income_Category_Churn', 'Card_Category_Churn'])

    test_train_models(
        cls.train_models,
        training_data,
        testing_data,
        training_labels,
        testing_labels)
