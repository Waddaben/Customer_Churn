# Training and analaysing machine learning models for predicting customer churn.
 Author: Wadda du Toit,
 Date: Jan 2022

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This projects is a machine learning pipeline dedicated to classifying customer churn.
The pipeline consists of two main parts:
1. Data analysis and preration 
2. Model training and tuning

In the data analysis part, the data is:
1. The data is uploaded
2. The data is explored through different summaries and representations of distributions 
   and correlations
3. Some of the data is recronstructed in propotion of churn
4. The data is split into training and testing data as well as labels

In the model training and tuning part:
1. A logistical regression model is trained
2. A random forest model is trained using cross validation
3. The results of the models are visualised
4. The impact of the features for the random forest model's classification is analysed
5. The final results of the models are visualised and stored 


## Running Files
How to run code:
1. The first step is to create a conda environment using the following line in the terminal:
> conda create --name <env_name> --requirements.txt
2. The second step is to activate conda environmemt
> conda activate <env_name>
3. The third step run the churn_library.py file with the following command
> ipython .\churn_library.py

What to expect:
1. Data will be uploaded and the details will be displayed in the terminal, along
   with figures showing the distribution of the data
2. The machine learning models will be trained on the data with the ROC curves displayed
   afterwards
3. The impact of the features for the random forest classifier will be displayed
4. The final models and their results will be stored

## Testing the .\churn_library.py file
1. Run the testing file to test all the functions using the following line in the terminal
> ipython churn_script_logging_and_tests.py
2. Go to .\logs\churn_library.logs to review successes and errors

