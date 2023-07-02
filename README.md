# Customer_Conversion_Prediction_Project

## Introduction

Customer_Conversion_Prediction_Project aims to predict customer conversion for an insurane company. Using historical data of the list of customers telecalled by sales team of the insurance company, a machine learning model is developed that can identify which type of customers are more likely to avail an insurance

The project aims to help the insurance company identify the customers that are most likely to convert, so that they can be targeted via call and the cost of telephonic marketing campaigns can be reduced

The historical sales data provided will be used to train and evaluate the performance of the machine learning models. The analysis of the model will be done to identify the important factors that contribute towards the conversion and the AUROC metric will be used to evaluate the model's performance. 

The main objective of the project is to develop an accurate and efficient model that can aid the insurance company in improving its sales conversion rate and reducing marketing costs

The built ML model is also deployed in streamlit. This streamlit app is a customer conversion predicor where, when all the details of the customer is filled by the sales / telemarketing representatives of the insurance company, The app will predict 'Yes' if the customer is likely to take insurance and predicts 'No' if the customer will turn down the offer

APP LINK: https://customerconversionpredictionproject-kp15xgyikzd.streamlit.app/

## Project Approach

### 1. Importing and Installing Necessary Libraries
I have imported necessary libraries like numpy, pandas, sklearn, imblearn, seaborn, matplotlib, etc to use in the code block

### 2. Insights of the Data
I have loaded the dataset raw_data.csv and converted it to a dataframe using pandas. After loading the data, I took insights of all the features and target and explained in detail about each feature

### 3. Generic Cleaning of Data
I have used functions like null_values, dtypes, drop_duplicates to do general overall cleaning of the data

### 4. Column Wise Cleaning 
For all the numeric features in the data, I've used 'describe' function to view the summary statistics of the data. Outliers in the column is detected using IQR approach and then it is clipped to q1 and q3 range  

For all the categorical features and targert, the unknown category of some of the features have been changed to the mode value of that feature

### 5. Final Checks
After cleaning all the features and target column wise, once again, I have done the stpes mentioned in the generic cleaning in order to ensure well cleaned data is used

### 6. Exploratory Data Analysis
The following data analysis has been done using seaborn and matplotlib

1. Countplot of the Distribution of Categorical Features in the Data
2. Boxplot of the Distribution of Continuous Features in the Data
3. Countplot of the Distribution of Target Variable in the Data
4. Feature v/s Target Plot for Categorical Columns
5. Feature v/s Target Plot for Continuous Columns

By comparing the results of countplot and feature v/s target plot of categorical columns, the insights of the plot are written and also I have given some suggestion to the insurance company based on the results arrived

The same is done to the continuous features also by comparing countplot and feature v/s target plot of continuous columns

By seeing the countplot of the distribution of target variable, it is clearly visible that this data is an imbalnced data

![image](https://github.com/Anitha-K-0711/Customer_Conversion_Prediction_Project/assets/115402011/17d07cc5-72d6-4966-aefe-ea97c2a47ecf)

### 7. Encoding Categorical and Target Columns
All the categorical and target columns are label encoded

The cleaned and encoded data is after this step is saved and downloaded as train.csv to use in the model deployment

### 8. Splitting the Data
Using train_test_split from model_selection package of sklearn library, I have split the data into x_train, x_test, y_train and y_test. 75% of the data goes to train and 25% of data goes to test

### 9. Balancing the Data
Using SMOTEENN technique from imblearn library, I have balanced the data 

SMOTEENN combines SMOTE and Edited Nearest Neighbours(ENN) techniques. SMOTEENN performs undersampling and oversampling at the same time

### 10. Fit ML Models
I have used 6 classification models and trained the data. The 6 models are,

1. Logistic Regression
2. K-Nearest Neigbors
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost

Hyper-parameter tuning for each model except Logistic Regression has been done and best value is found out on hit and trial and cross validation method. With this best hyper-parameter value, the data is trained to get the highest accuracy

All the models have been evaluated using the metric AUROC score

Out of the 6 Models, with the highest AUROC score of 0.91, XGBoost is the best fit model

![image](https://github.com/Anitha-K-0711/Customer_Conversion_Prediction_Project/assets/115402011/9a2200fc-5952-4574-9e80-b1342bc6ed99)

### 11. Feature Importance 
The top 3 important features contributing towards the model are,

1. Duration of the call in seconds
2. call_type
3. month

![image](https://github.com/Anitha-K-0711/Customer_Conversion_Prediction_Project/assets/115402011/19c83dd9-7bfc-4105-afdc-9aab17f0694d)

Refer customer_conversion_prediction.ipynb for the code block of the above steps

## Deployment

I have deployed the model using streamlit

This app is a Customer Conversion Predictor that can predict whether a client will subscribe to the insurance based on their age, job, marital status, education qualification. This app also predicts based on the details collected from customers like call type, day of the month, duration of the call, number of calls made, previous call outcome by the sales / telemarketing representatives or sales manager of the insurance company

Once the sales representative filled all the details of a customer and click [Predict] button, This app will predict whether the customer subscribe to insurance or not. If the prediction says [Yes], It means, the customer will buy the policy for sure. If the prediction says [No], It means, the customer will not buy the policy. By leveraging machine learning capabilities, the employees of the insurance company can gain predictive insights into customer conversion by comparing actual and predicted results

Refer main.py to view the code block for app deployment

App Link: https://customerconversionpredictionproject-kp15xgyikzd.streamlit.app/

## Further Scope

The current project has successfully built and evaluated a machine learning model to predict whether a customer will subscribe to an insurance policy. However there is still room for improvement and further scope in this project which includes,

1. Building a ML model with most important features: The data can be further trained with the ML models with only most important features of the data to check whether the results are getting fine tuned
2. Exploratory Data Analysis (EDA): In detail EDA is further required to the data in order to further understand and train the models to the data
3. Deployment: The plots of each feature and target need to be deployed in the streamlit app so that, the sales persons of the insurance company will have a better visualized ideas when approaching a customer
4. Model Comparison: In addition to the models evaluated in this project, other classification models could also be implemented and compared to identify the best performing model for this problem
5. Regular Maintenance: As the company's customer base grows and changes, the model's performance might degrade. Regular monitoring and maintenance of the model are necessary to ensure it continues to perform effectively

Overall, these enhancements and improvements could help the insurance company to optimize its outreach efforts and improve the success rate of selling insurance policies to potential customers
