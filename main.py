import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.combine import SMOTEENN
import xgboost
from xgboost import XGBClassifier
import streamlit as st
import numpy as np

df = pd.read_csv("train.csv")

df['y'] = df['y'].map({'yes' : 1, 'no' : 0})
df['job'] = df['job'].map({'blue-collar':0, 'entrepreneur':1, 'housemaid':2, 'services':3, 'technician':4, 'self-employed':5, 'admin.':6, 'management':7, 'unemployed':8, 'retired':9, 'student':10})
df['marital'] = df['marital'].map({'married':0,'divorced':1,'single':2})
df['education_qual'] = df['education_qual'].map({'primary':0,'secondary':1,'tertiary':2})
df['call_type'] = df['call_type'].map({'unknown':0,'telephone':1,'cellular':2})
df['mon'] = df['mon'].map({'may':0, 'jul':1,  'jan':2, 'nov':3, 'jun':4, 'aug':5, 'feb':6, 'apr':7, 'oct':8, 'sep':9, 'dec':10, 'mar':11})
df['prev_outcome'] = df['prev_outcome'].map({'unknown':0,'failure':1,'other':2,'success':3})

x = df[['age', 'job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'dur', 'num_calls', 'prev_outcome']].values
y = df['y'].values


x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.25,random_state=1)

from imblearn.combine import SMOTEENN
smt = SMOTEENN(sampling_strategy='all')
x_smt_tr,y_smt_tr = smt.fit_resample(x_tr,y_tr)

xgb_model = XGBClassifier(colsample_bytree = 0.4, learning_rate = 0.2, n_estimators = 100)
xgb_model.fit(x_smt_tr,y_smt_tr)

xgb_model.save_model('xgb_model.json')

model = XGBClassifier()
model.load_model('xgb_model.json')

@st.cache

def predict(age, job, marital, education_qual, call_type, day, mon, dur, num_calls, prev_outcome):
    if job == 'blue-collar':
        job = 0
    elif job == 'entrepreneur':
        job = 1
    elif job == 'housemaid':
        job = 2
    elif job == 'services':
        job = 3
    elif job == 'technician':
        job = 4
    elif job == 'self-employed':
        job = 5
    elif job == 'admin.':
        job = 6
    elif job == 'management':
        job = 7
    elif job == 'unemployed':
        job = 8
    elif job == 'retired':
        job = 9
    elif job == 'student':
        job = 10

    if marital == 'married':
        marital = 0
    elif marital == 'divorced':
        marital = 1
    elif marital == 'single':
        marital = 2

    if education_qual == 'primary':
        education_qual = 0
    elif education_qual == 'secondary':
        education_qual = 1
    elif education_qual == 'tertiary':
        education_qual = 2

    if call_type == 'unknown':
        call_type = 0
    elif call_type == 'telephone':
        call_type = 1
    elif call_type == 'cellular':
        call_type = 2

    if mon == 'may':
        mon = 0
    elif mon == 'jul':
        mon = 1
    elif mon == 'jan':
        mon = 2
    elif mon == 'nov':
        mon = 3
    elif mon == 'jun':
        mon = 4
    elif mon == 'aug':
        mon = 5
    elif mon == 'feb':
        mon = 6
    elif mon == 'apr':
        mon = 7
    elif mon == 'oct':
        mon = 8
    elif mon == 'sep':
        mon = 9
    elif mon == 'dec':
        mon = 10
    elif mon == 'mar':
        mon = 11

    if prev_outcome == 'unknown':
        prev_outcome = 0
    elif prev_outcome == 'failure':
        prev_outcome = 1
    elif prev_outcome == 'other':
        prev_outcome = 2
    elif prev_outcome == 'success':
        prev_outcome = 3

    prediction = model.predict(np.array([age, job, marital, education_qual, call_type, day, mon, dur, num_calls, prev_outcome]).reshape(1,-1))
    return prediction

st.title('Customer Conversion Predictor')

with st.expander('About the App'):
    st.markdown(
        '<div style="text-align: justify;">This app is a Customer Conversion Predictor that can predict whether a client will subscribe to the insurance based on their age, '
        'job, marital status, education qualification. This app also predicts based on the details collected from customers like call type, day of the month, '
        'duration of the call, number of calls made, previous call outcome by the sales / telemarketing representatives or sales manager of the '
        'insurance company.</div>', unsafe_allow_html=True)
    st.write(" ")
    st.markdown(
        '<div style="text-align: justify;">Once the sales representative filled all the details of a customer and click [Predict] button, This app will predict whether the customer '
        'subscribe to insurance or not. If the prediction says [Yes], It means, the customer will buy the policy for sure. If the prediction says [No],'
        ' It means, the customer will not buy the policy. By leveraging machine learning capabilities, the employees of the insurance company can gain '
        'predictive insights into customer conversion by comparing actual and predicted results.</div>',
        unsafe_allow_html=True)
    st.write(" ")

st.header('Please fill the following details:')

with st.form('Please fill the following details:'):
    age = st.number_input('Age', min_value = 18, max_value = 70, value = 18)
    job = st.selectbox('Job', ['student', 'housemaid', 'unemployed', 'entrepreneur', 'self-employed', 'retired', 'services', 'admin.', 'technician', 'management', 'blue-collar'])
    marital = st.selectbox('Marital Status', ['married', 'divorced', 'single'])
    education_qual = st.selectbox('Education Qualification', ['primary', 'secondary', 'tertiary'])
    call_type = st.selectbox('Call Type', ['unknown', 'telephone', 'cellular'])
    day = st.number_input('Day of the Month', min_value = 1, max_value = 31, value = 1)
    mon = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    dur = st.number_input('Duration of Call in seconds', min_value = 0, max_value = 640, value = 0)
    num_calls = st.number_input('Number of Calls', min_value = 1, max_value = 6, value = 1)
    prev_outcome = st.selectbox("Previous Call's Outcome", ['unknown', 'failure', 'other', 'success'])

    submitted = st.form_submit_button('Predict')
    if submitted:
        result = predict(age, job, marital, education_qual, call_type, day, mon, dur, num_calls, prev_outcome)
        if result == ([1]):
            st.success('Yes')
        if result == ([0]):
            st.error('No')
