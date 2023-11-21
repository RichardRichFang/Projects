
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def load_model(model_name):
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model

def main(df, selected_features, selected_missing_indicator_features, full_features, full_missing_indicator_features):
    st.title("Risk Performance Predict")
    feature_dictionary = {
    	'ExternalRiskEstimate': 'Consolidated version of risk markers',
    	'MSinceOldestTradeOpen': 'Months Since Oldest Trade Open',
    	'MSinceMostRecentTradeOpen': 'Months Since Most Recent Trade Open',
    	'AverageMInFile': 'Average Months in File',
    	'NumSatisfactoryTrades': 'Number Satisfactory Trades',
    	'NumTrades60Ever2DerogPubRec': 'Number Trades 60+ Ever',
    	'NumTrades90Ever2DerogPubRec': 'Number Trades 90+ Ever',
    	'PercentTradesNeverDelq': 'Percent Trades Never Delinquent',
    	'MSinceMostRecentDelq': 'Months Since Most Recent Delinquency',
    	'MaxDelq2PublicRecLast12M': 'Max Delq/Public Records Last 12 Months',
    	'MaxDelqEver': 'Max Delinquency Ever',
    	'NumTotalTrades': 'Number of Total Trades (total number of credit accounts)',
    	'NumTradesOpeninLast12M': 'Number of Trades Open in Last 12 Months',
    	'PercentInstallTrades': 'Percent Installment Trades',
    	'MSinceMostRecentInqexcl7days': 'Months Since Most Recent Inq excl 7days',
    	'NumInqLast6M': 'Number of Inq Last 6 Months',
    	'NumInqLast6Mexcl7days': 'Number of Inq Last 6 Months exclude 7days (Excluding the last 7 days removes inquiries that are likely due to price comparision shopping.)',
    	'NetFractionRevolvingBurden' : 'Net Fraction Revolving Burden. This is revolving balance divided by credit limit',
    	'NetFractionInstallBurden': 'Net Fraction Installment Burden. This is installment balance divided by original loan amount',
    	'NumRevolvingTradesWBalance': 'Number Revolving Trades with Balance',
    	'NumInstallTradesWBalance': 'Number Installment Trades with Balance',
    	'NumBank2NatlTradesWHighUtilization': 'Number Bank/Natl Trades w high utilization ratio',
    	'PercentTradesWBalance': 'Percent Trades with Balance'
    }
    model_options = ["Simple Version", "Complete Version"]
    model_name = st.sidebar.selectbox("	Choose Model：", model_options)

    if model_name == "Simple Version":
        model = load_model('model_filter.pkl')
        model_features = selected_features + selected_missing_indicator_features
        with open('scaler_filter.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        model = load_model('model_full.pkl')
        model_features = full_features + full_missing_indicator_features
        with open('scaler_full.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    user_input = {}
    box_op = ['Yes, Data is Available','No Bureau Record or No Investigation','No Usable/Valid Trades or Inquiries', 'Condition not Met (e.g. No Inquiries, No Delinquencies)']
    box_value = [0,-9,-8,-7]

    for feature in model_features:
        if not feature.endswith("=-7") and not feature.endswith("=-8"):
        	st.sidebar.markdown("""---""")
        	choice = st.sidebar.selectbox('Do you have data for ' + feature_dictionary[feature] + '?',box_op)
        	if choice == 'Yes, Data is Available':
        		if feature.startswith('Percent'):
        			value = st.sidebar.slider(f"{feature}：", 0, 100)
        			user_input[feature] = value
        		elif feature == 'MaxDelq2PublicRecLast12M':
        			MaxDelq2PublicRecLast12M_op = ['derogatory comment','120+ days delinquent','90 days delinquent','60 days delinquent','30 days delinquent',
        					'unknown delinquency','unknown delinquency','current and never delinquent','all other','all other']
        			choice_MaxDelq2PublicRecLast12M = st.sidebar.selectbox("Max Delq/Public Records Last 12 Months", MaxDelq2PublicRecLast12M_op)
        			value = MaxDelq2PublicRecLast12M_op.index(choice_MaxDelq2PublicRecLast12M)
        			user_input[feature] = value
        		elif feature == 'MaxDelqEver':
        			MaxDelqEver_op = ['No such value','derogatory comment','120+ days delinquent','90 days delinquent','60 days delinquent','30 days delinquent',
        					'unknown delinquency','current and never delinquent','all other']
        			choice_MaxDelqEver = st.sidebar.selectbox("Max Delinquency Ever", MaxDelqEver_op)
        			value = MaxDelqEver_op.index(choice_MaxDelqEver) + 1
        			user_input[feature] = value
        		else:
        			value = st.sidebar.number_input(f"{feature}：", value=0.0)
        			user_input[feature] = value
        	else:
        		value = box_value[box_op.index(choice)]
        		user_input[feature] = value
        st.write(f'{feature} is {value}')
    for feature in model_features:
        if feature not in user_input:
            if feature.endswith("=-7"):
                user_input[feature] = 1 if user_input[feature[:-3]] == -7 else 0
            elif feature.endswith("=-8"):
                user_input[feature] = 1 if user_input[feature[:-3]] == -8 else 0

    if st.button("Predict!"):
        input_df = pd.DataFrame([user_input.values()], columns=model_features)
            
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        st.success(f"Prediction Outcome：{'Good' if prediction[0] == 1 else 'Bad'}")
        st.write(f"Prob：{prediction_proba[0][int(prediction[0])]:.2f}")

    
if __name__ == "__main__":
    df = pd.read_csv("heloc_dataset_v1.csv")
    selected_features = ['ExternalRiskEstimate', 'AverageMInFile', 'NumSatisfactoryTrades',
                         'PercentTradesNeverDelq', 'PercentInstallTrades',
                         'MSinceMostRecentInqexcl7days', 'NumInqLast6M','NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance',
                         'NumBank2NatlTradesWHighUtilization']
    selected_missing_indicator_features = ['MSinceMostRecentInqexcl7days=-7', 'MSinceMostRecentInqexcl7days=-8',
                                           'NetFractionRevolvingBurden=-8', 'NumRevolvingTradesWBalance=-8',
                                           'NumBank2NatlTradesWHighUtilization=-8']

    
    full_features = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen',
                     'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades',
                     'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                     'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
                     'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
                     'NumTradesOpeninLast12M', 'PercentInstallTrades',
                     'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
                     'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
                     'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
                     'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    full_missing_indicator_features = ['MSinceMostRecentDelq=-7', 'MSinceMostRecentInqexcl7days=-7',
                                       'MSinceOldestTradeOpen=-8', 'MSinceMostRecentDelq=-8',
                                       'MSinceMostRecentInqexcl7days=-8', 'NetFractionRevolvingBurden=-8',
                                       'NetFractionInstallBurden=-8', 'NumRevolvingTradesWBalance=-8',
                                       'NumInstallTradesWBalance=-8', 'NumBank2NatlTradesWHighUtilization=-8',
                                       'PercentTradesWBalance=-8']

    main(df, selected_features, selected_missing_indicator_features, full_features, full_missing_indicator_features)



# In[ ]:




