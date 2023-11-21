
# coding: utf-8

# ## Load Library

# In[1]:


import pandas as pd
import numpy as np
import warnings
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel


# ## Load Data 

# In[2]:


df = pd.read_csv("heloc_dataset_v1.csv")


# In[3]:


#df = df[df != -9].dropna()
#df.replace([-7, -8], np.nan, inplace=True)
#mean_values = df.mean()
#df.fillna(mean_values, inplace=True)


# ## Inspect Data & Preprocessing

# ### Check label distribution

# In[3]:


df['RiskPerformance'] = df['RiskPerformance'].replace({'Bad': 0, 'Good': 1})
df_filter = df.copy()
df.columns


# In[4]:


verified_categories = df['RiskPerformance'].unique()
print(verified_categories)


# In[5]:


df.columns


# ### Missing data processing

# In[6]:


df = df[df['ExternalRiskEstimate'] != -9]

minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(df)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(df)

col_names_minus_7 = df.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = df.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
columns_all = df.columns.values.tolist() + col_names_minus_7 + col_names_minus_8

do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
 
pipe_example = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])


df = pipe_example.fit_transform(df)
df= pd.DataFrame(df, columns=columns_all)


# In[7]:


df.info()


# In[8]:


# minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(df1) # notice the -8
# minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(df1) # notice the -8
# arr2_t = minus_8_indicator_transformer.transform(df1)
# arr2_t

#union = FeatureUnion([("do nothing", do_nothing_imputer),
                     # ("missing_minus_7", MissingIndicator(missing_values=-7, features='missing-only')),
                      #("missing_minus_8", MissingIndicator(missing_values=-8, features='missing-only'))])
#arr1_extended = union.fit_transform(df1)


# ### Class Mean Value Inspection

# In[9]:


numeric_cols = df.select_dtypes(include='number').columns
grouped = df.groupby('RiskPerformance')[numeric_cols].mean()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(grouped)


# ### Correlation Inspection

# In[10]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.corr()


# ## Model part

# ### Diverse model experiments

# In[11]:


warnings.filterwarnings('ignore')

X =df.iloc[:, 1:]
y = df['RiskPerformance']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classifiers = [
    ('Logistic Regression', LogisticRegression(max_iter=500)),
    ('Support Vector Machines', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Trees', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting Machines', GradientBoostingClassifier()),
    ('XGBoost', xgb.XGBClassifier()),
    ('LightGBM', lgb.LGBMClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Neural Networks', MLPClassifier(max_iter=500))
]

for name, clf in classifiers:
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    print(f'{name}: Accuracy = {scores.mean():.4f}')


# ### Lasso Feature Selection

# In[12]:


X = df.iloc[:, 1:24]
y = df['RiskPerformance']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)

sfm = SelectFromModel(lr, threshold=0.1)
X_new = sfm.fit_transform(X_scaled, y)

support = sfm.get_support()
print(support)

selected_features = X.columns[support]
print(selected_features)


# In[13]:


df_filter = df.copy()
df_filter = df_filter[['ExternalRiskEstimate', 'AverageMInFile', 'NumSatisfactoryTrades',
       'PercentTradesNeverDelq', 'PercentInstallTrades',
       'MSinceMostRecentInqexcl7days', 'NumInqLast6M',
       'NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance',
       'NumBank2NatlTradesWHighUtilization']]


# In[14]:


df_filter = df_filter[df_filter['ExternalRiskEstimate'] != -9]

minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(df_filter)
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(df_filter)

col_names_minus_7 = df_filter.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
col_names_minus_8 = df_filter.columns.values[minus_8_indicator_transformer.features_].tolist() 
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
columns_all = df_filter.columns.values.tolist() + col_names_minus_7 + col_names_minus_8

do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')

feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
 
pipe_example = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])


df_filter = pipe_example.fit_transform(df_filter)
df_filter= pd.DataFrame(df_filter, columns=columns_all)


# In[16]:


df_filter.columns


# ### Grid search & Filtered features

# In[17]:


X = df_filter.iloc[:, 0:]
y = df['RiskPerformance']
scaler_filter = StandardScaler()
X_scaled_filter = scaler_filter.fit_transform(X)

with open('scaler_filter.pkl', 'wb') as f:
    pickle.dump(scaler_filter, f)


param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 
              'penalty': ['l1', 'l2'],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'class_weight': ['balanced', None],
              'max_iter': [200, 500],
              'multi_class': ['ovr', 'multinomial', 'auto'],
              'dual': [False, True]}

lr_combined = LogisticRegression()

grid_search = GridSearchCV(estimator=lr_combined, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_scaled_filter, y)

print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))


# ### BestModel （Select Feature

# In[21]:


model_filter = LogisticRegression(C=10, penalty='l2', solver='liblinear', class_weight='balanced', dual=True, max_iter=500, multi_class='ovr')
scores = cross_val_score(model_filter, X_scaled_filter, y, cv=5, scoring='accuracy')
print('Accuracy:', scores.mean())
model_filter.fit(X_scaled_filter, y)


# ### All Feature

# In[22]:


X = df.iloc[:, 1:]
y = df['RiskPerformance']

scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

with open('scaler_full.pkl', 'wb') as f:
    pickle.dump(scaler_full, f)

    
param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 
              'penalty': ['l1', 'l2'],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'class_weight': ['balanced', None],
              'max_iter': [200],
              'multi_class': ['ovr', 'multinomial', 'auto'],
              'dual': [False, True]}

lr_combined = LogisticRegression()

grid_search = GridSearchCV(estimator=lr_combined, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_scaled, y)

print("Best parameters: ", grid_search.best_params_)
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))


# In[24]:


model_full = LogisticRegression(C=1, penalty='l1', solver='saga', class_weight=None, dual=False, max_iter=200, multi_class='ovr')
scores = cross_val_score(model_full, X_scaled, y, cv=5, scoring='accuracy')
print('Accuracy:', scores.mean())
model_full.fit(X_scaled, y)


# In[25]:


with open('model_filter.pkl', 'wb') as f:
    pickle.dump(model_filter, f)

with open('model_full.pkl', 'wb') as f:
    pickle.dump(model_full, f)


# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer, MissingIndicator
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.feature_selection import SelectFromModel
# import streamlit as st
# 
# 
# def preprocess_data(df):
#     df['RiskPerformance'] = df['RiskPerformance'].replace({'Bad': 0, 'Good': 1})
#     
#     df = df[df['ExternalRiskEstimate'] != -9]
# 
#     minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(df)
#     minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(df)
# 
#     col_names_minus_7 = df.columns.values[minus_7_indicator_transformer.features_].tolist() 
#     col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
#     col_names_minus_8 = df.columns.values[minus_8_indicator_transformer.features_].tolist() 
#     col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
#     columns_all = df.columns.values.tolist() + col_names_minus_7 + col_names_minus_8
# 
#     do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')
# 
#     feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
#                                       ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
#                                       ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
#  
#     pipe_example = Pipeline([("expand features", feature_expansion), 
#                      ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
#                      ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])
# 
# 
#     df = pipe_example.fit_transform(df)
#     df = pd.DataFrame(df, columns=columns_all)
# 
#     X = df.iloc[:, 1:24]
#     y = df['RiskPerformance']
# 
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
# 
#     lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
# 
#     sfm = SelectFromModel(lr, threshold=0.1)
#     X_new = sfm.fit_transform(X_scaled, y)
# 
#     support = sfm.get_support()
# 
#     selected_features = X.columns[support]
# 
#     X_selected = pd.DataFrame(X_new, columns=selected_features)  
#     X_others = df.iloc[:, 24:36]  
#     X_combined = pd.concat([X_selected, X_others], axis=1) 
# 
#     return X_combined, y, X_scaled
# 
# 
# 
# @st.cache(suppress_st_warning=True)
# def load_data():
# 
#     df = pd.read_csv("heloc_dataset_v1.csv")
# 
#     X_combined, y, X_scaled = preprocess_data(df)
#     
#     return X_combined, y, X_scaled
# 
# def main():
# 
#     X_combined, y, X_scaled = load_data()
# 
# 
#     st.title("Credit Risk Prediction")
# 
#     model_option = st.selectbox("Select a model", ("Full Feature Model", "Filtered Feature Model"))
# 
#     st.subheader("Please enter feature values (if you don't know a value, use the mean or default value)")
# 
#     input_features = []
#     for feature in X_combined.columns:
#         input_features.append(
#             st.number_input(
#                 f"{feature} (Mean: {X_combined[feature].mean():.2f})", value=X_combined[feature].mean()
#             )
#         )
# 
#     if st.button("Predict"):
#         input_data = np.array(input_features).reshape(1, -1)
#         
#         if model_option == "Full Feature Model":
#             model = LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight=None, dual=False, max_iter=200, multi_class='ovr')
#             model.fit(X_scaled, y)
#             prediction = model.predict(input_data)
#         elif model_option == "Filtered Feature Model":
#             model = LogisticRegression(C=10.0, penalty='l2', solver='liblinear', class_weight=None, dual=True, max_iter=200, multi_class='auto')
#             model.fit(X_combined, y)
#             prediction = model.predict(input_data)
# 
#         st.write("Prediction result:")
#         st.write("Risk Performance: " + ("Good" if prediction[0] == 1 else "Bad"))
# 
# if __name__ == "__main__":
#     main()

# In[ ]:




