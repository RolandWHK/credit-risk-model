# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 01:38:58 2026

@author: otnie
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression
import scipy.stats as stat


st.set_page_config(page_title="Credit Risk EL Framework", layout="wide")
#%%
# --- STEP 1: CUSTOM CLASS DEFINITIONS ---
# You must include these because your .sav files were created with them.
class LogisticRegression_with_p_values:
  def __init__(self,*args,**kwargs):
    self.model = LogisticRegression(*args,**kwargs)

  def fit(self,X,y):
    self.model.fit(X,y)
    denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
    denom = np.tile(denom,(X.shape[1], 1)).T
    F_ij = np.dot((X / denom).T,X)
    Cramer_Rao = np.linalg.inv(F_ij)
    sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
    z_scores = self.model.coef_[0] / sigma_estimates
    p_values = [stat.norm.sf(abs(X)) * 2 for X in z_scores]
    self.coef_ = self.model.coef_
    self.intercept_ = self.model.intercept_
    self.p_values = p_values

class LinearRegression(LinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=1, positive=False):
      super().__init__(fit_intercept=fit_intercept, 
                        copy_X=copy_X, 
                        n_jobs=n_jobs, 
                        positive=positive)
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self
#%%
# --- STEP 2: LOAD MODELS ---
@st.cache_resource
def load_model1():
    with open("models/pd_model.sav", 'rb') as file:
        return pickle.load(file)
@st.cache_resource    
def load_model2():
    with open("models/lgd_model_stage_1.sav", 'rb') as file:
        return pickle.load(file)
@st.cache_resource    
def load_model3():
    with open("models/lgd_model_stage_2.sav", 'rb') as file:
        return pickle.load(file)
@st.cache_resource    
def load_model4():
    with open("models/ead_model.sav", 'rb') as file:
        return pickle.load(file)
    
model_wrapper1 = load_model1()
model1 = model_wrapper1.model

model_wrapper2 = load_model2()
# model2 = model_wrapper2.model

model_wrapper3 = load_model3()
#model3 = model_wrapper3.model

model_wrapper4 = load_model4()
# model4 = model_wrapper4.model
    
# def load_all_models():
#     pd_model = pickle.load(open('C:/Users/otnie/Documents/CR/Projects/PDmodel/pd_model.sav', 'rb'))
#     lgd_s1 = pickle.load(open('C:/Users/otnie/Documents/CR/Projects/PDmodel/lgd_model_stage_1.sav', 'rb'))
#     lgd_s2 = pickle.load(open('C:/Users/otnie/Documents/CR/Projects/PDmodel/lgd_model_stage_2.sav', 'rb'))
#     ead_model = pickle.load(open('C:/Users/otnie/Documents/CR/Projects/PDmodel/ead_model.sav', 'rb'))
#     return pd_model, lgd_s1, lgd_s2, ead_model

# pd_m, lgd1, lgd2, ead_m = load_all_models()




#%%

feature_columns1 = ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:w',
'term:36',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'acc_now_delinq:>=1',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86']

ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

# feature_columns2 = feature_columns1.remove(ref_categories)

feature_colunmns_lgd_ead = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F',
       'home_ownership:MORTGAGE', 'home_ownership:NONE',
       'home_ownership:OTHER', 'home_ownership:OWN',
       'verification_status:Not Verified',
       'verification_status:Source Verified', 'purpose:car',
       'purpose:debt_consolidation', 'purpose:educational',
       'purpose:home_improvement', 'purpose:house',
       'purpose:major_purchase', 'purpose:medical', 'purpose:moving',
       'purpose:other', 'purpose:renewable_energy',
       'purpose:small_business', 'purpose:vacation', 'purpose:wedding',
       'initial_list_status:w', 'term_int', 'emp_length_int',
       'mths_since_issue_d', 'mths_since_earliest_cr_line', 'funded_amnt',
       'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'total_acc',
       'acc_now_delinq', 'total_rev_hi_lim']



#%%

def preprocess_inputs_pd(user_data, feature_columns):
    # 1. Initialize a DataFrame with 104 zeros (matching your model features)
    feature_columns = model1.feature_names_in_
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # 2. Categorical Logic (Non-References)
    # If the user selects a non-reference, set that column to 1.
    # If they select a reference (e.g., Grade G), all columns for that group stay 0.
    
    grade_val  = user_data['grade']
    if grade_val == 'A': input_df['grade:A'] = 1
    elif grade_val == 'B': input_df['grade:B'] = 1
    elif grade_val == 'C': input_df['grade:C'] = 1
    elif grade_val == 'D': input_df['grade:D'] = 1
    elif grade_val == 'E': input_df['grade:E'] = 1
    elif grade_val == 'F': input_df['grade:F'] = 1
    # elif grade_val == 'G': input_df['grade:G'] = 1
    
    home_val = user_data['home_ownership']
    if home_val == 'OWN': input_df['home_ownership:OWN'] = 1 
    elif home_val == 'MORTGAGE': input_df['home_ownership:MORTGAGE'] = 1
    # elif home_val in ['RENT', 'OTHER', 'NONE', 'ANY']: input_df['home_ownership:RENT_OTHER_NONE_ANY'] = 1 
    
    state = user_data['addr_state']
    if state in ['NM', 'VA']: input_df['addr_state:NM_VA'] = 1
    elif state == 'NY': input_df['addr_state:NY'] = 1
    elif state in ['OK', 'TN', 'MO', 'LA', 'MD', 'NC']: input_df['addr_state:OK_TN_MO_LA_MD_NC'] = 1
    elif state == 'CA': input_df['addr_state:CA'] = 1
    elif state in ['UT', 'KY', 'AZ', 'NJ']: input_df['addr_state:UT_KY_AZ_NJ'] = 1
    elif state in ['AR', 'MI', 'PA', 'OH', 'MN']: input_df['addr_state:AR_MI_PA_OH_MN'] = 1
    elif state in ['RI', 'MA', 'DE', 'SD', 'IN']: input_df['addr_state:RI_MA_DE_SD_IN'] = 1
    elif state in ['GA', 'WA', 'OR']: input_df['addr_state:GA_WA_OR'] = 1
    elif state in ['WI', 'MT']: input_df['addr_state:WI_MT'] = 1
    elif state == 'TX': input_df['addr_state:TX'] = 1
    elif state in ['IL', 'CT']: input_df['addr_state:IL_CT'] = 1
    elif state in ['KS', 'SC', 'CO', 'VT', 'AK', 'MS']: input_df['addr_state:KS_SC_CO_VT_AK_MS'] = 1
    elif state in ['WV', 'NH', 'WY', 'DC', 'ME', 'ID']: input_df['addr_state:WV_NH_WY_DC_ME_ID'] = 1
    # elif state in ['ND', 'NE', 'IA', 'NV', 'FL', 'HI', 'AL']: input_df['addr_state:ND_NE_IA_NV_FL_HI_AL'] = 1
    # Reference: 'ND_NE_IA_NV_FL_HI_AL' (stay 0)
    
    purp = user_data['purpose']
    if purp == 'credit_card': input_df['purpose:credit_card'] = 1
    elif purp == 'debt_consolidation': input_df['purpose:debt_consolidation'] = 1
    elif purp in ['other', 'medical', 'vacation']: input_df['purpose:oth__med__vacation'] = 1
    elif purp in ['major_purchase', 'car', 'home_improvement']: input_df['purpose:major_purch__car__home_impr'] = 1
    # elif purp in ['education', 'sm_b', 'wedding', 'ren_en', 'moving', 'house']: input_df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = 1

    # purpose:educ__sm_b__wedd__ren_en__mov__house
    
    verif = user_data['verification']
    if verif == 'Not Verified': input_df['verification_status:Not Verified'] = 1
    elif verif =='Source Verified': input_df['verification_status:Source Verified'] = 1
    # elif verif == 'Verified': input_df['verification_status:Verified'] = 1 
    
    initial_list = user_data['initial_list_status']
    if initial_list == 'w': input_df['initial_list_status:w'] = 1
    # elif initial_list == 'f': input_df['initial_list_status:f'] = 1 
    
    
    # 3. Continuous Variable Logic (Binning)
    # We must replicate the 'Fine Classing' and 'Weight of Evidence' bins exactly.
    
    # Interest Rate
    ir = user_data['int_rate']
    if ir < 9.548: input_df['int_rate:<9.548'] = 1
    elif 9.548 <= ir < 12.025: input_df['int_rate:9.548-12.025'] = 1
    elif 12.025 <= ir < 15.74: input_df['int_rate:12.025-15.74'] = 1
    elif 15.74 <= ir < 20.281: input_df['int_rate:15.74-20.281'] = 1
    # elif ir > 20.281: input_df['int_rate:>20.281'] = 1
    # Note: >20.281 is your reference category, so if ir > 20.281, all stay 0.

    # Annual Income
    ai = user_data['annual_inc']
    # if ai < 20000: input_df['annual_inc:<20K'] = 1
    if 30000 < ai <= 40000: input_df['annual_inc:20K-30K'] = 1
    elif 40000 < ai <= 50000: input_df['annual_inc:30K-40K'] = 1
    elif 50000 < ai <= 60000: input_df['annual_inc:40K-50K'] = 1
    elif 60000 < ai <= 70000: input_df['annual_inc:50K-60K'] = 1
    elif 60000 < ai <= 70000: input_df['annual_inc:60K-70K'] = 1
    elif 70000 < ai <= 80000: input_df['annual_inc:70K-80K'] = 1
    elif 80000 < ai <= 90000: input_df['annual_inc:80K-90K'] = 1
    elif 90000 < ai <= 10000: input_df['annual_inc:90K-100K'] = 1
    elif 100000 < ai <= 120000: input_df['annual_inc:100K-120K'] = 1
    elif 120000 < ai <= 140000: input_df['annual_inc:120K-140K'] = 1
    elif ai > 140000: input_df['annual_inc:>140K'] = 1
    # Reference: <20K

    # Debt-to-Income (DTI)
    dti = user_data['dti']
    if dti <= 1.4: input_df['dti:<=1.4'] = 1
    elif 1.4 < dti <= 3.5: input_df['dti:1.4-3.5'] = 1
    elif 3.5 < dti <= 7.7: input_df['dti:3.5-7.7'] = 1
    elif 7.7 < dti <= 10.5: input_df['dti:7.7-10.5'] = 1
    elif 10.5 < dti <= 16.1: input_df['dti:10.5-16.1'] = 1
    elif 16.1 < dti <= 20.3: input_df['dti:16.1-20.3'] = 1
    elif 20.3 < dti <= 21.7: input_df['dti:20.3-21.7'] = 1
    elif 21.7 < dti <= 22.4: input_df['dti:21.7-22.4'] = 1
    elif 22.4 < dti <= 35: input_df['dti:22.4-35'] = 1
    # elif dti > 35: input_df['dti:>35'] = 1
    # Reference: >35

    # Months Since Last Delinquency
    m_delinq = user_data['mths_since_last_delinq']
    if m_delinq == -1: # Use -1 to represent 'Missing' in your UI
        input_df['mths_since_last_delinq:Missing'] = 1
    # elif 0 <= m_delinq <= 3: input_df['mths_since_last_delinq:0-3'] = 1
    elif 4 <= m_delinq <= 30: input_df['mths_since_last_delinq:4-30'] = 1
    elif 31 <= m_delinq <= 56: input_df['mths_since_last_delinq:31-56'] = 1
    elif m_delinq >= 57: input_df['mths_since_last_delinq:>=57'] = 1
    # Reference: 0-3
    
    #Months Since Last Record
    m_record = user_data['mths_since_last_record']
    if m_record == -1:
        input_df['mths_since_last_record:Missing'] = 1
    # elif 0 <= m_record <= 2: input_df['mths_since_last_record:0-2']
    elif 3 <= m_record <= 20: input_df['mths_since_last_record:3-20'] = 1
    elif 21 <= m_record <= 31: input_df['mths_since_last_record:21-31'] = 1
    elif 32 <= m_record <= 80: input_df['mths_since_last_record:32-80'] = 1
    elif 81 <= m_record <= 86: input_df['mths_since_last_record:81-86'] = 1
    elif 86 < m_record: input_df['mths_since_last_record:>86'] = 1 
    # elif 0 <= m_record <= 2: input_df['mths_since_last_record:0-2']
    # reference category
    
    
    # Acc Now Delinq
    m_acc_d = user_data['acc_now_delinq']
    if m_acc_d >= 1: input_df['acc_now_delinq:>=1'] = 1
    # elif m_acc_d == 0: input_df['acc_now_delinq:0'] = 1 
    
    #Inq Last 6 months
    m_inq = user_data['inq_last_6mths']
    if m_inq == 0: input_df['inq_last_6mths:0'] = 1
    elif 1 <= m_inq <= 2: input_df['inq_last_6mths:1-2'] = 1
    elif 3 <= m_inq <= 6: input_df['inq_last_6mths:3-6'] = 1
    # elif m_inq > 6: input_df['inq_last_6mths:>6'] = 1
    
    
    #Months Since Earliest
    m_mths_since_earliest = user_data['mths_since_earliest_cr_line']
    # if m_mths_since_earliest in range(140): input_df['mths_since_earliest_cr_line:<140'] = 1
    if m_mths_since_earliest in range(140, 165): input_df['mths_since_earliest_cr_line:141-164'] = 1 
    elif m_mths_since_earliest in range(165, 248): input_df['mths_since_earliest_cr_line:165-247'] = 1 
    elif m_mths_since_earliest in range(248, 271): input_df['mths_since_earliest_cr_line:248-270'] = 1
    elif m_mths_since_earliest in range(271, 353): input_df['mths_since_earliest_cr_line:271-352'] = 1 
    elif m_mths_since_earliest in range(353, 587): input_df['mths_since_earliest_cr_line:>352'] = 1
    
    # Months since Issue 
    m_mths_since_issue_d = user_data['mths_since_issue_d']
    if m_mths_since_issue_d in range(38): input_df['mths_since_issue_d:<38'] = 1
    elif m_mths_since_issue_d in range(38, 40): input_df['mths_since_issue_d:38-39'] = 1
    elif m_mths_since_issue_d in range(40, 42): input_df['mths_since_issue_d:40-41'] = 1
    elif m_mths_since_issue_d in range(42, 49): input_df['mths_since_issue_d:42-48'] = 1
    elif m_mths_since_issue_d in range(49, 53): input_df['mths_since_issue_d:49-52'] = 1
    elif m_mths_since_issue_d in range(53, 65): input_df['mths_since_issue_d:53-64'] = 1
    elif m_mths_since_issue_d in range(65, 85): input_df['mths_since_issue_d:65-84'] = 1
    # elif m_mths_since_issue_d in range(85, 126): input_df['mths_since_issue_d:>84'] = 1
    
    #Employment Length
    m_emp_length = user_data['emp_length_int']
    # if m_emp_length == 0: input_df['emp_length:0'] = 1
    if m_emp_length == 1: input_df['emp_length:1'] = 1 
    elif m_emp_length in range(2, 5): input_df['emp_length:2-4'] = 1
    elif m_emp_length in range(5, 7): input_df['emp_length:5-6'] = 1
    elif m_emp_length in range(7, 10): input_df['emp_length:7-9'] = 1
    elif m_emp_length == 10: input_df['emp_length:10'] = 1
    
    #Term  int months 
    m_term_int = user_data['term_int']
    if m_term_int == 36: input_df['term:36'] = 1 
    # elif m_term_int == 60: input_df['term:60'] = 1
    


    return input_df

def preprocess_input_lgd_ead(user_data):
    feature_columns = model_wrapper2.model.feature_names_in_
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    grade_val  = user_data['grade']
    if grade_val == 'A': input_df['grade:A'] = 1
    elif grade_val == 'B': input_df['grade:B'] = 1
    elif grade_val == 'C': input_df['grade:C'] = 1
    elif grade_val == 'D': input_df['grade:D'] = 1
    elif grade_val == 'E': input_df['grade:E'] = 1
    elif grade_val == 'F': input_df['grade:F'] = 1
    # elif grade_val == 'G': input_df['grade:G'] = 1
    
    home_val = user_data['home_ownership']
    if home_val == 'OWN': input_df['home_ownership:OWN'] = 1 
    elif home_val == 'MORTGAGE': input_df['home_ownership:MORTGAGE'] = 1
    elif home_val == 'NONE': input_df['home_ownership:NONE'] = 1
    elif home_val == 'OTHER': input_df['home_ownership:OTHER'] = 1
    # elif home_val in ['RENT', 'OTHER', 'NONE', 'ANY']: input_df['home_ownership:RENT_OTHER_NONE_ANY'] = 1
    
    verif = user_data['verification']
    if verif == 'Not Verified': input_df['verification_status:Not Verified'] = 1
    elif verif =='Source Verified': input_df['verification_status:Source Verified'] = 1
    # elif verif == 'Verified': input_df['verification_status:Verified'] = 1
    
    purp = user_data['purpose']
    if purp == 'debt_consolidation': input_df['purpose:debt_consolidation'] = 1
    elif purp == 'educational': input_df['purpose:educational'] = 1
    elif purp == 'home_improvement': input_df['purpose:home_improvement'] = 1
    elif purp == 'house': input_df['purpose:house'] = 1
    elif purp == 'major_purchase': input_df['purpose:major_purchase'] = 1
    elif purp == 'medical': input_df['purpose:medical'] = 1
    elif purp == 'moving': input_df['purpose:moving'] = 1
    elif purp == 'other': input_df['purpose:other'] = 1
    elif purp == 'renewable_energy': input_df['purpose:renewable_energy'] = 1
    elif purp == 'small_business': input_df['purpose:small_business'] = 1
    elif purp == 'vacation': input_df['purpose:vacation'] = 1
    elif purp == 'wedding': input_df['purpose:wedding'] = 1
    
    initial_list = user_data['initial_list_status']
    if initial_list == 'w': input_df['initial_list_status:w'] = 1
    # elif initial_list == 'f': input_df['initial_list_status:f'] = 1 
    
    input_df['term_int'] = user_data['term_int']
    input_df['emp_length_int'] = user_data['emp_length_int']
    input_df['mths_since_issue_d'] = user_data['mths_since_issue_d']
    input_df['mths_since_earliest_cr_line'] = user_data['mths_since_earliest_cr_line']
    input_df['int_rate'] = user_data['int_rate']
    input_df['installment'] = user_data['installment']
    input_df['annual_inc'] = user_data['annual_inc']
    input_df['dti'] = user_data['dti']
    input_df['delinq_2yrs'] = user_data['delinq_2yrs']
    input_df['inq_last_6mths'] = user_data['inq_last_6mths']
    input_df['mths_since_last_delinq'] = user_data['mths_since_last_delinq']
    input_df['mths_since_last_record'] = user_data['mths_since_last_record']
    input_df['open_acc'] = user_data['open_acc']
    input_df['pub_rec'] = user_data['pub_rec']
    input_df['acc_now_delinq'] = user_data['acc_now_delinq']
    input_df['total_rev_hi_lim'] = user_data['total_rev_hi_lim']

    
    
    
    
    return input_df
    

#%%
# --- STEP 3: UI SETUP ---
st.title("🏦 Expected Loss Estimation Dashboard")

# Navigation
page = st.sidebar.selectbox("Select Page", ["Individual Loan Calculator", "Model Insights"])

if page == "Individual Loan Calculator":
    st.header("Loan Application Input")
    
    # --- UI INPUTS ---
    col1, col2, col3, col4, col5, col6= st.columns(6)
    with col1:
        loan_amt = st.number_input("Loan Amount ($)", min_value=1000, value=10000)
        grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
        addr_state = st.selectbox("State", ['NY', 'CA', 'TX', 'FL', 'IL', 'GA', 'NJ', 'VA', 'AZ', 'Other'])
    with col2:
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        term = st.selectbox("Term (Months)", [36, 60])
        dti = st.number_input("Debt-to-Income (DTI)", min_value=0.0, value=15.0)
        purpose = st.selectbox("Purpose", ['credit_card', 'debt_consolidation', 'major_purchase', 
            'medical', 'other', 'educational', 'home_improvement', 'house', 'moving', 'renewable_energy', 'small_business', 'vacation', 'wedding'])
    with col3:
        home = st.selectbox("Home Ownership", ['MORTGAGE', 'OWN', 'RENT'])
        verif = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
        m_delinq = st.number_input("Months Since Last Delinquency (-1 if none)", value=-1)
        emp_len = st.slider("Employment Length (Years)", 0, 10, 5)
    with col4:
        acc_now_delinq = st.number_input("Number of accounts now in deliquency")
        mths_since_earliest_cr_line = st.number_input("Months Since Earliest Credit Line")
        mths_since_issue_d = st.number_input("Months Since Issue Date")
        initial_list_status = st.selectbox("Initial List Status", ['f', 'w'])
    with col5:
        installment = st.number_input("Installment of the loan")
        delinq_2yrs = st.number_input("Delinquency 2 years")
        inq_last_6mths = st.number_input("Inquiry Last 6 months")
        mths_since_last_record = st.number_input("Months since Last Record")
    with col6:
        open_acc = st.number_input("Open Accounts")
        pub_rec = st.number_input("Public Record")
        total_rev_hi_lim = st.number_input("Total revolving high limit")
        
        
    if st.button("Calculate Expected Loss"):
        # 1. Package Raw UI Data
        user_data = {
            'grade': grade, 'home_ownership': home, 'verification': verif,
            'term_int': term, 'int_rate': int_rate, 'annual_inc': income,
            'dti': dti, 'mths_since_last_delinq': m_delinq,
            'addr_state': addr_state, 'purpose': purpose,
            'emp_length_int': emp_len, 'acc_now_delinq': acc_now_delinq, 
            'inq_last_6mths': inq_last_6mths, 'mths_since_last_record': mths_since_last_record,
            'mths_since_earliest_cr_line': mths_since_earliest_cr_line, 'mths_since_issue_d': mths_since_issue_d,
            'initial_list_status': initial_list_status
        }
        
        user_data_lgd_ead = {
            'grade': grade, 'home_ownership': home, 'verification': verif,
            'purpose': purpose, 'initial_list_status': initial_list_status, 'term_int': term,
            'emp_length_int': emp_len, 'mths_since_issue_d': mths_since_issue_d, 'mths_since_earliest_cr_line':mths_since_earliest_cr_line,
            'funded_amount': loan_amt, 'int_rate': int_rate, 'installment': installment,#installment,
            'annual_inc': income, 'dti': dti, 'delinq_2yrs': delinq_2yrs,#delinq_2yrs, 
            'inq_last_6mths': inq_last_6mths,#inq_last_6mths,
            'mths_since_last_delinq': m_delinq, 'mths_since_last_record':mths_since_last_record, 'open_acc': open_acc,
            'pub_rec':pub_rec, 'acc_now_delinq': acc_now_delinq, 'total_rev_hi_lim': total_rev_hi_lim}

        # 2. RUN PREPROCESSING (Now properly inside the button block)
        input_df = preprocess_inputs_pd(user_data, feature_columns1)
        
        input_df_lgd_ead = preprocess_input_lgd_ead(user_data_lgd_ead)
        
        

        # 3. PIPELINE EXECUTION
        # PD Prediction (Probability of class 1: Default)
        pd_val = model1.predict_proba(input_df)[0][0] 
        # st.header(model_wrapper2.model.feature_names_in_)
        # st.header(input_df_lgd_ead.columns)
        # LGD Stage 1 (Classification) then Stage 2 (Regression)
        recovery_st_1 = model_wrapper2.model.predict(input_df_lgd_ead)[0]
        recovery_st_2 = model_wrapper3.predict(input_df_lgd_ead)[0]
        recovery_all = recovery_st_1 * recovery_st_2
        recovery_all = np.where(recovery_all < 0, 0, recovery_all)
        recovery_all = np.where(recovery_all > 1, 1, recovery_all)
        # if recovery_occured_binary == 1:
        #     recovery_rate = model_wrapper3.predict(input_df)[0]
        #     recovery_rate = np.clip(recovery_rate, 0, 1) # Cap clinical bounds
        # else:
        #     recovery_rate = 0
        # st.header(recovery_st_1)
        # st.header(recovery_st_2)
        # st.header(recovery_all)
        lgd_val = 1 - recovery_all #
 
        # # EAD Prediction
        ead_val = model_wrapper4.predict(input_df_lgd_ead)[0] * loan_amt 
        
        # # Expected Loss
        expected_loss = pd_val * lgd_val * ead_val 

        # 4. DISPLAY
        st.divider()
        res_col1, res_col2, res_col3, res_col4, res_col5= st.columns(5)
        res_col1.metric("PD (%)", f"{pd_val:.2%}")
        res_col2.metric("LGD (%)", f"{lgd_val:.2%}")
        res_col3.metric("EAD ($)", f"${ead_val:,.2f}")
        res_col4.metric("Expected Loss ($)", f"${expected_loss:,.2f}")
        # res_col5.metric("Ratio:", f"{expected_loss/loan_amt:,.2f}")
        
    if st.button("Calculate Expected (Prime Borrower)"):
        user_data = {
            'grade': 'A', 'home_ownership': 'OWN', 'verification': 'Source Verified',
            'term_int': 36, 'int_rate': 10, 'annual_inc': 95000,
            'dti': 8, 'mths_since_last_delinq': 0,
            'addr_state': 'CA', 'purpose': 'car',
            'emp_length_int': 8, 'acc_now_delinq': 0, 
            'inq_last_6mths': 0, 'mths_since_last_record': 0,
            'mths_since_earliest_cr_line': 70, 'mths_since_issue_d': 40,
            'initial_list_status': 'w'
        }
        
        user_data_lgd_ead = {
            'grade': 'A', 'home_ownership': 'OWN', 'verification': 'Source Verified',
            'purpose': 'car', 'initial_list_status': 'w', 'term_int': 36,
            'emp_length_int': 8, 'mths_since_issue_d': 40, 'mths_since_earliest_cr_line':70,
            'funded_amount': 10000, 'int_rate': 10, 'installment': 277.77,#installment,
            'annual_inc': 95000, 'dti': 8, 'delinq_2yrs': 0,#delinq_2yrs, 
            'inq_last_6mths': 0,#inq_last_6mths,
            'mths_since_last_delinq': 0, 'mths_since_last_record':0, 'open_acc': 0,
            'pub_rec':0, 'acc_now_delinq': 0, 'total_rev_hi_lim': 25000}

        # 2. RUN PREPROCESSING (Now properly inside the button block)
        input_df = preprocess_inputs_pd(user_data, feature_columns1)
        
        input_df_lgd_ead = preprocess_input_lgd_ead(user_data_lgd_ead)
        
        

        # 3. PIPELINE EXECUTION
        # PD Prediction (Probability of class 1: Default)
        pd_val = model1.predict_proba(input_df)[0][0] 
        # st.header(model_wrapper2.model.feature_names_in_)
        # st.header(input_df_lgd_ead.columns)
        # LGD Stage 1 (Classification) then Stage 2 (Regression)
        recovery_st_1 = model_wrapper2.model.predict(input_df_lgd_ead)[0]
        recovery_st_2 = model_wrapper3.predict(input_df_lgd_ead)[0]
        recovery_all = recovery_st_1 * recovery_st_2
        recovery_all = np.where(recovery_all < 0, 0, recovery_all)
        recovery_all = np.where(recovery_all > 1, 1, recovery_all)
        # if recovery_occured_binary == 1:
        #     recovery_rate = model_wrapper3.predict(input_df)[0]
        #     recovery_rate = np.clip(recovery_rate, 0, 1) # Cap clinical bounds
        # else:
        #     recovery_rate = 0
        # st.header(recovery_st_1)
        # st.header(recovery_st_2)
        # st.header(recovery_all)
        lgd_val = 1 - recovery_all #
 
        # # EAD Prediction
        ead_val = model_wrapper4.predict(input_df_lgd_ead)[0] * loan_amt 
        
        # # Expected Loss
        expected_loss = pd_val * lgd_val * ead_val 

        # 4. DISPLAY
        st.divider()
        res_col1, res_col2, res_col3, res_col4, res_col5= st.columns(5)
        res_col1.metric("PD (%)", f"{pd_val:.2%}")
        res_col2.metric("LGD (%)", f"{lgd_val:.2%}")
        res_col3.metric("EAD ($)", f"${ead_val:,.2f}")
        res_col4.metric("Expected Loss ($)", f"${expected_loss:,.2f}")
        
    if st.button("Calculate Expected (Bad Borrower)"):
        user_data = {
            'grade': 'G', 'home_ownership': 'MORTGAGE', 'verification': 'Not Verified',
            'term_int': 60, 'int_rate': 10, 'annual_inc': 75000,
            'dti': 15, 'mths_since_last_delinq': 10,
            'addr_state': 'NV', 'purpose': 'sm_b',
            'emp_length_int': 4, 'acc_now_delinq': 1, 
            'inq_last_6mths': 2, 'mths_since_last_record': 12,
            'mths_since_earliest_cr_line': 60, 'mths_since_issue_d': 25,
            'initial_list_status': 'f'
        }
        
        user_data_lgd_ead = {
            'grade': 'G', 'home_ownership': 'MORTGAGE', 'verification': 'Not Verified',
            'purpose': 'small_business', 'initial_list_status': 'f', 'term_int': 60,
            'emp_length_int': 4, 'mths_since_issue_d': 25, 'mths_since_earliest_cr_line':60,
            'funded_amount': 10000, 'int_rate': 10, 'installment': 250,#installment,
            'annual_inc': 75000, 'dti': 15, 'delinq_2yrs': 3,#delinq_2yrs, 
            'inq_last_6mths': 2,#inq_last_6mths,
            'mths_since_last_delinq': 10, 'mths_since_last_record':12, 'open_acc': 2,
            'pub_rec':2, 'acc_now_delinq': 1, 'total_rev_hi_lim': 20000}

        # 2. RUN PREPROCESSING (Now properly inside the button block)
        input_df = preprocess_inputs_pd(user_data, feature_columns1)
        
        input_df_lgd_ead = preprocess_input_lgd_ead(user_data_lgd_ead)
        
        

        # 3. PIPELINE EXECUTION
        # PD Prediction (Probability of class 1: Default)
        pd_val = model1.predict_proba(input_df)[0][0] 
        # st.header(model_wrapper2.model.feature_names_in_)
        # st.header(input_df_lgd_ead.columns)
        # LGD Stage 1 (Classification) then Stage 2 (Regression)
        recovery_st_1 = model_wrapper2.model.predict(input_df_lgd_ead)[0]
        recovery_st_2 = model_wrapper3.predict(input_df_lgd_ead)[0]
        recovery_all = recovery_st_1 * recovery_st_2
        recovery_all = np.where(recovery_all < 0, 0, recovery_all)
        recovery_all = np.where(recovery_all > 1, 1, recovery_all)
        # if recovery_occured_binary == 1:
        #     recovery_rate = model_wrapper3.predict(input_df)[0]
        #     recovery_rate = np.clip(recovery_rate, 0, 1) # Cap clinical bounds
        # else:
        #     recovery_rate = 0
        # st.header(recovery_st_1)
        # st.header(recovery_st_2)
        # st.header(recovery_all)
        lgd_val = 1 - recovery_all #
 
        # # EAD Prediction
        ead_val = model_wrapper4.predict(input_df_lgd_ead)[0] * loan_amt 
        
        # # Expected Loss
        expected_loss = pd_val * lgd_val * ead_val 

        # 4. DISPLAY
        st.divider()
        res_col1, res_col2, res_col3, res_col4, res_col5= st.columns(5)
        res_col1.metric("PD (%)", f"{pd_val:.2%}")
        res_col2.metric("LGD (%)", f"{lgd_val:.2%}")
        res_col3.metric("EAD ($)", f"${ead_val:,.2f}")
        res_col4.metric("Expected Loss ($)", f"${expected_loss:,.2f}")

elif page == "Model Insights":
    st.header("Model Performance & Methodology")
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/gini.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/ks.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/CCF_dist.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/ead_residuals.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/lgd_residuals.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/recovery_rate_dist.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/roc_curve_lgd.png')
    st.image('C:/Users/otnie/Documents/CR/Projects/PDmodel/roc_curve.png')
    # You can use st.image('roc_curve.png') if you save your plots as images.