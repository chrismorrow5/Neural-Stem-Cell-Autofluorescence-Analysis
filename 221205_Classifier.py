# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:54:26 2022

@author: agillette
"""#Based on schmitz_r-lymphocyte_activation/helper.py 

#%% Section 1 - Import required packages

import umap.umap_ as umap
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import holoviews as hv
hv.extension("bokeh")
from holoviews import opts
from holoviews.plotting import list_cmaps

#%% Section 2 Set-up for classifiers + ROC curves


def calculate_roc_rf(rf_df, key='Group'): 
    
    # Need to binarize the problem as a 'One vs. all' style approach for ROC classification
    classes = ['A', 'Q']

    #designate train/test data, random forest classifier
    X, y = rf_df.iloc[:,:-1], rf_df[[key]]
    y = label_binarize(y, classes=classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, shuffle=True)
    y_train = np.ravel(y_train)
    clf = RandomForestClassifier(random_state=0)
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    y_pred = clf.fit(X_train, y_train).predict(X_test)


    # Compute ROC curve and ROC area for each class
    
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('')
    plt.legend(loc="lower right", fontsize = 20)
    plt.show()
    
#%% Section 3 - Read in and set up dataframe 

#Read in dataframe    

# all_df = pd.read_csv('Z:/0-Projects and Experiments/RS - lymphocyte activation/data/AllCellData.csv')
all_df = pd.read_csv('C:/Users/cmorc/Desktop/Example/042219_flim_outputs.csv')


# load new nk cells 
#df_nk = pd.read_csv('Data files/UMAPs, boxplots, ROC curves (Python)/NKdonors11-29.csv')
#df_nk = df_nk.rename(columns={'n.t1.mean' : 'NADH_t1', 
#                              'n.t2.mean' : 'NADH_t2', 
#                             'n.a1.mean' : 'NADH_a1', 
#                              'n.tm.mean' : 'NADH_tm', 
#                              'f.t1.mean' : 'FAD_t1', 
#                              'f.t2.mean' : 'FAD_t2',
#                              'f.a1.mean' : 'FAD_a1', 
#                              'rr.mean' : 'Norm_RR', 
#                              'f.tm.mean' : 'FAD_tm', 
#                              'npix' : 'Cell_Size_Pix'
#                              })

## Concat dicts
#df_concat = pd.concat([all_df,df_nk])
#df_concat['Organoid'].unique()
#df_concat['Cell_Type'].unique()

#df_all = df_concat

##%%

#Add combination variables to data set
#all_df.drop(['nt1', 'nt2', 'na1', 'fi', 'ft1', 'ft2', 'fa1'], axis=1, inplace=True)
all_df['Type_Group'] = all_df['Group'] #+ ': ' + all_df['Activation']
#all_df['Donor_Activation'] = all_df['Cell_Type'] +' '+ all_df['Donor'] + ': ' + all_df['Activation']
#all_df['Donor_CellType'] = all_df['Donor'] + ': ' + all_df['Cell_Type'] 

df_data = all_df.copy()

#%% Section 4 - All cell activation classifier

print('Cell Type Classifier')

#List of OMI variables we want in the classifier (**Make sure Activation is last item in list)
list_omi_parameters = ['ni', 'nt1', 'nt2', 'na1', 'fi', 'ft1', 'ft2', 'fa1', 'Group']

   
#Make copy of main data frame, pull out OMI variables we want in classifier
all_df_edit = all_df.copy()
all_df_edit = all_df_edit[list_omi_parameters]
classes = ['A', 'Q']

#Split training/testing data, random forest classifier
X, y = all_df_edit.iloc[:,:-1], all_df_edit[['Group']]
y = label_binarize(y, classes=classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, shuffle=True)
y_train = np.ravel(y_train)
clf = RandomForestClassifier(random_state=0)
y_score = clf.fit(X_train, y_train).predict_proba(X_test)
y_pred = clf.fit(X_train, y_train).predict(X_test)

#Calculate and display confusion matrix
factor = pd.factorize(all_df_edit[['Group']].squeeze())
definitions = factor[1]
reversefactor = dict(zip(range(5), definitions))
y_test_rf = np.vectorize(reversefactor.get)(y_test)
y_pred_rf = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(np.ravel(y_test_rf), y_pred_rf, rownames=['Actual Group Type'], colnames=['Predicted Group Type']))

#Print features with weight in classifier
for col, feature in zip(np.flip(all_df_edit.columns[np.argsort(clf.feature_importances_)]), np.flip(np.argsort(clf.feature_importances_))):
    print(col, clf.feature_importances_[feature])

#Generate ROC curve
omi_params_umap = all_df_edit.copy()
calculate_roc_rf(omi_params_umap)    

#Print metrics to assess classifier performance
print('Accuracy score =', accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))