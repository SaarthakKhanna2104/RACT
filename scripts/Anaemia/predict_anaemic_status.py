	import sys
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt


def do_linear_regression(X,y):
	lr = GridSearchCV(linear_model.LinearRegression(),verbose=10,param_grid={})
	lr.fit(X,y)

	return lr


def fit_svm_model(X,y):
	svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1,verbose=True)
	svr_rbf.fit(X,y)


def write_to_file(df):
	df.to_csv(path_or_buf='cleaned_surgical_data.csv',sep=',')


def get_train_and_test(df):
	test_df = df[df['ResultAfterSurgery'].isnull()]
	df.dropna(subset=['ResultAfterSurgery'], how='all',axis=0,inplace=True)
	return (test_df,df)


def get_dummies_for_columns(df):
	############ one hot encoding for surgical procedure ##########
	df_surgical_procedure = pd.get_dummies(df['SURG_PROCEDURE'])
	df = pd.concat([df,df_surgical_procedure],axis=1)
	del df['SURG_PROCEDURE']

	df_surgical_speciality = pd.get_dummies(df['SURGICAL_SPECIALTY'])
	df = pd.concat([df,df_surgical_speciality],axis=1)
	del df['SURGICAL_SPECIALTY']

	df_surgeon_hash_name = pd.get_dummies(df['Surgeon_Hash_Name'])
	df = pd.concat([df,df_surgeon_hash_name],axis=1)
	del df['Surgeon_Hash_Name']

	############ one hot encoding for patient type ##########
	df_patient_type = pd.get_dummies(df['PATIENT_TYPE'])
	df = pd.concat([df,df_patient_type],axis=1)
	del df['PATIENT_TYPE']

	df_allogenic_blood_transfusion = pd.get_dummies(df['Allogeneic_Blood_Transfusion'])
	df = pd.concat([df,df_allogenic_blood_transfusion],axis=1)
	del df['Allogeneic_Blood_Transfusion']

	return df


def get_heatmap(df):
	pearson_corr = df.corr(method='pearson')
	sns.set()
	ax = sns.heatmap(pearson_corr,vmin=-1,vmax=1)
	ax.set_xticklabels(list(df.select_dtypes(['float64']).columns),rotation=90,size=5)
	# print list(df.select_dtypes(['float64']).columns)[::-1]
	ax.set_yticklabels(list(df.select_dtypes(['float64']).columns)[::-1],rotation=0,size=5)
	plt.savefig('heatmap_with_null_values.png')


def remove_rows(df):
	print df.loc[[all([a,b]) for a,b in zip(df['SN_BM_PRBC_Ordered']==0,df['SN_BM_Red_Blood_Cells']>0)]].index.tolist()
	df.drop(df.loc[[all([a,b]) for a,b in zip(df['SN_BM_PRBC_Ordered']==0,df['SN_BM_Red_Blood_Cells']>0)]].index,inplace=True)
	df.drop(df.loc[[all([a,b]) for a,b in zip(df['SN_BM_PRBC_Ordered']==0,df['SN_BM_Fresh_Frozen_Plasma']>0)]].index,inplace=True)
	df.drop(df.loc[[all([a,b]) for a,b in zip(df['SN_BM_PRBC_Ordered']==0,df['SN_BM_Platelets']>0)]].index,inplace=True)
	df.drop(df.loc[[all([a,b]) for a,b in zip(df['SN_BM_PRBC_Ordered']==0,df['SN_BM_Cryoprecipitate']>0)]].index,inplace=True)

	return df


def fill_missing_values(df):
	df['SN_BM_Pre_Op_INR'] = df['SN_BM_Pre_Op_INR'].fillna(0.0)
	df['SN_BM_Pre_Op_Platelet_Count'] = df['SN_BM_Pre_Op_Platelet_Count'].fillna(0.0)

	return df


def convert_dtypes(df):
	df['SURGICAL_SPECIALTY']=df['SURGICAL_SPECIALTY'].apply(lambda x: x.upper())
	df['SURG_PROCEDURE'] = df['SURG_PROCEDURE'].astype('category')

	df['SURG_PROCEDURE']=df['SURG_PROCEDURE'].apply(lambda x: x.upper())
	df['SURGICAL_SPECIALTY'] = df['SURGICAL_SPECIALTY'].astype('category')

	df['PATIENT_TYPE']=df['PATIENT_TYPE'].apply(lambda x: x.upper())
	df['PATIENT_TYPE'] = df['PATIENT_TYPE'].astype('category')

	df['Allogeneic_Blood_Transfusion']=df['Allogeneic_Blood_Transfusion'].apply(lambda x: x.upper())
	df['Allogeneic_Blood_Transfusion'] = df['Allogeneic_Blood_Transfusion'].astype('category')

	df['Surgeon_Hash_Name']=df['Surgeon_Hash_Name'].apply(lambda x: x.upper())
	df['Surgeon_Hash_Name'] = df['Surgeon_Hash_Name'].astype('category')

	df['SN_BM_Pre_Op_INR'] = df['SN_BM_Pre_Op_INR'].astype('float64')
	df['SN_BM_Pre_Op_Platelet_Count'] = df['SN_BM_Pre_Op_Platelet_Count'].astype('float64')

	return df


def replace_garbage_values(df):
	df = df.replace(".",np.nan)
	return df


def rename_columns(df):
	df = df.rename(columns={'Allogeneic Blood Transfusion':'Allogeneic_Blood_Transfusion',\
		'Surgeon Hash Name':'Surgeon_Hash_Name','SN - BM - Pre-Op INR':'SN_BM_Pre_Op_INR',\
		'SN - BM - Pre-Op Platelet Count':'SN_BM_Pre_Op_Platelet_Count','SN - BM - PRBC Ordered':'SN_BM_PRBC_Ordered',\
		'SN - BM - Red Blood Cells':'SN_BM_Red_Blood_Cells','SN - BM - Fresh Frozen Plasma':'SN_BM_Fresh_Frozen_Plasma',\
		'SN - BM - Platelets':'SN_BM_Platelets','SN - BM - Cryoprecipitate':'SN_BM_Cryoprecipitate'})

	return df


def drop_columns(df,columns):
	for column in columns:
		del df[column]
	return df


def read_file(file_path):
	df = pd.read_csv(file_path,sep=',')
	return df


if __name__ == '__main__':
	file_path = sys.argv[1]

	df = read_file(file_path)
	df = drop_columns(df,['Masked FIN','Sequence No.','Duration of Surgery (hh:mm).1','EBL'])
	df = rename_columns(df)
	df = replace_garbage_values(df)
	df = convert_dtypes(df)
	df = fill_missing_values(df)

	df = remove_rows(df)
	print df.describe()
	exit(0)
	df = get_dummies_for_columns(df)

	

	test_df, train_df = get_train_and_test(df)
	target_train_df, target_test_df = train_df.ix[:,-1],test_df.ix[:,-1]
	
	train_df = drop_columns(train_df,['ResultAfterSurgery'])
	test_df = drop_columns(test_df,['ResultAfterSurgery'])

	train_df = preprocessing(train_df)
	test_df = preprocessing(test_df)

	# fit_svm_model(train_df,target_train_df)
	lr = do_linear_regression(train_df,target_train_df)
	print lr.predict(test_df)
	print metrics.mean_squared_error(lr.predict(test_df),target_test_df)

