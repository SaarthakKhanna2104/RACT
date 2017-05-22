import sys
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


'''
y = 

    NO  YES
0  1.0  0.0
1  1.0  0.0
2  1.0  0.0
3  1.0  0.0
4  1.0  0.0

'''


def get_roc(X,y,model):
	y_pred = model.predict_proba(X,batch_size=32,verbose=10)
	fpr, tpr, _ = roc_curve(y[:,0],y_pred[:,0])
	roc_auc = auc(fpr,tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve for Autologous Transfusion')
	plt.legend(loc="lower right")
	plt.savefig('../Visualizations/Recommendation/roc_autologous.png')



def evaluate_model(X,y,model):
	score = model.evaluate(X_test,y_test,batch_size=32)
	print "\n", model.metrics_names," ==> "
	print score


def get_training_loss_vis(history):
	################# Visualizing the training and CV error ##################################
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train','CV'],loc='upper right')
	plt.savefig('../Visualizations/Recommendation/recommendation_train_cv_loss_mlp_20.png')


def deeper_model(neurons1=80,neurons2=8,dropout=0.0):
	model = Sequential()
	model.add(Dense(neurons1,input_dim=1026,kernel_initializer='normal',activation='relu'))
	model.add(Dense(neurons2,kernel_initializer='normal',activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(2,kernel_initializer='normal',activation='softmax'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model


############## 1 hidden layer architecture with Best parameters #########################
############## neurons1 = number of neurons in layer 1 #############################
def baseline_model(neurons1=50,dropout=0.0):
	model = Sequential()
	model.add(Dense(neurons1,input_dim=1026,kernel_initializer='normal',activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(2,kernel_initializer='normal',activation='softmax'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model



############# Perform Grid Search on the parameters ############################
def do_grid_search(model):
	############ Experimenting with the batch_size, epochs, dropout, ###########
	############ neurons in layer 1 and neurons in layer 2 #####################
	
	batch_size = [512] ###### best batch size
	epochs = [40] ######## best number of epochs
	dropout = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	neurons1 = [80] ####### best number of neurons in layer1 is 80
	neurons2 = [1,8,15,25,35,55] ####### best is 10 with dropout 0.0
	# param_grid_bm = dict(epochs=epochs,batch_size=batch_size,neurons1=neurons1,dropout=dropout)
	param_grid_dm = dict(epochs=epochs,batch_size=batch_size,neurons1=neurons1,neurons2=neurons2,dropout=dropout)
	grid = GridSearchCV(estimator=model,param_grid=param_grid_dm,verbose=10)
	grid_result = grid.fit(X_train,y_train)

	return (grid,grid_result)


def build_model():
	model = KerasClassifier(build_fn=deeper_model,verbose=0)
	return model


def split_into_training_and_test(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
	return (X_train,X_test,y_train,y_test)


def preprocessing(df):
	############## Deleting the columns that are not going to be used in the model ###########
	##########################################################################################
	del df['Unnamed: 0']
	del df['Masked FIN']
	del df['Sequence No.']
	del df['SN - BM - PRBC Ordered']
	del df['EBL']
	del df['Duration of Surgery (hh:mm).1']
	del df['ResultAfterSurgery']
	
	del df['SN - BM - Red Blood Cells']
	del df['SN - BM - Fresh Frozen Plasma']
	del df['SN - BM - Platelets']
	del df['SN - BM - Cryoprecipitate']
	############## Converting the target column to category ##################################
	# df['Allogeneic Blood Transfusion'] = df['Allogeneic Blood Transfusion'].astype('category')
	# cat_columns = df.select_dtypes(['category']).columns
	# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

	############## Get predictor and target variables ########################################
	y = df['Allogeneic Blood Transfusion']
	del df['Allogeneic Blood Transfusion']
	X = df

	############## Get dummy variables for the categorical columns ###########################
	X = pd.get_dummies(X,drop_first=True)
	y = pd.get_dummies(y)

	return (X.values,y.values)

if __name__ == '__main__':
	seed = 7
	np.random.seed(seed)

	df = pd.read_csv('../Data/data_after_predicting_anaemia_values.csv',sep=',')
	X,y = preprocessing(df)
	X_train,X_test,y_train,y_test = split_into_training_and_test(X,y)
	
	# history,model = build_model(X_train,y_train)
	# get_training_loss_vis(history)
	model = build_model()
	grid,grid_result = do_grid_search(model)


	########## Getting ROC ####################
	get_roc(X_test,y_test,grid.best_estimator_)

	########## Printing the results of the grid search ########
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))



