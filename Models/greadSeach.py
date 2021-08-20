from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd
from skmultiflow.data import DataStream

from skmultiflow.trees import HoeffdingTreeRegressor
#bagging IJCNN 
#from streaming_random_patches_regressor import StreamingRandomPatchesRegressor
from skmultiflow.data import FileStream

from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection import EDDM
from skmultiflow.drift_detection import KSWIN

from oselm import OSELMRegressor#, OSELMClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor


import numpy as np
import data_exp

#stream=FileStream("./Data/Mex_Hat_3_d.csv")

#dataset_bike_current=data_exp.bike_current()



def params(X_train,Y_train, name):
		
	LASSO=[]
	parameters={'selection':('cyclic', 'random')}
	ht_reg = linear_model.Lasso()
	clfLA = GridSearchCV(ht_reg, parameters)
	clfLA.fit(X_train,Y_train)
	LASSO.append(clfLA.best_params_)
	LASSO.append(clfLA.best_score_)


	RF=[]
	parameters={'n_estimators':[100,500],'criterion':("mse", "mae")}
	ht_reg = RandomForestRegressor()
	clfRF = GridSearchCV(ht_reg, parameters)
	clfRF.fit(X_train,Y_train)
	RF.append(clfRF.best_params_)
	RF.append(clfRF.best_score_)

	MLP=[]
	parameters={'hidden_layer_sizes':[100,300],'activation':('identity', 'logistic', 'tanh'),'solver':('lbfgs', 'sgd', 'adam'),'learning_rate':('constant', 'invscaling', 'adaptive'),'learning_rate_init':[0.001,0.1]}
	ht_reg = MLPRegressor()
	clfMLP = GridSearchCV(ht_reg, parameters)
	clfMLP.fit(X_train,Y_train)
	MLP.append(clfMLP.best_params_)
	MLP.append(clfMLP.best_score_)

	TREE=[]
	parameters={'criterion':("mse", "friedman_mse", "mae"),'splitter':("best", "random"),"max_depth":[1,5],"min_samples_split":[2,5],"min_samples_leaf":[1,5],"max_features":("auto", "sqrt", "log2")}
	ht_reg = DecisionTreeRegressor()
	clfT = GridSearchCV(ht_reg, parameters)
	clfT.fit(X_train,Y_train)
	TREE.append(clfT.best_params_)
	TREE.append(clfT.best_score_)


	ht=[]
	parameters={'leaf_prediction':('mean','perceptron'),'learning_ratio_decay':[-1,1],'learning_ratio_perceptron':[-1,1]}
	ht_reg = HoeffdingTreeRegressor()
	clfHT = GridSearchCV(ht_reg, parameters)
	clfHT.fit(X_train,Y_train)
	ht.append(clfHT.best_params_)
	ht.append(clfHT.best_score_)


	parameters={'n_hidden':[10,30],
				'activation_func':('tanh', 'sigmoid')
				}
	OS_ELM=[]
	ELM=OSELMRegressor()
	clfE = GridSearchCV(ELM, parameters)
	clfE.fit(X_train,Y_train)
	OS_ELM.append(clfE.best_params_)
	OS_ELM.append(clfE.best_score_)
	
	PRF=[]
	parameters={'n_estimators':[10,100],'aggregation_method':('median','mean'),'drift_detection_criteria':('mse','mae','predictions')}
	mdl=AdaptiveRandomForestRegressor()
	clfF = GridSearchCV(mdl, parameters)
	clfF.fit(X_train,Y_train)
	PRF.append(clfF.best_params_)
	PRF.append(clfF.best_score_)

	Boosting=[]
	parameters={'n_estimators':[100,500],'criterion':('friedman_mse', 'mse', 'mae')}
	mdl=GradientBoostingRegressor()
	clfB = GridSearchCV(mdl, parameters)
	clfB.fit(X_train,Y_train)
	Boosting.append(clfB.best_params_)
	Boosting.append(clfB.best_score_)

	Bagging=[]
	parameters={'base_estimator':(linear_model.Lasso(**clfLA.best_params_),RandomForestRegressor(**clfRF.best_params_),MLPRegressor(**clfMLP.best_params_),DecisionTreeRegressor(**clfT.best_params_),HoeffdingTreeRegressor(**clfHT.best_params_),OSELMRegressor(**clfE.best_params_),AdaptiveRandomForestRegressor(**clfF.best_params_),GradientBoostingRegressor(**clfB.best_params_))}
	mdl=BaggingRegressor()
	clf = GridSearchCV(mdl, parameters)
	clf.fit(X_train,Y_train)
	Bagging.append(clf.best_params_)
	Bagging.append(clf.best_score_)

	salvarParametros={"Bagging":Bagging,
					"Boosting":Boosting,
					"LASSO":LASSO,
					"RF":RF,
					"MLP":MLP,
					"TREE":TREE,
					"OSELM":OS_ELM,
					"HTREE":ht,
					'PRF':PRF}

	df=pd.DataFrame(salvarParametros,columns=["Bagging","Boosting","LASSO","RF","MLP","TREE","OSELM","HTREE","PRF"])
	df.to_csv('../Results/algorithmsRidge/parameters/params_'+name+'.csv',index=False)


'''
dataset_bike_Drift=data_exp.bike_Drift()
stream=DataStream(dataset_bike_Drift)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(366)
name="bike_Drift"
params(X_train,Y_train, name)

dataset_bike_Drift=data_exp.bike_Drift()
stream=DataStream(dataset_bike_Drift)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(366)
name="bike_Drift"
params(X_train,Y_train, name)


dataset_mult=data_exp.mult()
stream=DataStream(dataset_mult)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(1000)
name="mult"
params(X_train,Y_train, name)

dataset_fried1=data_exp.friedman1() 
stream=DataStream(dataset_fried1)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(1000)
name="friedman1"
params(X_train,Y_train, name)

dataset_fried3=data_exp.friedman3() 
stream=DataStream(dataset_fried1)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(1000)
name="friedman3"
params(X_train,Y_train, name)

dataset_mex=data_exp.mex_hat_3d()
stream=DataStream(dataset_mex)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(1000)
name="mex_hat_3d"
params(X_train,Y_train, name)


dataset_mex=data_exp.FCCU1_Drift()
stream=DataStream(dataset_mex)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(52)
name="FCCU1_Drift"
params(X_train,Y_train, name)

dataset_mex=data_exp.FCCU2_Drift()
stream=DataStream(dataset_mex)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(52)
name="FCCU2_Drift"
params(X_train,Y_train, name)

dataset_mex=data_exp.FCCU3_Drift()
stream=DataStream(dataset_mex)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(52)
name="FCCU3_Drift"
params(X_train,Y_train, name)

'''
dataset_mex=data_exp.uci_airquality_Drift()
stream=DataStream(dataset_mex)
stream.prepare_for_use()
X_train,Y_train=stream.next_sample(3355)
name="uci_airquality_Drift"
params(X_train,Y_train, name)
