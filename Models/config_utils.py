from warnings import filterwarnings
filterwarnings('ignore')
from matplotlib import pyplot as plt

import numpy as np
from sklearn import datasets
from scipy.special import expit
from sklearn.svm import OneClassSVM
import data_exp
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.metrics import mean_squared_error


from skmultiflow.data import ConceptDriftStream
import pandas as pd
import evaluation as ev 
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.drift_detection import KSWIN
from skmultiflow.lazy import KNNRegressor
from sklearn.linear_model import BayesianRidge
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection import PageHinkley
from sklearn.metrics import mean_absolute_error
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.lazy import KNNRegressor
import math
import json
import time 

from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
import ast

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mean_relative_error(y_true, y_pred):
	#mean_absolute_error
	#print("Y", y_true,y_pred)
	res=abs(y_true-y_pred)/mean_absolute_error(y_true,y_pred)
	#print("RES",res,mean_absolute_error(y_true,y_pred))
	'''if len(y_pred) != len(y_true):
	 	print("RelMAE: the lengths of input vectors must be the same.")
	ferror=mean(abs(y_true- y_pred))
	ferrorbench=(mean(abs(y_true-)))'''  
	return  res#abs(y_true-y_pred)/mean_absolute_error(y_true,y_pred)#abs(y_true)

def input_detector(y_true, y_pred):
	error=mean_relative_error(y_true, y_pred)
	for i, err in enumerate(error):
		if (err<1):
			error[i]=0
		else:
			error[i]=1
	return error
	

def get_train(stream,bath_size_train,models_detection,model_base,mdl):
	X_train,Y_train=stream.next_sample(bath_size_train)
	if model_base != "---":
		#print("Model Base", type(model_base))
		model_base= ast.literal_eval(model_base)
	#mdl=model_base.fit(X_train,Y_train)
	for detection in models_detection:
		if detection == 'adwin0':
			adwin0 =ADWIN(delta=0)
			
		elif detection == 'adwin0001':
			adwin0001 =ADWIN(delta=0.001)
			
		elif detection == 'adwin0002':
			adwin0002 =ADWIN(delta=0.002)

		elif detection == 'adwin0005':
			adwin0005 =ADWIN(delta=0.95)
					
		elif detection == 'adwin05':
			adwin05 =ADWIN(delta=0.5)			

		elif detection == 'adwin1':
			adwin1 =ADWIN(delta=1)	

		elif detection == 'kswin':
			kswin = KSWIN(alpha=0.05,window_size=100,stat_size=30)
			for k in Y_train:
				kswin.add_element(k)

		elif detection == 'ddm':
			ddm=DDM()

		elif detection == 'eddm':
			eddm=EDDM()

		elif detection == 'hddm_A':
			hddm_a = HDDM_A()

		elif detection == 'hddm_W':
			hddm_w = HDDM_W()

		elif detection == 'pagehinkley':
			ph=PageHinkley()
			for k in Y_train:
				ph.add_element(k)

		else:
			status_detection = None
			print('not implemented: ' + detection)

	if mdl=='Bagging_HTREE':
		train_models={'adwin0':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=HoeffdingTreeRegressor(**model_base)).fit(X_train,Y_train),
					  }
	elif mdl=="Boosting":
		
		train_models={'adwin0':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[GradientBoostingRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial':GradientBoostingRegressor(**model_base).fit(X_train,Y_train),
					  'No_Partial':GradientBoostingRegressor(**model_base).fit(X_train,Y_train),
					 }
	
	elif mdl=='Bagging_Boosting':
		train_models={'adwin0':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=GradientBoostingRegressor(**model_base)).fit(X_train,Y_train),
					  }
	elif mdl=="HTREE":
		train_models={'adwin0':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial':HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),
					  'No_Partial':HoeffdingTreeRegressor(**model_base).fit(X_train,Y_train),
					}
	elif mdl=="LASSO":
		train_models={'adwin0':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[linear_model.Lasso(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[linear_model.Lasso(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[linear_model.Lasso(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[linear_model.Lasso(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[linear_model.Lasso(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[linear_model.Lasso(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[linear_model.Lasso(**model_base).fit(X_train,Y_train),ph],
					  'Partial':linear_model.Lasso(**model_base).fit(X_train,Y_train),
					  'No_Partial':linear_model.Lasso(**model_base).fit(X_train,Y_train),
					  }	
	elif mdl=='Bagging_LASSO':
		train_models={'adwin0':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=linear_model.Lasso(**model_base)).fit(X_train,Y_train),
					  }
	elif mdl=="TREE":
		train_models={'adwin0':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[DecisionTreeRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial':DecisionTreeRegressor(**model_base).fit(X_train,Y_train),
					  'No_Partial':DecisionTreeRegressor(**model_base).fit(X_train,Y_train),
					  }
	elif mdl=='Bagging_TREE':
		train_models={'adwin0':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=DecisionTreeRegressor(**model_base)).fit(X_train,Y_train),
					  }
	elif mdl=="MLP":
		train_models={'adwin0':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[MLPRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[MLPRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[MLPRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[MLPRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[MLPRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[MLPRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[MLPRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial':MLPRegressor(**model_base).fit(X_train,Y_train),
					  'No_Partial':MLPRegressor(**model_base).fit(X_train,Y_train),
					  }
	elif mdl=='Bagging_MLP':
		train_models={'adwin0':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=MLPRegressor(**model_base)).fit(X_train,Y_train),
					  }
	elif mdl=="PRF":
		train_models={'adwin0':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial_OSELM':AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),
					  'OSELM':AdaptiveRandomForestRegressor(**model_base).fit(X_train,Y_train),
					  }
	elif mdl=='Bagging_PRF':
		train_models={'adwin0':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=AdaptiveRandomForestRegressor(**model_base)).fit(X_train,Y_train),
					 }

	elif mdl=="RF":
		train_models={'adwin0':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0],
					  'adwin0001':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin0005],
					  'adwin05':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin05],
					  'adwin1':[RandomForestRegressor(**model_base).fit(X_train,Y_train),adwin1],
					  'kswin':[RandomForestRegressor(**model_base).fit(X_train,Y_train),kswin],
					  'ddm':[RandomForestRegressor(**model_base).fit(X_train,Y_train),ddm],
					  'eddm':[RandomForestRegressor(**model_base).fit(X_train,Y_train),eddm],
					  'hddm_A':[RandomForestRegressor(**model_base).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[RandomForestRegressor(**model_base).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[RandomForestRegressor(**model_base).fit(X_train,Y_train),ph],
					  'Partial':RandomForestRegressor(**model_base).fit(X_train,Y_train),
					  'No_Partial':RandomForestRegressor(**model_base).fit(X_train,Y_train),
					  }
	elif mdl=='Bagging_RF':
		train_models={'adwin0':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0],
					  'adwin0001':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0001],
					  'adwin0002':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0002],
					  'adwin0005':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin0005],
					  'adwin05':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin05],
					  'adwin1':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),adwin1],
					  'kswin':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),kswin],
					  'ddm':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),ddm],
					  'eddm':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),eddm],
					  'hddm_A':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),hddm_a],
					  'hddm_W':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),hddm_w],
					  'pagehinkley':[BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),ph],
					  'Partial':BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),
					  'No_Partial':BaggingRegressor(base_estimator=RandomForestRegressor(**model_base)).fit(X_train,Y_train),
					  }
			
	return stream,train_models

def get_test(stream, 
			bath_size_test, 
			models_detection,partial_or_Fit):

	if (partial_or_Fit == "partial"):

		
		statusADWIN0=[]
		statusADWIN0001=[]
		statusADWIN0002=[]
		statusADWIN0005=[]
		statusADWIN05=[]
		statusADWIN1=[]
		statusKSWIN=[]
		statusDDM=[]
		statusEDDM=[]
		statusHDDM_A=[]
		statusHDDM_W=[]
		statusPH=[]
		teste=0

	
		y_predADWIN0=[]
		y_predADWIN0001=[]
		y_predADWIN0002=[]
		y_predADWIN0005=[]
		y_predADWIN05=[]
		y_predADWIN1=[]
		y_predKSWIN=[]
		y_predDDM=[]
		y_predEDDM=[]
		y_predHDDM_A=[]
		y_predHDDM_W=[]
		y_predPH=[]

		
		timeADWIN0=[]
		timeADWIN0001=[]
		timeADWIN0002=[]
		timeADWIN0005=[]
		timeADWIN05=[]
		timeADWIN1=[]
		timeKSWIN=[]
		timeDDM=[]
		timeEDDM=[]
		timeHDDM_A=[]
		timeHDDM_W=[]
		timePH=[]
		timePartial=[]
		timeNo_Partial=[]

		y_predNo_Partial=[]
		y_predPartial=[]

		
		adwin0=models_detection['adwin0'][1]
		mdladwin0=models_detection['adwin0'][0]

		adwin0001=models_detection['adwin0001'][1]
		mdladwin0001=models_detection['adwin0001'][0]

		adwin0002=models_detection['adwin0002'][1]
		mdladwin0002=models_detection['adwin0002'][0] 

		adwin0005=models_detection['adwin0005'][1]
		mdladwin0005=models_detection['adwin0005'][0] 

		adwin05=models_detection['adwin05'][1]
		mdladwin05=models_detection['adwin05'][0] 

		adwin1=models_detection['adwin1'][1]
		mdladwin1=models_detection['adwin1'][0]

		kswin=models_detection['kswin'][1]
		mdlkswin=models_detection['kswin'][0]

		ddm=models_detection['ddm'][1]
		mdlddm=models_detection['ddm'][0]

		eddm=models_detection['eddm'][1]
		mdleddm=models_detection['eddm'][0]

		hddm_A=models_detection['hddm_A'][1]
		mdlhddm_A=models_detection['hddm_A'][0]

		hddm_W=models_detection['hddm_W'][1]
		mdlhddm_W=models_detection['hddm_W'][0]

		ph=models_detection['pagehinkley'][1]
		mdlph=models_detection['pagehinkley'][0]

		mdlNo_Partial=models_detection['No_Partial']
		mdlPartial=models_detection['Partial']
		seg=[]

		while stream.has_more_samples()==True:
			x,y=stream.next_sample(bath_size_test)
			teste+=1
			
			for detection in models_detection:
				if detection=='Partial':
					start_time=time.time()
					y_pred1=mdlPartial.predict(x)
					timePred=time.time()-start_time
					timePartial.append(timePred)
					mdlPartial.partial_fit(x,y)
					y_predPartial.extend(y_pred1)

				elif detection=='No_Partial':
					start_time=time.time()
					y_pred=mdlNo_Partial.predict(x)
					timePred=time.time()-start_time
					timeNo_Partial.append(timePred)
					y_predNo_Partial.extend(y_pred)

				
				elif detection == 'adwin0':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0.predict(x)
					y_predADWIN0.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0.add_element(drift_input)
						if adwin0.detected_change():
							auxStatus=1
							status=1
							adwin0.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0.append(timePred)
					if auxStatus==1:
						mdladwin0.partial_fit(x,y)
						
					statusADWIN0.extend(status_All)

				elif detection == 'adwin0001':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0001.predict(x)
					y_predADWIN0001.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1

						adwin0001.add_element(drift_input)
						if adwin0001.detected_change():
							auxStatus=1
							status=1
							adwin0001.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0001.append(timePred)
					if auxStatus==1:
						mdladwin0001.partial_fit(x,y)

					statusADWIN0001.extend(status_All)

				elif detection == 'adwin0002':			
					status_All=[]
					start_time=time.time()
					pred=mdladwin0002.predict(x)
					y_predADWIN0002.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0002.add_element(drift_input)
						if adwin0002.detected_change():
							auxStatus=1
							status=1
							adwin0002.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0002.append(timePred)
					if auxStatus==1:
						mdladwin0002.partial_fit(x,y)

					statusADWIN0002.extend(status_All)

				elif detection == 'adwin0005':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0005.predict(x)
					y_predADWIN0005.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0005.add_element(drift_input)
						if adwin0005.detected_change():
							auxStatus=1
							status=1
							adwin0005.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0005.append(timePred)
					if auxStatus==1:
						mdladwin0005.partial_fit(x,y)
						
					statusADWIN0005.extend(status_All)

				elif detection == 'adwin05':
					status_All=[]
					start_time=time.time()
					pred=mdladwin05.predict(x)
					y_predADWIN05.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin05.add_element(drift_input)
						if adwin05.detected_change():
							auxStatus=1
							status=1
							adwin05.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN05.append(timePred)
					if auxStatus==1:
						mdladwin05.partial_fit(x,y)

					statusADWIN05.extend(status_All)

				elif detection == 'adwin1':
					status_All=[]
					start_time=time.time()
					pred=mdladwin1.predict(x)
					y_predADWIN1.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin1.add_element(drift_input)
						if adwin1.detected_change():
							auxStatus=1
							status=1
							adwin1.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN1.append(timePred)
					if auxStatus==1:
						mdladwin1.partial_fit(x,y)

					statusADWIN1.extend(status_All)

				elif detection == 'kswin':			
					status_All=[]
					start_time=time.time()
					pred=mdlkswin.predict(x)
					y_predKSWIN.extend(pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i in pred:
						status=0
						
						kswin.add_element(i)
						if kswin.detected_change():
							auxStatus=1
							status=1
							kswin.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeKSWIN.append(timePred)
					if auxStatus==1:
						mdlkswin.partial_fit(x,y)
						
					statusKSWIN.extend(status_All)

				elif detection == 'ddm':			
					status_All=[]
					start_time=time.time()
					pred=mdlddm.predict(x)
					y_predDDM.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						
						status=0
						drift_input=err
						ddm.add_element(drift_input)
						if ddm.detected_change():
							auxStatus=1
							status=1
							ddm.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeDDM.append(timePred)
					if auxStatus==1:
						mdlddm.partial_fit(x,y)
					statusDDM.extend(status_All)

				elif detection == 'eddm':	
					status_All=[]
					start_time=time.time()
					pred=mdleddm.predict(x)
					y_predEDDM.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						drift_input=err
						eddm.add_element(drift_input)
						if eddm.detected_change():
							auxStatus=1
							status=1
							eddm.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeEDDM.append(timePred)
					if auxStatus==1:
						mdleddm.partial_fit(x,y)
					statusEDDM.extend(status_All)

				elif detection == 'hddm_A':				
					status_All=[]
					start_time=time.time()
					pred=mdlhddm_A.predict(x)
					y_predHDDM_A.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						drift_input=err
						hddm_A.add_element(drift_input)
						if hddm_A.detected_change():
							auxStatus=1
							status=1
							hddm_A.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeHDDM_A.append(timePred)
					if auxStatus==1:
						mdlhddm_A.partial_fit(x,y)
					statusHDDM_A.extend(status_All)


				elif detection == 'hddm_W':				
					status_All=[]
					start_time=time.time()
					pred=mdlhddm_W.predict(x)
					y_predHDDM_W.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=1
					for i,err in enumerate(error):
						status=0
						drift_input=err
						hddm_W.add_element(drift_input)
						if hddm_W.detected_change():
							auxStatus=1
							status=1
							hddm_W.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeHDDM_W.append(timePred)
					if auxStatus==1:
						mdlhddm_W.partial_fit(x,y)
					statusHDDM_W.extend(status_All)

				elif detection == 'pagehinkley':				
					status_All=[]
					start_time=time.time()
					pred=mdlph.predict(x)
					y_predPH.extend(pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i in pred:
						status=0
						ph.add_element(i)
						if ph.detected_change():
							auxStatus=1
							status=1
							ph.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timePH.append(timePred)
					if auxStatus==1:
						mdlph.partial_fit(x,y)
						
					statusPH.extend(status_All)

			timeM={'timeADWIN0':timeADWIN0,
				'timeADWIN0001':timeADWIN0001,
				'timeADWIN0002':timeADWIN0002,
				'timeADWIN0005':timeADWIN0005,
				'timeADWIN05':timeADWIN05,
				'timeADWIN1':timeADWIN1,
				'timeDDM':timeDDM,
				'timeEDDM':timeEDDM,
				'timeHDDM_A':timeHDDM_A,
				'timeHDDM_W':timeHDDM_W,
				'timeKSWIN':timeKSWIN,
				'timePH':timePH,
				'timeNo_Partial':timeNo_Partial,
				'timePartial':timePartial}

			statusModels={'statusADWIN0':statusADWIN0,
						'y_predADWIN0':y_predADWIN0,
					  'statusADWIN0001':statusADWIN0001,
					  'y_predADWIN0001':y_predADWIN0001,
					  'statusADWIN0002':statusADWIN0002,
					  'y_predADWIN0002':y_predADWIN0002,
					  'statusADWIN0005':statusADWIN0005,
					  'y_predADWIN0005':y_predADWIN0005,
					  'statusADWIN05':statusADWIN05,
					  'y_predADWIN05':y_predADWIN05,
					  'statusADWIN1':statusADWIN1,
					  'y_predADWIN1':y_predADWIN1,
					  'statusDDM':statusDDM,
					  'y_predDDM':y_predDDM,
					  'statusEDDM':statusEDDM,
					  'y_predEDDM':y_predEDDM,
					  'statusHDDM_A':statusHDDM_A,
					  'y_predHDDM_A':y_predHDDM_A,
					  'statusHDDM_W':statusHDDM_W,
					  'y_predHDDM_W':y_predHDDM_W,
					  'statusKSWIN':statusKSWIN,
					  'y_predKSWIN':y_predKSWIN,
					  'statusPH':statusPH,
					  'y_predPH':y_predPH,
					  'y_predNo_Partial':y_predNo_Partial,
					  'y_predPartial':y_predPartial}

		return statusModels, timeM,seg

	elif (partial_or_Fit == "fit"):

		
		statusADWIN0=[]
		statusADWIN0001=[]
		statusADWIN0002=[]
		statusADWIN0005=[]
		statusADWIN05=[]
		statusADWIN1=[]
		statusKSWIN=[]
		statusDDM=[]
		statusEDDM=[]
		statusHDDM_A=[]
		statusHDDM_W=[]
		statusPH=[]
		teste=0

		
		y_predADWIN0=[]
		y_predADWIN0001=[]
		y_predADWIN0002=[]
		y_predADWIN0005=[]
		y_predADWIN05=[]
		y_predADWIN1=[]
		y_predKSWIN=[]
		y_predDDM=[]
		y_predEDDM=[]
		y_predHDDM_A=[]
		y_predHDDM_W=[]
		y_predPH=[]

		
		timeADWIN0=[]
		timeADWIN0001=[]
		timeADWIN0002=[]
		timeADWIN0005=[]
		timeADWIN05=[]
		timeADWIN1=[]
		timeKSWIN=[]
		timeDDM=[]
		timeEDDM=[]
		timeHDDM_A=[]
		timeHDDM_W=[]
		timePH=[]
		timePartial=[]
		timeNo_Partial=[]

		y_predNo_Partial=[]
		y_predPartial=[]

		adwin0=models_detection['adwin0'][1]
		mdladwin0=models_detection['adwin0'][0]

		adwin0001=models_detection['adwin0001'][1]
		mdladwin0001=models_detection['adwin0001'][0]

		adwin0002=models_detection['adwin0002'][1]
		mdladwin0002=models_detection['adwin0002'][0] 

		adwin0005=models_detection['adwin0005'][1]
		mdladwin0005=models_detection['adwin0005'][0] 

		adwin05=models_detection['adwin05'][1]
		mdladwin05=models_detection['adwin05'][0] 

		adwin1=models_detection['adwin1'][1]
		mdladwin1=models_detection['adwin1'][0]

		kswin=models_detection['kswin'][1]
		mdlkswin=models_detection['kswin'][0]

		ddm=models_detection['ddm'][1]
		mdlddm=models_detection['ddm'][0]

		eddm=models_detection['eddm'][1]
		mdleddm=models_detection['eddm'][0]

		hddm_A=models_detection['hddm_A'][1]
		mdlhddm_A=models_detection['hddm_A'][0]

		hddm_W=models_detection['hddm_W'][1]
		mdlhddm_W=models_detection['hddm_W'][0]

		ph=models_detection['pagehinkley'][1]
		mdlph=models_detection['pagehinkley'][0]

		mdlNo_Partial=models_detection['No_Partial']
		mdlPartial=models_detection['Partial']
		seg=[]

		while stream.has_more_samples()==True:
			x,y=stream.next_sample(bath_size_test)
			teste+=1
			
			for detection in models_detection:
				if detection=='Partial':
					start_time=time.time()
					y_pred1=mdlPartial.predict(x)
					timePred=time.time()-start_time
					timePartial.append(timePred)
					mdlPartial=mdlPartial.fit(x,y)
					y_predPartial.extend(y_pred1)

				elif detection=='No_Partial':
					start_time=time.time()
					y_pred=mdlNo_Partial.predict(x)
					timePred=time.time()-start_time
					timeNo_Partial.append(timePred)
					y_predNo_Partial.extend(y_pred)

				
				elif detection == 'adwin0':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0.predict(x)
					y_predADWIN0.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0.add_element(drift_input)
						if adwin0.detected_change():
							auxStatus=1
							status=1
							adwin0.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0.append(timePred)
					if auxStatus==1:
						mdladwin0=mdladwin0.fit(x,y)
						
					statusADWIN0.extend(status_All)

				elif detection == 'adwin0001':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0001.predict(x)
					y_predADWIN0001.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1

						adwin0001.add_element(drift_input)
						if adwin0001.detected_change():
							auxStatus=1
							status=1
							adwin0001.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0001.append(timePred)
					if auxStatus==1:
						mdladwin0001=mdladwin0001.fit(x,y)

					statusADWIN0001.extend(status_All)

				elif detection == 'adwin0002':			
					status_All=[]
					start_time=time.time()
					pred=mdladwin0002.predict(x)
					y_predADWIN0002.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0002.add_element(drift_input)
						if adwin0002.detected_change():
							auxStatus=1
							status=1
							adwin0002.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0002.append(timePred)
					if auxStatus==1:
						mdladwin0002=mdladwin0002.fit(x,y)

					statusADWIN0002.extend(status_All)

				elif detection == 'adwin0005':
					status_All=[]
					start_time=time.time()
					pred=mdladwin0005.predict(x)
					y_predADWIN0005.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin0005.add_element(drift_input)
						if adwin0005.detected_change():
							auxStatus=1
							status=1
							adwin0005.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN0005.append(timePred)
					if auxStatus==1:
						mdladwin0005=mdladwin0005.fit(x,y)
						
					statusADWIN0005.extend(status_All)

				elif detection == 'adwin05':
					status_All=[]
					start_time=time.time()
					pred=mdladwin05.predict(x)
					y_predADWIN05.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin05.add_element(drift_input)
						if adwin05.detected_change():
							auxStatus=1
							status=1
							adwin05.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN05.append(timePred)
					if auxStatus==1:
						mdladwin05=mdladwin05.fit(x,y)

					statusADWIN05.extend(status_All)

				elif detection == 'adwin1':
					status_All=[]
					start_time=time.time()
					pred=mdladwin1.predict(x)
					y_predADWIN1.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						if err==1:
							drift_input=0
						else: 
							drift_input=1
						adwin1.add_element(drift_input)
						if adwin1.detected_change():
							auxStatus=1
							status=1
							adwin1.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeADWIN1.append(timePred)
					if auxStatus==1:
						mdladwin1=mdladwin1.fit(x,y)

					statusADWIN1.extend(status_All)

				elif detection == 'kswin':			
					status_All=[]
					start_time=time.time()
					pred=mdlkswin.predict(x)
					y_predKSWIN.extend(pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i in pred:
						status=0
						
						kswin.add_element(i)
						if kswin.detected_change():
							auxStatus=1
							status=1
							kswin.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeKSWIN.append(timePred)
					if auxStatus==1:
						mdlkswin=mdlkswin.fit(x,y)
						
					statusKSWIN.extend(status_All)

				elif detection == 'ddm':			
					status_All=[]
					start_time=time.time()
					pred=mdlddm.predict(x)
					y_predDDM.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						
						status=0
						drift_input=err
						ddm.add_element(drift_input)
						if ddm.detected_change():
							auxStatus=1
							status=1
							ddm.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeDDM.append(timePred)
					if auxStatus==1:
						mdlddm=mdlddm.fit(x,y)
					statusDDM.extend(status_All)

				elif detection == 'eddm':	
					status_All=[]
					start_time=time.time()
					pred=mdleddm.predict(x)
					y_predEDDM.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						drift_input=err
						eddm.add_element(drift_input)
						if eddm.detected_change():
							auxStatus=1
							status=1
							eddm.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeEDDM.append(timePred)
					if auxStatus==1:
						mdleddm=mdleddm.fit(x,y)
					statusEDDM.extend(status_All)

				elif detection == 'hddm_A':				
					status_All=[]
					start_time=time.time()
					pred=mdlhddm_A.predict(x)
					y_predHDDM_A.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i,err in enumerate(error):
						status=0
						drift_input=err
						hddm_A.add_element(drift_input)
						if hddm_A.detected_change():
							auxStatus=1
							status=1
							hddm_A.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeHDDM_A.append(timePred)
					if auxStatus==1:
						mdlhddm_A=mdlhddm_A.fit(x,y)
					statusHDDM_A.extend(status_All)


				elif detection == 'hddm_W':				
					status_All=[]
					start_time=time.time()
					pred=mdlhddm_W.predict(x)
					y_predHDDM_W.extend(pred)
					error=input_detector(y, pred)
					cont=0
					pos_pred=[]
					auxStatus=1
					for i,err in enumerate(error):
						status=0
						drift_input=err
						hddm_W.add_element(drift_input)
						if hddm_W.detected_change():
							auxStatus=1
							status=1
							hddm_W.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timeHDDM_W.append(timePred)
					if auxStatus==1:
						mdlhddm_W=mdlhddm_W.fit(x,y)
					statusHDDM_W.extend(status_All)

				elif detection == 'pagehinkley':				
					status_All=[]
					start_time=time.time()
					pred=mdlph.predict(x)
					y_predPH.extend(pred)
					cont=0
					pos_pred=[]
					auxStatus=0
					for i in pred:
						status=0
						ph.add_element(i)
						if ph.detected_change():
							auxStatus=1
							status=1
							ph.reset()
						status_All.append(status)
					timePred=time.time()-start_time
					timePH.append(timePred)
					if auxStatus==1:
						mdlph=mdlph.fit(x,y)
						
					statusPH.extend(status_All)

			timeM={'timeADWIN0':timeADWIN0,
				'timeADWIN0001':timeADWIN0001,
				'timeADWIN0002':timeADWIN0002,
				'timeADWIN0005':timeADWIN0005,
				'timeADWIN05':timeADWIN05,
				'timeADWIN1':timeADWIN1,
				'timeDDM':timeDDM,
				'timeEDDM':timeEDDM,
				'timeHDDM_A':timeHDDM_A,
				'timeHDDM_W':timeHDDM_W,
				'timeKSWIN':timeKSWIN,
				'timePH':timePH,
				'timeNo_Partial':timeNo_Partial,
				'timePartial':timePartial}

			statusModels={'statusADWIN0':statusADWIN0,
						'y_predADWIN0':y_predADWIN0,
					  'statusADWIN0001':statusADWIN0001,
					  'y_predADWIN0001':y_predADWIN0001,
					  'statusADWIN0002':statusADWIN0002,
					  'y_predADWIN0002':y_predADWIN0002,
					  'statusADWIN0005':statusADWIN0005,
					  'y_predADWIN0005':y_predADWIN0005,
					  'statusADWIN05':statusADWIN05,
					  'y_predADWIN05':y_predADWIN05,
					  'statusADWIN1':statusADWIN1,
					  'y_predADWIN1':y_predADWIN1,
					  'statusDDM':statusDDM,
					  'y_predDDM':y_predDDM,
					  'statusEDDM':statusEDDM,
					  'y_predEDDM':y_predEDDM,
					  'statusHDDM_A':statusHDDM_A,
					  'y_predHDDM_A':y_predHDDM_A,
					  'statusHDDM_W':statusHDDM_W,
					  'y_predHDDM_W':y_predHDDM_W,
					  'statusKSWIN':statusKSWIN,
					  'y_predKSWIN':y_predKSWIN,
					  'statusPH':statusPH,
					  'y_predPH':y_predPH,
					  'y_predNo_Partial':y_predNo_Partial,
					  'y_predPartial':y_predPartial}

		return statusModels, timeM,seg
	else:
		print('not implemented: fit or partial')

def dataset_config(dataset,params_batch_train, params_batch_test,models,model_base,mdl,partial_or_Fit):
	dataset_status=[]
	
	status_batch=[]
	statusSeg=[]
	datasetSeg=[]
	timeModels=[]
	for p_train in params_batch_train:
		stream=DataStream(dataset)
		stream.prepare_for_use()
		stream,train_models=get_train(stream,p_train, models,model_base,mdl)
		
		for p_test in params_batch_test:

			statusModels,timeM,seg=get_test(stream,p_test,train_models,partial_or_Fit)
			print("MDL",mdl)
			status_batch.append(statusModels)
			statusSeg.append(seg)
			stream.restart()
			stream.next_sample(p_train)


	dataset_status.append(status_batch)
	datasetSeg.append(statusSeg)
	timeModels.append(timeM)

	return dataset_status,timeModels,datasetSeg

def params_Gread(paramters,seed):

	Boosting=paramters['Boosting'][0].replace("}","")
	Boosting=Boosting+",'random_state':"+str(seed)+"}"

	Bagging=paramters['Bagging'][0].replace("}","")
	Bagging=Bagging+",'random_state':"+str(seed)+"}"

	lasso=paramters['LASSO'][0].replace("}","")
	lasso=lasso+",'random_state':"+str(seed)+"}"

	mlp=paramters['MLP'][0].replace("}","")
	mlp=mlp+",'random_state':"+str(seed)+"}"

	tree=paramters['TREE'][0].replace("}","")
	tree=tree+",'random_state':"+str(seed)+"}"

	rf=paramters['RF'][0].replace("}","")
	rf=rf+",'random_state':"+str(seed)+"}"

	htree=paramters['HTREE'][0].replace("}","")
	htree=htree+",'random_state':"+str(seed)+"}"

	prf=paramters['PRF'][0].replace("}","")
	prf=prf+",'random_state':"+str(seed)+"}"

	res=[Boosting,
		lasso,
		mlp,
		tree,
		htree,
		Boosting,
		lasso,
		mlp,
		tree,
		htree]

	return res

def main():
	for i in range(30):
		seed=i
		np.random.seed(seed)
		#dataset_mult=data_exp.mult()
		#dataset_fried1=data_exp.friedman1() 
		#dataset_fried3=data_exp.friedman3()
		#dataset_mex=data_exp.mex_hat_3d()
		dataset_bike_current=data_exp.bike_current()
		dataset_bike_Drift=data_exp.bike_Drift()
		dataset_FCCU1=data_exp.FCCU1_Drift()
		#dataset_FCCU2=data_exp.FCCU2_Drift()
		#dataset_FCCU3=data_exp.FCCU3_Drift()
		

		names=['FCCU1']#,'FCCU2','FCCU3']
		#['bike_current','bike_Drift']
		#['mult','friedman1','friedman3','mex_hat_3d']

		datasets=[dataset_FCCU1]#,dataset_FCCU2,dataset_FCCU3]
		#[dataset_bike_current,dataset_bike_Drift]
		
		#[dataset_mult,dataset_fried1,dataset_fried3,dataset_mex]
				  

		models={'adwin0','adwin0001','adwin0002','adwin0005','adwin05','adwin1','ddm','eddm','hddm_A','hddm_W','kswin','pagehinkley'}

		mdl=["Boosting","LASSO","MLP","TREE","HTREE",'Bagging_Boosting','Bagging_LASSO',"Bagging_MLP","Bagging_TREE","Bagging_HTREE"]
		partial_or_Fit=["fit","fit","fit","fit","partial","fit","fit","fit","fit","fit"]

		#paramters= data_exp.params_bike_Drift()
		#mult=params_Gread(paramters,seed)
		
		paramters= data_exp.params_FCCU1_Drift()#data_exp.params_mult()
		mult=params_Gread(paramters,seed)
		'''
		paramters= data_exp.params_FCCU2_Drift()#data_exp.params_friedman1()
		fried1=params_Gread(paramters,seed)

		paramters= data_exp.params_FCCU3_Drift()#data_exp.params_friedman3()
		fried3=params_Gread(paramters,seed)

		paramters= data_exp.params_mex_hat_3d()
		mex=params_Gread(paramters,seed)'''

		mdl_base=[mult]#fried1,fried3]#,mex]

		bath_size_test=[13]

		train_=[51]

		print("seed",i)
		for k,dataset in enumerate(datasets):
			models_base=mdl_base[k]
			aux_train_=train_[k]
			aux_bath_size_test=bath_size_test[k]
			print(names[k])
			for p, valmdl in enumerate(mdl):
				print("valmdl",valmdl)

				statusModels,timeModels,datasetSeg=dataset_config(dataset,[aux_train_],[aux_bath_size_test],models,models_base[p],valmdl,partial_or_Fit[p])
				
				#print("timeModels[0]",timeModels[0])
			
				dfTime=pd.DataFrame(timeModels[0],columns=[
					'timeADWIN0','timeADWIN0001','timeADWIN0002',
					'timeADWIN0005','timeADWIN05','timeADWIN1',
					'timeDDM','timeEDDM','timeHDDM_A','timeHDDM_W',
					'timeKSWIN','timePH','timeNo_Partial','timePartial'])


				df1=pd.DataFrame(statusModels[0][0],columns=[	
					'statusADWIN0','statusADWIN0001',
					'statusADWIN0002','statusADWIN0005','statusADWIN05',
					'statusADWIN1','statusDDM','statusEDDM','statusHDDM_A',
					'statusHDDM_W','statusKSWIN','statusPH',
				  'y_predADWIN0','y_predADWIN0001',	  'y_predADWIN0002','y_predADWIN0005',
				  'y_predADWIN05','y_predADWIN1',
				  'y_predDDM','y_predEDDM', 'y_predHDDM_A','y_predHDDM_W',
				  'y_predKSWIN', 'y_predPH','y_predNo_Partial',
				  'y_predPartial'])
							

				for i in range(np.size(bath_size_test)):
					df1.to_csv('../BRACIS-2021/'+names[k]+str(bath_size_test[i])+'SEED'+str(seed)+str(valmdl)+'.csv',index=False)
					dfTime.to_csv('../BRACIS-2021/'+'TIME-'+names[k]+str(bath_size_test[i])+'SEED'+str(seed)+str(valmdl)+'.csv',index=False)
if __name__ == '__main__':
	main()