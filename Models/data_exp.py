import numpy as np
import pandas as pd

import sklearn.datasets
from skmultiflow.data import FileStream
from skmultiflow.data import DataStream
import normalize as norm
import math

#For sintetc dataset
def load_data(dataset,data):

    if dataset == 0:
        #stream=FileStream("Dataset/MultNormArtigo.csv")
        data=mult()
        stream=DataStream(data)
        stream.prepare_for_use()
        size=stream.n_remaining_samples()
        x,y=stream.next_sample(size)

    elif dataset == 1:
        data=mex_hat_3d()
        stream=DataStream(data)
        stream.prepare_for_use()
        size=stream.n_remaining_samples()
        x,y=stream.next_sample(size)

    elif dataset == 2:
        data=friedman1()
        stream=DataStream(data)
        stream.prepare_for_use()
        size=stream.n_remaining_samples()
        x,y=stream.next_sample(size)

    elif dataset == 3:
        data=friedman3()
        stream=DataStream(data)
        stream.prepare_for_use()
        size=stream.n_remaining_samples()
        x,y=stream.next_sample(size)

    elif dataset==4:
        stream=DataStream(data)
        stream.prepare_for_use()
        size=stream.n_remaining_samples()
        x,y=stream.next_sample(size)

    else:

        raise NotImplementedError('A valid dataset needs to be given.')

    return x, y

def mult():
    dataset = pd.read_csv('../Dataset/Mult.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def params_mult():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_mult.csv',delimiter =',')
    return dataset

def multDetector():
    dataset = pd.read_csv('../Dataset/MultDetector.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def mex_hat_3d():
    dataset = pd.read_csv('../Dataset/Mex_Hat_3_d.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def params_mex_hat_3d():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_mex_hat_3d.csv',delimiter =',')
    return dataset

def mex_Hat_3_dDetector():
    dataset = pd.read_csv('../Dataset/Mex_Hat_3_dDetector.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def friedman1():
    dataset = pd.read_csv('../Dataset/Friedman1.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def params_friedman1():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_friedman1.csv',delimiter =',')
    return dataset

def friedman1Detector():
    dataset = pd.read_csv('../Dataset/Friedman1Detector.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return dataset

def friedman3():
    dataset = pd.read_csv('../Dataset/Friedman3.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df
def params_friedman3():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_friedman3.csv',delimiter =',')
    return dataset

def friedman3Detector():
    dataset = pd.read_csv('../Dataset/Friedman3Detector.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def bike_current():
    dataset = pd.read_csv('../Dataset/reais/bike_current.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def bike_Drift():
    dataset = pd.read_csv('../Dataset/reais/BikeDrift.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def params_bike_Drift():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_bike_Drift.csv',delimiter =',')

    return dataset

def bike_trend_removed():
    dataset = pd.read_csv('../Dataset/reais/bike_trend_removed.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def uci_airquality():
    dataset = pd.read_csv('../Dataset/reais/uci_airquality.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def uci_airquality_Drift():
    dataset = pd.read_csv('../Dataset/reais/uci_airquality_Drift.csv',delimiter =',')
    #normalization=norm.normEstatistic(dataset)
    #dataset = pd.DataFrame(normalization)
    return dataset

def FCCU1():
    dataset = pd.read_csv('../Dataset/reais/FCCU1.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def params_FCCU1_Drift():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_FCCU1_Drift.csv',delimiter =',')

    return dataset

def params_FCCU2_Drift():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_FCCU2_Drift.csv',delimiter =',')

    return dataset

def params_FCCU3_Drift():
    dataset = pd.read_csv('../Results/algorithmsRidge/parameters/params_FCCU3_Drift.csv',delimiter =',')

    return dataset  
def FCCU1_Drift():
    dataset = pd.read_csv('../Dataset/reais/FCCU1_Drift.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def FCCU2():
    dataset = pd.read_csv('../Dataset/reais/FCCU2.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def FCCU2_Drift():
    dataset = pd.read_csv('../Dataset/reais/FCCU2_Drift.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def FCCU3():
    dataset = pd.read_csv('../Dataset/reais/FCCU3.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def FCCU3_Drift():
    dataset = pd.read_csv('../Dataset/reais/FCCU3_Drift.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return df

def dataset1d():
    dataset = pd.read_csv('../Dataset/Base1d.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    df = pd.DataFrame(normalization)
    return dataset

def synthetic_01():
    dataset = pd.read_csv('../Dataset/data/synthetic_01.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    dataset = pd.DataFrame(normalization)
    return dataset

def synthetic_02():
    dataset = pd.read_csv('../Dataset/data/synthetic_02.csv',delimiter =',')
    normalization=norm.normEstatistic(dataset)
    dataset = pd.DataFrame(normalization)
    return dataset

def synthetic_03():
    dataset = pd.read_csv('../Dataset/data/synthetic_03.csv',delimiter =',')
    #normalization=norm.normEstatistic(dataset)
    #dataset = pd.DataFrame(normalization)
    return dataset

def synthetic_04():
    dataset = pd.read_csv('../Dataset/data/synthetic_04.csv',delimiter =',')
    #normalization=norm.normEstatistic(dataset)
    #dataset = pd.DataFrame(normalization)
    return dataset



def domain(begin, end, data):
    #print(data[0])
    #class_value=0
    class_value_all=[]
    for i in data:
        class_value=0
        for j in i:
            #print("j",j)
            if (not (begin <= j and j<=end)):
                #print("entrou")
                class_value = 1
        class_value_all.append(class_value)
    return class_value_all

def domain_var(begin,end,x):
    class_value_all=[]
    for j in x:
        class_value=0
        if (not(begin <= j and j<=end)):
            class_value=1
        class_value_all.append(class_value)
    return class_value_all

def domain_variables(x1,x2,x3,x4, size):
    class_value_all=[]
    for i in range(size):
        class_status=0
        if x1[i]==1 or x2[i]==1 or x3[i]==1 or x4[i]==1:
            class_status=1
        class_value_all.append(class_status) 
    return class_value_all

def test():
    dataset1 = pd.read_csv('../Dataset/Base1d.csv',delimiter =',')
    mult = pd.read_csv('../Dataset/Mult.csv',delimiter =',')
    Mex_Hat_3_d = pd.read_csv('../Dataset/Mex_Hat_3_d.csv',delimiter =',')
    friedman1 = pd.read_csv('../Dataset/Friedman1.csv',delimiter =',')
    friedman3 = pd.read_csv('../Dataset/Friedman3.csv',delimiter =',')

    multDetector = pd.read_csv('../Dataset/MultDetector.csv',delimiter =',')
    friedman1Detector=pd.read_csv('../Dataset/Friedman1Detector.csv',delimiter =',')
    friedman3Detector=pd.read_csv('../Dataset/Friedman3Detector.csv',delimiter =',')
    mex_Hat_3_dDetector = pd.read_csv('../Dataset/Mex_Hat_3_dDetector.csv',delimiter =',')


    x,y=load_data(4,dataset1)
    class_dataset1=domain(-1,0.4,x)

    x,y=load_data(4,mult)
    class_Mult=domain(0,0.7,x)

    x,y=load_data(4,friedman1)
    class_friedman1=domain(0,0.7,x)

    x,y=load_data(4,friedman3)
    x1=domain_var(0,0.7,friedman3['x1'])
    x2=domain_var(40*math.pi,(((560*math.pi)/10)*7),friedman3['x2'])
    x3=domain_var(0,0.7,friedman3['x3'])
    x4=domain_var(1,((11/10)*7),friedman3['x4'])
    class_friedman3=domain_variables(x1,x2,x3,x4, 5000)

    x,y=load_data(4,Mex_Hat_3_d)
    class_Mex_Hat_3_d=domain(-4*math.pi,(((4*math.pi)*20)/50),x)

    x,y=load_data(4,multDetector)
    class_MultDetector=domain(0,0.7,x)

    x,y=load_data(4,friedman1Detector)
    class_friedman1Detector=domain(0,0.7,x)

    x,y=load_data(4,friedman3Detector)
    x1=domain_var(0,0.7,friedman3Detector['x1'])
    x2=domain_var(40*math.pi,(((560*math.pi)/10)*7),friedman3Detector['x2'])
    x3=domain_var(0,0.7,friedman3Detector['x3'])
    x4=domain_var(1,((11/10)*7),friedman3Detector['x4'])
    class_friedman3Detector=domain_variables(x1,x2,x3,x4, 5000)

    x,y=load_data(4,mex_Hat_3_dDetector)
    class_Mex_Hat_3_dDetector=domain(-4*math.pi,(((4*math.pi)*20)/50),x)

    
    save={'class_dataset1':class_dataset1,
          'class_Mult':class_Mult,
          'class_friedman1':class_friedman1,
          'class_friedman3':class_friedman3,
          'class_Mex_Hat_3_d':class_Mex_Hat_3_d,
          'class_MultDetector':class_MultDetector,
          'class_friedman1Detector':class_friedman1Detector,
          'class_friedman3Detector':class_friedman3Detector,
          'class_Mex_Hat_3_dDetector':class_Mex_Hat_3_dDetector}

    df=pd.DataFrame(save,columns=['class_dataset1',
                                  'class_Mult',
                                  'class_friedman1',
                                  'class_friedman3',
                                  'class_Mex_Hat_3_d',
                                  'class_MultDetector',
                                  'class_friedman1Detector',
                                  'class_friedman3Detector',
                                  'class_Mex_Hat_3_dDetector'])
    df.to_csv('../Results/class_datasets.csv',index=False)
    
    print("resultado,", class_friedman1.count(1))
    

    print("resultado,", class_friedman1.count(1))

if __name__ == '__main__':
    test()
    pass

