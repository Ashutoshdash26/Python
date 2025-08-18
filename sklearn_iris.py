import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
Iris=load_iris()
Iris.data
Iris.data.shape
Iris.feature_names
data=pd.DataFrame(data=Iris.data,columns=Iris.feature_names)
data
data1=pd.DataFrame(data=Iris.data,columns=Iris.feature_names),Iris.target
data1 = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data1['target']=Iris.target
data1.head()