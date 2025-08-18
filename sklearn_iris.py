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
print(data1[45:50])
print(data1[50:60])
print(data1[101:105])
n=55
data1.head(n)
l=50
data1[l:n]
data1.sample(n=5)
data1.sample(frac=1)
suffled_data=data1.sample(frac=1)
suffled_data
shuffled_data=data1.sample(frac=1).reset_index(drop=True)
shuffled_data 


iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

setosa = data[data['species'] == 'setosa']
versicolor = data[data['species'] == 'versicolor']
virginica = data[data['species'] == 'virginica']

print("Setosa samples:", len(setosa))
print("Versicolor samples:", len(versicolor))
print("Virginica samples:", len(virginica))


