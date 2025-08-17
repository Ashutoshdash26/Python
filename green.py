import numpy as np
arr=np.array([1,2,3],[4,5,6])
print(arr)
import numpy as np
df=pd.read_csv(r"C:\Users\KIIT\Music\list\mca 3rd\shi.csv")
df
arr=np.array([[1,2,3],[4,5,6]])
print(arr)
ar=[[1,2,3],[4,5,6]]
print(ar)
Ar=np.arange(10)
Ar
ar1=np.arange(5,20,3)
ar1
ar2=np.arange(18).reshape(6,3)
ar2
ar3=np.arange(20).reshape(2,2,5)
ar3
A=np.arange(10)
A
A.reshape(2,5)
A
A=np.arange(15).reshape(5,3)
A
print(A.shape)
print(A.ndim)
print(A.dtype)
print(A.size)
print(A.itemsize)
print(A.shape[0])


import numpy as np

ar9 = np.ones((2, 3))   

print("Enter 6 elements in array:")

for i in range(ar9.shape[0]):        
    for j in range(ar9.shape[1]):    
        ar9[i][j] = int(input(f"Enter the {(i*ar9.shape[1])+j+1}th element: "))

print("\nFinal array elements are:")
for i in range(ar9.shape[0]):
    for j in range(ar9.shape[1]):
        print(f"Element at ({i},{j}) = {ar9[i][j]}")
