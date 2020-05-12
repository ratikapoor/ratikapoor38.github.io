```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import scipy
```


```python
#Let's consider a matrix X
X = np.array([ [3, 1,1],[-1, 1,0],[-2,-2,-1]])
```


```python
#here we are extracting only the first principal axis 
pca = PCA(n_components=1)
pca.fit(X)
```




    PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)




```python
# singular values are length of rows projected on principal axes
print("Singular value:",pca.singular_values_)

# explained variance is square of length projected to 1st principal axis 
# which is square of singular value divided by (number of rows - 1)
print("explained variance computed from singular value:",np.square(pca.singular_values_[0])/(X.shape[0]-1))

print("explained variance:",pca.explained_variance_)

```

    Singular value: [4.38010876]
    explained variance computed from singular value: 9.592676385936228
    explained variance: [9.59267639]
    


```python
# pca.components are the right singular vector which is the 
# eigen vector of the covariance matrix
pca.components_
```




    array([[0.83234965, 0.45180545, 0.32103877]])




```python
"""Compute the eigen vector of covariance matrix which should match the pca component"""
print(scipy.linalg.eigh(np.matmul(X.T,X)))

#vector corresponding to highest eigen value is the 1st principal axis i.e,
#The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i]
scipy.linalg.eigh(np.matmul(X.T,X))[1][:,1]
```

    (array([-8.36312203e-17,  2.81464723e+00,  1.91853528e+01]), array([[-0.23570226,  0.50163583,  0.83234965],
           [-0.23570226, -0.86041634,  0.45180545],
           [ 0.94280904, -0.08969513,  0.32103877]]))
    




    array([ 0.50163583, -0.86041634, -0.08969513])




```python
#Project the data on 1st principal axis
AV=np.dot(X,pca.components_[0])

#normalizing the eigen vector  we get u1 which will be seen below is the eigen vector of XX.T
u1=AV/pca.singular_values_[0]
```


```python
#estimate the left singular vector which should be equal to Av/sigma
print(scipy.linalg.eigh(np.matmul(X,X.T)))
```

    (array([-5.36390107e-15,  2.81464723e+00,  1.91853528e+01]), array([[-0.57735027,  0.33069022,  0.74653242],
           [-0.57735027, -0.81186114, -0.08688008],
           [-0.57735027,  0.48117093, -0.65965234]]))
    


```python
#It can be seen the eigen vector corresponding to XX.T having highest eigen value is
# similar to u1 computed above
print(scipy.linalg.eigh(np.matmul(X,X.T))[1][:,2])
print(u1)
```

    [ 0.74653242 -0.08688008 -0.65965234]
    [ 0.74653242 -0.08688008 -0.65965234]
    
