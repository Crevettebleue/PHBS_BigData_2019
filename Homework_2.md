
# Big Data Analysis
- Climate Change


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```


```python
data1_set = 'climate_change_1.csv'
data1_df = pd.read_csv(data1_set)
trainset_df = data1_df.head(284)
testset_df = data1_df[284:]
#Y_train = preprocessing.scale(trainset_df['Temp'])
#X_train = preprocessing.scale(trainset_df.loc[:,trainset_df.columns != 'Temp'])
#Y_test = preprocessing.scale(testset_df['Temp'])
#X_test = preprocessing.scale(testset_df.loc[:,testset_df.columns != 'Temp'])
Y_train = trainset_df['Temp']
X_train = trainset_df.loc[:,trainset_df.columns != 'Temp']
Y_test = testset_df['Temp']
X_test = testset_df.loc[:,testset_df.columns != 'Temp']
```

# Problem 1 
### Question 1:


```python
def closed_form_1(X,Y):
#RSS = (Y - X %% beta)^2
#RSS = t(Y) %% Y - 2*t(Y) %% X %% beta + t(beta) %% t(x) %% x %% beta
#dRSS/dbeta = -2 * t(X) %% Y + 2 t(x) %% X %% beta
# set dRSS/dbeta = 0
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y.T)
```

### Question 2:

The mathematical formula for the linear model is the following:

$ Y_i = \alpha + \beta X_i+\epsilon_i$


```python
def Rsquare(X_train, X_test,Y_train,Y_test):
#Beta from training set
    betas = closed_form_1(X_train,Y_train)
    
#Yhat from training set
    y_hat = X_train.dot(betas)
    
#Rsquare of training set
    RSS_train = sum(np.square(y_hat-np.mean(Y_train)))
    TSS_train = sum(np.square(Y_train-np.mean(Y_train)))
    Rtrain = RSS_train/TSS_train

#y_hat from testing set
    y_hat_test = X_test.dot(betas)
    
#Rsquare of test set
    RSS_test = sum(np.square(y_hat_test-np.mean(Y_train)))
    TSS_test = sum(np.square(Y_test-np.mean(Y_train)))
    Rtest = RSS_test/TSS_test
    
    return Rtrain, Rtest

```


```python
Rsquare(X_train, X_test,Y_train,Y_test)
```




    (0.7489870281266865, 0.6416955268744396)




```python
closed_form_1(X_train,Y_train)
```




    array([-4.24887416e-02, -6.16714874e-03,  6.48363535e-02,  6.46994230e-03,
           -1.57088222e-04,  2.29289620e-02, -1.00275979e-02,  6.77771347e-03,
            5.49139594e-02, -1.64648727e+00])




```python
import
```

### Question 4


```python
data2_set = 'climate_change_2.csv'
data2_df = pd.read_csv(data2_set)
Y_set2 = trainset_df['Temp']
X_set2 = trainset_df.loc[:,trainset_df.columns != 'Temp']
```


```python
closed_form_1(X_set2,Y_set2)
```




    array([-4.24887416e-02, -6.16714874e-03,  6.48363535e-02,  6.46994230e-03,
           -1.57088222e-04,  2.29289620e-02, -1.00275979e-02,  6.77771347e-03,
            5.49139594e-02, -1.64648727e+00])




```python
Rsquare(X_train,X_set2,Y_train,Y_set2)
```




    (0.7489870281266865, 0.7489870281266865)



The above solution is unreasonable as it forces the regression over the 10 variables of the train set. However, climate_change_2 has 11 variables to test. A specific 11 variable regression must be done to train and test for the data of climate_change_2. 

# Problem 2 
### Question 1:
The loss function is the following:

L1 loss function:

$  PRSS(\beta) = \sum_{i=1}^n(y_i - z_i^T \beta)^2 + \lambda \sum_{j=1}^p |\beta_j| $

Ridge loss function:

$ PRSS(\beta) = \sum_{i=1}^n (y_i - z_i^T \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 $

### Question 2:


```python
def closed_form_2(X,Y,λ):
#same logic as previously, just add the lambda*identity matrix 
    I = np.identity(np.size(X,1))
    return np.linalg.inv((X.T.dot(X))+λ*I).dot(X.T).dot(Y.T)
```


```python
def RsquareRidge(X_train, X_test,Y_train,Y_test,λ):
#Beta from training set
    betas = closed_form_2(X_train,Y_train,λ)
    
#Yhat from training set
    y_hat = X_train.dot(betas)
    
#Rsquare of training set
    RSS_train = sum(np.square(y_hat-np.mean(Y_train)))
    TSS_train = sum(np.square(Y_train-np.mean(Y_train)))
    train = RSS_train/TSS_train

#y_hat from testing set
    y_hat_test = X_test.dot(betas)
    
#Rsquare of test set
    RSS_test = sum(np.square(y_hat_test-np.mean(Y_train)))
    TSS_test = sum(np.square(Y_test-np.mean(Y_train)))
    test = RSS_test/TSS_test
    
    return train, test,λ
```


```python
RsquareRidge(X_train,X_test,Y_train,Y_test,0.01)
```




    (0.7435834685829527, 0.6645348202429334, 0.01)




```python
closed_form_2(X_train,Y_train,0.01)
```




    array([-4.17761906e-02, -6.08912747e-03,  6.38588627e-02,  6.57976853e-03,
           -1.42369912e-04,  2.20783218e-02, -1.00098106e-02,  6.74015643e-03,
            5.40287874e-02, -1.55907411e+00])



###  Question 3:
    


Ridge regression belongs a class of regression tools that use L2 regularization. The other type of regularization, L1 regularization, limits the size of the coefficients by adding an L1 penalty equal to the absolute value of the magnitude of coefficients. This sometimes results in the elimination of some coefficients altogether, which can yield sparse models. L2 regularization adds an L2 penalty, which equals the square of the magnitude of coefficients. All coefficients are shrunk by the same factor (so none are eliminated). Unlike L1 regularization, L2 will not result in sparse models.

Additionally, the tuning parameter (λ) controls the strength of the penalty term. When λ = 0, ridge regression equals least squares regression. When λ goes towards infinity, the coefficient are running towards 0.

### Question 4


```python
para_λ = [0.0001,0.001,0.01,0.1,1,10]
for ii in para_λ:
    print(RsquareRidge(X_train, X_test, Y_train, Y_test, ii))
```

    (0.748928468298267, 0.6419332178957088, 0.0001)
    (0.7484058186717164, 0.6440636887668141, 0.001)
    (0.7435834685829527, 0.6645348202429334, 0.01)
    (0.7181180086134995, 0.8090106075287486, 0.1)
    (0.6969605262871925, 1.0933584050360983, 1)
    (0.6881983166156359, 1.2797641349563642, 10)
    

# Problem 3

### Question 1

Workflow to select feature:

we will be using backward elimination:

1) build the model with all features availible in the dataset

2)  eliminate the highest p-value if 

$ p-value < 0.05 $

3) repeat until there is no p-value that satisfies the above terms


### Question 2

as we established that some variables are redundant, L1 regression seems more appropriate

We will use the sklearn linear_regression.Lasso in order to train the model

# Problem 4


```python
#gradient descent
def gradientDescent(X,α):
    size = np.size(X[0])
    param = np.random.uniform(0,1,size)
    param_update = param + α * 2 * (X.T.dot((Y - X.dot(param))))
    while sum(param == param_update) != size:
        param = param_update
        param_update = param + α * 2 * (X.T.dot((Y - X.dot(param))))
```
