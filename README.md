# Project-5
In this project we will compare different regularization techniques like Ridge, Lasso, Elastic Net, SCAD, and Square root Lasso. We will compare their performance by applying all methods on the data we will simulate in the second question.

To start, let's create two sklean compliant functions, one for Squaroot Lasso and another one for SCAD. After creating these function we are going to use them in conuction with GridSearchCV to find the optimal hyper-parameter when we are guven x and y data variables. 

Let us import all required libraries that we will use to successfully compare our tuning hyper-parameters. 

```Python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
```

```Python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
```

After importing all general libraries, we are going to import an optimizer to help optimize import variables for our function.

```Python
from scipy.optimize import minimize
```

We also import the Base Estimator and Regressor Mixin to make our function more Sklearncompliant.

```Python
from sklearn.base import BaseEstimator, RegressorMixin
```
Importing jit to speed up our processing time 

```Python
from numba import njit
```

Our Square Root Lasso sklearn compliant function

```Python
class SQRTLasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
  
    def fit(self, x, y):
        alpha=self.alpha
        @njit
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        @njit
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = (-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)
          return output.flatten()
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
       
```
```Python
model1 = SQRTLasso (alpha = 0.6)
model 1

```

SMOOTHLY CLIPPED ABSOLUTE DEVIATION (SCAD) compliant function

Let us start by defining scad functions
```Python
#defining scad functions
@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```
```Python
class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        
        beta0 = np.zeros((x.shape[1],1))
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```
``Python
model_scad = SCAD(a = 2, lam = 1)
```

# After defining our functions, we are going to simulate 100 datasets each with 1200 features, 200 observations and we are using toeplitz correlation structure

```Python
n = 200
p = 1200
```
```Python
beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))
```
```Python
# we need toeplitz([1,0.8,0.8**2,0.8**3,0.8**4,...0.8**1199])
v = []
for i in range(p):
  v.append(0.8**i)
```

```Python
mu = [0]*p
sigma = 3.5
# Generate the random samples.
np.random.seed(123)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n) # this where we generate some fake data
y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))
```

```Python
@njit
def f_grad(x,y,beta,alpha):
  n=x.shape[0]
  p=x.shape[1]
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  output = (-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)
  return output
```

```Python
f_grad(x,y,beta_star,0.1)

```
array([[ 0.00278222],
       [-0.00424794],
       [ 0.13998555],
       ...,
       [ 0.13293468],
       [ 0.14684336],
       [ 0.14280593]])
       
       
```Python
model1 = SQRTLasso(alpha = 0.8)
#get the coefficients
model1.coef_
```
array([ 7.30022896e-03,  5.96712134e-08,  4.36657145e-02, ...,
        4.37475042e-12,  2.10572396e-10, -7.21272105e-11])
        
```python
#check the distance between the model coefficient and beta_star
np.linalg.norm(model1.coef_-beta_star,ord=2)
```
3.7216598978757958

Apply Lasso and check how many important variables are recovered

```Python
model_lasso = Lasso(alpha=0.5,fit_intercept=False,max_iter=10000)
```

```PYthon
model_lasso.fit(x,y)
model_lasso.coef_
```
array([ 0.87672602,  2.60465118,  0.16529373, ..., -0.        ,
       -0.        , -0.        ])
       
```Python
np.linalg.norm(model_lasso.coef_-beta_star,ord=2)
```      
3.9394787659620483

```Python
model_ridge = Ridge(alpha =0.6, fit_intercept = False, max_iter = 10000)
model_ridge.fit(x,y)
model_ridge.coef_

```
array([[ 0.73288445,  0.81692647,  0.69790345, ..., -0.07933362,
        -0.05808182, -0.04977748]])

```Python
np.linalg.norm(model_ridge.coef_-beta_star,ord=2)
```
3.0025146102134515

```Python
model_elasticnet = ElasticNet(alpha =0.5, fit_intercept = False, max_iter = 10000)
model_elasticnet.fit(x,y)
model_elasticnet.coef_
```
array([ 1.02324856,  1.36101409,  0.73375141, ..., -0.        ,
       -0.        , -0.        ])
       
```Python
np.linalg.norm(model_elasticnet.coef_-beta_star,ord=2)
```
1.6659615299388049

Let us apply all these variable selection methods above with GRID Search CV for tuning the hyperparameters.

```Python
grid = GridSearchCV(estimator=SQRTLasso(),cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 20)})
grid.fit(x, y)
```
GridSearchCV(cv=10, estimator=SQRTLasso(),
             param_grid={'alpha': array([0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
       0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
       0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
       0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ])},
             scoring='neg_mean_squared_error')
             
```Python
grid = GridSearchCV(estimator=model_elasticnet,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 15)})
grid.fit(x, y)

```
GridSearchCV(cv=10,
             estimator=ElasticNet(alpha=0.5, fit_intercept=False,
                                  max_iter=10000),
             param_grid={'alpha': array([0.        , 0.07142857, 0.14285714, 0.21428571, 0.28571429,
       0.35714286, 0.42857143, 0.5       , 0.57142857, 0.64285714,
       0.71428571, 0.78571429, 0.85714286, 0.92857143, 1.        ])},
             scoring='neg_mean_squared_error')
             
Let us use the Lasso Classifier. our alpha will be between 0 and 1.

```Python
grid = GridSearchCV(estimator=model_lasso,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 10)})
grid.fit(x, y)
```
GridSearchCV(cv=10,
             estimator=Lasso(alpha=0.1, fit_intercept=False, max_iter=10000),
             param_grid={'alpha': array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
       0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])},
             scoring='neg_mean_squared_error')
             
Let us use the Ridge Classifier. our alpha will be between 0 and 1.
             
```Python
grid = GridSearchCV(estimator=model_ridge,cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0, 1, 5)})
grid.fit(x, y)
```
GridSearchCV(cv=10,
             estimator=Ridge(alpha=0.1, fit_intercept=False, max_iter=10000),
             param_grid={'alpha': array([0.  , 0.25, 0.5 , 0.75, 1.  ])},
             scoring='neg_mean_squared_error')

After applying gridsearch to all our hyper-parameter methods, we are going to design a validation model which will help us to find the prediction error for methods. With this validation model we will not scale our data because our simmulated data does not need to be scaled.

```Python
def validate(model,x,y,nfolds,rs):
  kf = KFold(n_splits=nfolds,shuffle=True,random_state=rs)
  PE = []
  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    xtest = x[idxtest]
    ytest = y[idxtest]
    model.fit(xtrain,ytrain)
    PE.append(mean_absolute_error(ytest,model.predict(xtest)))
  return np.mean(PE)
```


```Python
#SQLTLasso
validate(model1,x,y,10,123)
```
8.065411489186724

```Python
validate(model_lasso,x,y,10,123)
```
3.3017017342248822

```Python
validate(model_ridge,x,y,10,123)
```
4.808266134153092

```Python
validate(model_elasticnet,x,y,10,123)
```
3.2997666058322865

After applying our validation method, we would conclude that Elastic Net and Lasso has the best performance compared to other variable selection methods.
