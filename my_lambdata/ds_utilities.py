import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from pdb import set_trace as breakpoint

def enlarge(n):
    ''' this function will multiply the input by 100'''
    return n * 2000

def train_validation_test_split(X, y, train_size=0.7, val_size=0.1, test_size=0.2, random_state=None, shuffle=True):
          
   X_train_val, X_test, y_train_val, y_test = train_test_split(
       X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
  
   X_train, X_val, y_train, y_val = train_test_split(
       X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
       random_state=random_state, shuffle=shuffle)
  
   return X_train, X_val, X_test, y_train, y_val, y_test

class MyLinearRegression:
    
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
    
    def __repr__(self):
        return "I am a Linear Regression model!"

    def fit(self, X, y):
        """
        Fit model coefficients.

        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # add bias if fit_intercept is True
        if self._fit_intercept:
            X_biased = np.c_[np.ones(X.shape[0]), X]
        else:
            X_biased = X
        
        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)
        
        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

if __name__ == '__main__':
    y = int(input("Choose a number: "))
    print(y, enlarge(y))

    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']
    #breakpoint()

    X = 10*np.random.random(size=(20,2))
    y = 3.5*X.T[0]-1.2*X.T[1]+2*np.random.randn(20)

import numpy as np

fig, ax = plt.subplots(1,2,figsize=(10,3))

ax[0].scatter(X.T[0],y)
ax[0].set_title("Output vs. first feature")
ax[0].grid(True)
ax[1].scatter(X.T[1],y)
ax[1].set_title("Output vs. second feature")
ax[1].grid(True)
fig.tight_layout()
plt.show()   

mlr = MyLinearRegression()
print(df.shape)
print(mlr)
    
X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
    df[['ash', 'hue']], df['target'])