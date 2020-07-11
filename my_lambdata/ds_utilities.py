import pandas as pd
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

if __name__ == '__main__':
    #y = int(input("Choose a number: "))
    #print(y, enlarge(y))

    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']
    #breakpoint()

    print(df.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(df[['ash', 'hue']], df['target'])