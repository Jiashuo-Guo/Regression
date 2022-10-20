import numpy as np
from scipy.special import softmax

def decoupage_donnees(X,Y,R,prop_test=0.5):
     # Découpe les données initiales en DEUX sous-ensembles distincts d'apprentissage Det de test
    
    
    N = X.shape[0]
    nb_var = X.shape[1]
    X_app = X[:int((1-prop_test)*N),:]
    Y_app = Y[:int((1-prop_test)*N),:]
    R_app = R[:int((1-prop_test)*N),:]
   
    X_test = X[int((1-prop_test)*N):,:]
    Y_test = Y[int((1-prop_test)*N):,:]

    return X_app, Y_app, R_app, X_test,Y_test
    

