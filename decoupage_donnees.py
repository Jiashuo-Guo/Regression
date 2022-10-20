import numpy as np
from scipy.special import softmax

def decoupage_donnees(X,Y,R,prop_test=0.5):
    """ Découpe les données initiales en trois sous-ensembles distincts d'apprentissage, de validation et de test
    
    Parametres
    ----------
    x : matrice des données de dimension [N, nb_var]
    d : matrice des valeurs cibles [N, nb_cible]
    prop_val : proportion des données de validation sur l'ensemble des données (entre 0 et 1)
    prop_test : proportion des données de test sur l'ensemble des données (entre 0 et 1)
    
    avec N : nombre d'éléments, nb_var : nombre de variables prédictives, nb_cible : nombre de variables cibles

    Retour
    -------
    x_app : matrice des données d'apprentissage
    d_app : matrice des valeurs cibles d'apprentissage
    
    x_test : matrice des données d'apprentissage
    d_test : matrice des valeurs cibles d'apprentissage

    """
    #######################
    ##### A compléter ##### 
    #######################
    N = X.shape[0]
    nb_var = X.shape[1]
    X_app = X[:int((1-prop_test)*N),:]
    Y_app = Y[:int((1-prop_test)*N),:]
    R_app = R[:int((1-prop_test)*N),:]
   
    X_test = X[int((1-prop_test)*N):,:]
    Y_test = Y[int((1-prop_test)*N):,:]

    

    
    
    

    return X_app, Y_app, R_app, X_test,Y_test
    

