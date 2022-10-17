import numpy as np


def normalisation(X):
    """
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    X_norm : matrice des données centrées-réduites de dimension [N, nb_var]
    mu : moyenne des variables de dimension [1,nb_var]
    sigma : écart-type des variables de dimension [1,nb_var]

    """

    mu = X.mean(0)
    sigma = X.std(0)
    X_norm = (X-mu)/sigma


    return X_norm, mu, sigma


