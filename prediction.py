import numpy as np
from sigmoide import *


def prediction(X,theta,nb_mod):
    """ Predit la classe de chaque élement de X
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives


    Retour
    -------
    p : matrices de dimension [N,1] indiquant la classe de chaque élement de X (soit 0, soit 1)

    """

    N = X.shape[0]
    p = np.zeros((N,1))
    a = sigmoide(X,theta,nb_mod)
    am = a.max(1)
    for i in range(N):
        for j in range(nb_mod):
            if a[i,j] == am[i]:
                p[i] = j

    return p


