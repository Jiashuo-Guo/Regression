import numpy as np
from sigmoide import *


def prediction(X,theta):
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
    for i in range(N):
        if sigmoide(X,theta)[i]>0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p


