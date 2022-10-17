import numpy as np
from sigmoide import *


def calcule_cout(X, Y, theta):
    """ Calcule la valeur de la fonction cout (moyenne des différences au carré)
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Return
    -------
    cout : nombre correspondant à la valeur de la fonction cout (moyenne des différences au carré)

    """
    N = X.shape[0]
    cout = -(Y*np.log(sigmoide(X,theta))+(np.ones((N,1))-Y)*np.log(np.ones((N,1))-sigmoide(X,theta))).sum(0)

    return cout
