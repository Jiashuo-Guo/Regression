import numpy as np
from sigmoide import *


def calcule_cout(X, R, theta,nb_mod):
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
    a = R.dot((np.log(sigmoide(X,theta,nb_mod))).T)+(np.ones((N,nb_mod))-R).dot((np.log(np.ones((N,nb_mod))-sigmoide(X,theta,nb_mod)).T))
    cout = -np.trace(a)
    return cout
