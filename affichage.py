import matplotlib.pyplot as plt
import numpy as np


def affichage(X, Y):
    """ Affichage en 2 dimensions des données et de la courbe de régression linéaire déterminée par theta
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    None

    """
    N = X.shape[0]
    plt.figure()
    apparance = ['bx','ro','g,','y<','kp']
    for i in range(N):
        for j in range(int(Y.max())+1):
            if Y[i] == j:
                plt.plot(X[i,1],X[i,2],apparance[j])
    plt.show()
    