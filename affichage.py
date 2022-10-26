import matplotlib.pyplot as plt
import numpy as np

def acp(X):
    
    M = np.eye(X.shape[1])
    D = np.eye(X.shape[0]) / X.shape[0]
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    L,U = np.linalg.eig(Xcov_ind)
    indices = np.argsort(L)[::-1]
    val_p_ind = np.sort(L)[::-1]
    vect_p_ind = U[:,indices]
    fact_ind = X.dot(M.dot(vect_p_ind))
    return fact_ind 

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
    X = acp(X)
    N = X.shape[0]
    plt.figure()
    apparance = ['bx','ro','g<','y<','kp']
    for i in range(N):
        for j in range(int(Y.max())+1):
            if Y[i] == j:
                plt.plot(X[i,1],X[i,2],apparance[j])
    plt.show()
    