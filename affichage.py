import matplotlib.pyplot as plt
import numpy as np

def acp(X):
    
    # Calcul de la matrice M (métrique)
    M = np.eye(X.shape[1])
    
    # Calcul de la matrice D (poid des individus)
    D = np.eye(X.shape[0]) / X.shape[0]
    
    # Calcul de la matrice de covariance pour les individus
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    val_p_ind = np.sort(L)[::-1]
    vect_p_ind = U[:,indices]
    
    # Calcul des facteurs pour les individus
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
    apparance = ['bx','ro','g,','y<','kp']
    for i in range(N):
        for j in range(int(Y.max())+1):
            if Y[i] == j:
                plt.plot(X[i,1],X[i,2],apparance[j])
    plt.show()
    