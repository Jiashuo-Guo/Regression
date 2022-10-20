import numpy as np

def lecture_donnees(nom_fichier, delimiteur=','):
    """ Lit le fichier contenant les données et renvoiee les matrices correspondant

    Parametres
    ----------
    nom_fichier : nom du fichier contenant les données
    delimiteur : caratère délimitant les colonne dans le fichier ("," par défaut)

    Retour
    -------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    """
    data = np.loadtxt(nom_fichier,delimiter=delimiteur)
    nb_var = data.shape[1]-1
    N = data.shape[0]
    X = data[:,:-1]
    Y = data[:,-1].reshape(N,1)
    Y = Y-Y.min(0)
    R = np.zeros((N,int(Y.max())+1))
    for i in range(N):
        for j in range(int(Y.max())+1):
            if Y[i] == j:
                R[i,j] = 1
    nb_mod = int(Y.max())+1
    return X, Y, N, nb_var, R,nb_mod
