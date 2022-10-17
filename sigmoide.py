import numpy as np


def sigmoide(z,theta):
    """ Calcule la valeur de la fonction sigmoide appliquée à z
    
    Parametres
    ----------
    z : peut être un scalaire ou une matrice

    Return
    -------
    s : valeur de sigmoide de z. Même dimension que z

    """
    a = np.exp(-z.dot(theta.T))
    N = a.shape[0]
    s = np.ones((N,1))/(np.ones((N,1))+a)

    return s
