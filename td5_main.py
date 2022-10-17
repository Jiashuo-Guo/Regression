import matplotlib.pyplot as plt
import numpy as np
from lecture_donnees import *
from normalisation import *
from descente_gradient import *
from affichage import *
from prediction import *
from taux_classification import *


# ===================== Partie 1: Lecture et normalisation des données=====================
print("Lecture des données ...")

X, Y, N, nb_var = lecture_donnees("TD5-donnees/notes.txt")


# Affichage des 10 premiers exemples du dataset
print("Affichage des 10 premiers exemples du dataset : ")
for i in range(0, 10):
    print(f"x = {X[i,:]}, y = {Y[i]}")
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")

X, mu, sigma = normalisation(X)

# Ajout d'une colonne de 1 à X (pour theta0)
X = np.hstack((np.ones((N,1)), X)) 

# Affichage des points en 2D et représentation de leur classe réelle par une couleur
if nb_var == 2 :
    plt.figure(0)
    plt.title("Disposition des points en 2D")
    affichage(X,Y)

# ===================== Partie 2: Descente du gradient =====================
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.001
nb_iters = 5000

# Initialisation de theta et réalisation de la descente du gradient
theta = np.zeros((1,nb_var+1))

theta, J_history = descente_gradient(X, Y, theta, alpha, nb_iters)

# Affichage de l'évolution de la fonction de cout lors de la descente du gradient
plt.figure(1)
plt.title("Evolution de le fonction de cout lors de la descente du gradient")
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel("Nombre d'iterations")
plt.ylabel("Cout J")

# Affichage de la valeur de theta
print(f"Theta calculé par la descente du gradient : {theta}")

# Evaluation du modèle
Ypred = prediction(X,theta)

# print("Taux de classification : ", taux_classification(Ypred,Y))

# Affichage des points en 2D et représentation de leur classe prédite par une couleur
if nb_var == 2 :
    plt.figure(2)
    plt.title("Disposition des points en 2D")
    affichage(X,Ypred)
    
plt.show()

print("Regression logistique Terminée.")