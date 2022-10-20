import matplotlib.pyplot as plt
import numpy as np

from lecture_donnees import *
from normalisation import *
from descente_gradient import *
from affichage import *
from prediction import *
from taux_classification import *
from decoupage_donnees import *



# ===================== Partie 1: Lecture et normalisation des données=====================

print("Lecture des données ...")

X, Y, N, nb_var, R, nb_mod = lecture_donnees("wine.txt")
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
# if nb_var == 2 :
affichage(X,Y)

# Découpage des données en sous-ensemble d'apprentissage, de validation et de test
X_app, Y_app, R_app, X_test,Y_test =  decoupage_donnees(X,Y,R)

# ===================== Partie 2: Descente du gradient =====================
affichage(X_app,Y_app)
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.01
nb_iters = 5000

# Initialisation de theta et réalisation de la descente du gradient
theta = np.zeros((nb_mod,nb_var+1))

theta, J_history = descente_gradient(X_app, R_app, theta, alpha, nb_iters,nb_mod)

# Affichage de l'évolution de la fonction de cout lors de la descente du gradient
plt.figure(1)
plt.title("Evolution de le fonction de cout lors de la descente du gradient")
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel("Nombre d'iterations")
plt.ylabel("Cout J")

# Affichage de la valeur de theta
print(f"Theta calculé par la descente du gradient : {theta}")
print(sigmoide(X_app,theta,nb_mod))
# Evaluation du modèle
Ypred = prediction(X_app,theta,nb_mod)
print(Ypred)
# print("Taux de classification : ", taux_classification(Ypred,Y))

# Affichage des points en 2D et représentation de leur classe prédite par une couleur
# if nb_var == 2 :

affichage(X_app,Ypred)
plt.show()

print("Regression logistique Terminée.")

# ===================== Partie 3: Test =====================
affichage(X_test,Y_test)

Ypred = prediction(X_test,theta,nb_mod)
print(Ypred)

affichage(X_test,Ypred)
plt.show()

accuracy = (Ypred == Y_test).astype(int).sum() / Ypred.shape[0]
print("Resultat du test:", accuracy)