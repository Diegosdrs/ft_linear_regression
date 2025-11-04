# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    bonus.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/04 13:30:39 by dsindres          #+#    #+#              #
#    Updated: 2025/11/04 13:21:32 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

def predict_(x, thetas):
    # Vérification des types
    if not isinstance(x, np.ndarray) or not isinstance(thetas, np.ndarray):
        return None

    # Vérification des tailles vides
    if x.size == 0 or thetas.size == 0:
        return None

    # Vérification des dimensions
    if len(x.shape) != 2 or thetas.shape != (x.shape[1] + 1, 1):
        return None

    # Étape 1 : Ajout d'une colonne de 1 au début de x pour obtenir X'
    ones = np.ones((x.shape[0], 1))
    x_prime = np.concatenate((ones, x), axis=1)

    # Étape 2 : Multiplication matricielle entre X' et thetas
    y_hat = x_prime.dot(thetas)

    return y_hat

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    X = np.array(data[["km"]])
    Y = np.array(data[["price"]])       

    data = pd.read_csv("thetas.csv")
    theta_0 = data['theta_0'].iloc[0]
    theta_1 = data['theta_1'].iloc[0]
    theta_arr = np.array([[theta_0], [theta_1]])

    X_norm = (X - X.min()) / (X.max() - X.min())
    Y_pred = predict_(X_norm, theta_arr)

    # denormaliser Y_pred
    Y_real = Y_pred * (Y.max() - Y.min()) + Y.min()

    perte = loss_(Y_real, Y)
    rmse = np.sqrt(2 * perte)
    print(f"      L'erreur moyenne de prix est de {rmse:.2f} euros")

             
    # Tracé du graphe
    plt.scatter(X, Y, color='blue', label='Prix réels') # Données réelles
    plt.plot(X, Y_real, color='red', label='Prix prédits') # Régression linéaire (droite)
    plt.title('Prix des voitures en fonction du kilometrage')
    plt.xlabel('Km')
    plt.ylabel('Prix estime')
    plt.legend()
    plt.grid(True)
    plt.show()