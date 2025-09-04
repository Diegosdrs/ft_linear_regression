# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/04 11:35:04 by dsindres          #+#    #+#              #
#    Updated: 2025/09/04 15:00:51 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if not isinstance (thetas, np.ndarray):
            self.thetas = np.array(thetas, dtype=float)
        else:
            self.thetas = thetas.astype(float)

    def fit_(self, x, y):
        m = len(y)  # Nombre d'exemples
        x_ = np.c_[np.ones((m, 1)), x]  # Ajouter la colonne des 1 pour le biais (θ0)
        
        # Effectuer la descente de gradient
        for _ in range(self.max_iter):
            gradient = (1 / m) * x_.T.dot(x_.dot(self.thetas) - y)  # Calcul du gradient
            self.thetas -= self.alpha * gradient  # Mise à jour des paramètres
        return self.thetas

    def minmax(self, x):
        if not isinstance (x, np.ndarray):
            return None
        if x.size == 0:
            return None
        new_x = (x - x.min()) / (x.max() - x.min())
        return new_x