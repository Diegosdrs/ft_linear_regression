# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/04 10:34:57 by dsindres          #+#    #+#              #
#    Updated: 2025/09/04 13:30:23 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR


if __name__ == "__main__":
    
    print("\n ~~~~ Bienvenu à la magnifique Régression linéaire ! ~~~~\n\n")
    print("      Ici nous prédirons le prix de ta future voiture !\n")
    
    try:
        km = float(input("      Quelle est son kilométrage ? "))
    except ValueError:
        print("\n      Erreur : tu dois entrer un nombre valide !\n")
        exit(1)
        
    file_thetas = Path("thetas.csv")
    if file_thetas.exists():
        if file_thetas.stat().st_size == 0:
            print("\n      Nous avons trouver un prix de 0 euro... \n      Mais désolé de te faire une fausse joie le calcul n'est pas bon.")
            print("      Entraine notre modèle avec le second programme pour obtenir la bonne valeur !\n")
        else:
            data = pd.read_csv("thetas.csv")
            theta_0 = data['theta_0'].iloc[0]
            theta_1 = data['theta_1'].iloc[0]

            #normaliser le km de data.csv
            data_2 = pd.read_csv("data.csv")
            data_km = np.array(data_2[["km"]])
            data_price = np.array(data_2[["price"]])
            
            km_norm = (km - data_km.min()) / (data_km.max() - data_km.min())
            
            prix_final_norm = theta_0 + theta_1 * km_norm

            #denormaliser le prix sinon valeur tjrs entre 0 et 1
            prix = prix_final_norm * (data_price.max() - data_price.min()) + data_price.min()
            print(f"      Son prix est de {prix:.2f} euros\n")
            
    else:
        print("\n      Nous avons trouver un prix de 0 euro... \n      Mais désolé de te faire une fausse joie le calcul n'est pas bon.")
        print("      Entraine notre modèle avec le second programme pour obtenir la bonne valeur !\n")
