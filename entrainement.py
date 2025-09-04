# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    entrainement.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/04 10:29:44 by dsindres          #+#    #+#              #
#    Updated: 2025/09/04 15:01:33 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    X = np.array(data[["km"]])
    Y = np.array(data[["price"]])
    
    # on commence avec thetas 0 = 0 et theta 1 = 0
    myLR = MyLR(thetas = [[0.0], [0.0]], alpha = 0.1, max_iter = 100000)

    # normaliser les valeurs 
    X_norm = myLR.minmax(X)
    Y_norm = myLR.minmax(Y)
    
    # calcul des bonnes valeurs de thetas grace a la machine 
    thetas = myLR.fit_(X_norm, Y_norm)
    print(f"thetas = \n{thetas}")

    #mettre les thetas dans le fichier csv pour le programme 0
    df = pd.DataFrame({
        "theta_0": thetas[0],
        "theta_1": thetas[1]
    })

    df.to_csv("thetas.csv", index=False)

    
    
    