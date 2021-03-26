import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

##################################
#######Import code propre#########
##################################


from Algorithme.SVM import SVM
from Algorithme.KLR import KLR


##################################
########Import des données########
##################################

print("Import des données..")

"""
X_train_0= pd.read_csv("data/Xtr0_mat100.csv", delim_whitespace=True,header=None).values
X_train_1= pd.read_csv("data/Xtr1_mat100.csv", delim_whitespace=True,header=None).values
X_train_2= pd.read_csv("data/Xtr2_mat100.csv", delim_whitespace=True,header=None).values



X_test_0= pd.read_csv("data/Xte0_mat100.csv", delim_whitespace=True,header=None).values
X_test_1= pd.read_csv("data/Xte1_mat100.csv", delim_whitespace=True,header=None).values
X_test_2= pd.read_csv("data/Xte2_mat100.csv", delim_whitespace=True,header=None).values"""


labels0= pd.read_csv("data/Ytr0.csv",index_col=0).values.ravel() # to convert to 2000 not 2000,1
labels1= pd.read_csv("data/Ytr1.csv",index_col=0).values.ravel()
labels2= pd.read_csv("data/Ytr2.csv",index_col=0).values.ravel()

#On transforme les labels en -1,+1
labels0=2*labels0-1
labels1=2*labels1-1
labels2=2*labels2-1


X_train_0 = (pd.read_csv('data/Xtr0.csv',header=None).values).tolist()
X_train_1 = (pd.read_csv('data/Xtr1.csv',header=None).values).tolist()
X_train_2 = (pd.read_csv('data/Xtr2.csv',header=None).values).tolist()

X_train_0 = np.array(X_train_0)[1:,1] #On garde seulement les séquences
X_train_1 = np.array(X_train_1)[1:,1]
X_train_2 = np.array(X_train_2)[1:,1]

#Test
X_test_0 = (pd.read_csv('data/Xte0.csv',header=None).values).tolist()
X_test_1 = (pd.read_csv('data/Xte1.csv',header=None).values).tolist()
X_test_2 = (pd.read_csv('data/Xte2.csv',header=None).values).tolist()

print("Terminé.\n")
##################################
##########Calcul du SVM###########
##################################
#0.007,0.005,0.005

best_C_0,best_C_1,best_C_2 =  0.006,0.004,0.004
k_taille=6

print(f"Calcul du SVM : Jeu de données numéro 1 pour C={best_C_0} et noyau {k_taille}-spectrum. Veuillez patienter..")
_a,_b, val0_main =SVM(1,best_C_0,np.array(labels0),np.array(X_train_0),np.array(X_test_0)[1:,1], True)
print("Terminé.\n")


print(f"Calcul du SVM : Jeu de données numéro 2 pour C={best_C_1} et noyau {k_taille}-spectrum. Veuillez patienter..")
_a,_b, val1_main =SVM(1,best_C_1,np.array(labels1),np.array(X_train_1),np.array(X_test_1)[1:,1], True)
print("Terminé.\n")


print(f"Calcul du SVM : Jeu de données numéro 3 pour C={best_C_2} et noyau {k_taille}-spectrum. Veuillez patienter..")
_a,_b, val2_main =SVM(1,best_C_2,np.array(labels2),np.array(X_train_2),np.array(X_test_2)[1:,1], True)
print("Terminé.\n")


"""

Exemple d'utilisation du KLR : 
print(f"Calcul du KLR : Jeu de données numéro 1 pour C={best_C_0} et noyau {k_taille}-spectrum. Veuillez patienter..")
_a,_b, val0_main =KLR(0.001,np.array(labels0),np.array(X_train_0),np.array(X_test_0), False,0.5,0.001)
print("Terminé.\n")
print(_a)
print(_b)
"""
##################################
##########Export du CSV###########
##################################



prediction=np.concatenate((val0_main,val1_main,val2_main)).astype(int)

Prediction=pd.DataFrame(prediction,index=range(3000),columns=["Bound"])
Prediction['Id'] = [i for i in range(3000)]
Prediction.to_csv("Yte.csv",index=False,header=True)

print("Les predictions sont disponibles dans le fichier Yte.csv .")
