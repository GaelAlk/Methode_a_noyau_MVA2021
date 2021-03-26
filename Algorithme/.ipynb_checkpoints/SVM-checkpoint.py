import numpy as np
import pandas as pd

import cvxopt
import cvxopt.solvers

from Noyau.Spectrum import calcul_K_spectrum
from Noyau.Gaussien import calcul_K


def SVM(sigmaa,C,y,donnees,test,spectrum):
    """
    Algorithme de C-SVM implémenté pour un noyau gaussien et k-Spectrum.
    Nous utilisons cvxopt pour résoudre le problème QP.
    
    Entrée : 
    
    sigmaa : float,  Variance du noyau gaussien
    C : float, Paramètre C du C-SVM
    donnees : np.array, Ensemble d'entrainement 
    test : np.array, Ensemble de test sur lequel les prédictions sont calculées
    spectrum : bool, True pour calculer le noyau k-spectrum False pour noyau gaussien
    
    Sortie : 
    
    Prediction sur l'ensemble d'entrainement,
    Pondérations des labels sur l'ensemble de test,
    Prédictions sur l'ensemble de test. 
    
    """
    
    """
    Calcul de la matrice de Gram
    """
    if(spectrum==True):
        K0=calcul_K_spectrum(donnees,donnees, train=True)
    else:
        K0=calcul_K(sigmaa,donnees,donnees)

    n=donnees.shape[0]

    """
    Résolution du SVM
    """
    
    K = cvxopt.matrix(K0, tc='d')
    my = cvxopt.matrix(-y, tc='d')
    G = cvxopt.matrix(np.vstack((np.diag(y),np.diag(-y))), tc='d')
    h = cvxopt.matrix(np.hstack((C*np.ones(n),np.zeros(n))), tc='d')
    sol = cvxopt.solvers.qp(K, my, G, h)
    a = np.ravel(sol['x'])
    
    """
    Prédiction sur l'ensemble d'entraînement
    """
    prediction=[]
    for i in np.arange(n):
        prediction.append(np.sign(np.dot(a,K0[:,i])))
    prediction=np.array(prediction,dtype=int)

    """
    Calcul matrice de Gram pour prédire sur l'ensemble de test 
    """
    
    if(spectrum==True):
        K1=calcul_K_spectrum(donnees,test, train=False)
    else:
        K1=calcul_K(sigmaa,donnees,test)

    n=test.shape[0]
    val_main=[]
    for i in np.arange(n):
        val_main.append(np.sign(np.dot(a,K1[:,i])))
    val_main=np.array(val_main,dtype=int)
    val_main=np.where(np.sign(val_main)==-1,np.zeros(val_main.shape[0]),np.sign(val_main)) # to fit to 0/1 values
    
    #print("Sur l'ensemble d'entrainement : ",np.mean(prediction==y) )
    #print("Et sinon : ", np.mean(val_main))
    
    return np.mean(prediction==y) , np.mean(val_main) , val_main