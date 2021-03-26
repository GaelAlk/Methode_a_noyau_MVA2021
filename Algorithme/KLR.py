import numpy as np 
import random
#import seaborn as sns
#from Utils import *

###########"Définition de fonctions simples mais utilisés fréquemment
sigmoid = lambda x: 1./(1+np.exp(-x)) # sigmoid
lprimelog = lambda x: sigmoid(-x)
lprime2log = lambda x : sigmoid(x)*sigmoid(-x)
eta = lambda x,w: sigmoid(np.dot(w.transpose(),x)) #fonction eta du rapport
inverse_sigmoid = lambda x: math.log(1/x-1)  #inverse de la sigmoid
#####################################################################



from Noyau.Spectrum import calcul_K_spectrum
from Noyau.Gaussien import calcul_K


def KLR(sigmaa,y,donnees,test,spectrum,lamb,tol):
    """
    Algorithme de Kernel Regression Logistic implémenté pour un noyau gaussien et k-Spectrum.
    
    Entrée : 
    
    sigmaa : float,  Variance du noyau gaussien
    donnees : np.array, Ensemble d'entrainement 
    test : np.array, Ensemble de test sur lequel les prédictions sont calculées
    spectrum : bool, True pour calculer le noyau k-spectrum False pour noyau gaussien
    lamb : Parametre lambda de régularisation (WKRR)
    tol : Tolérance
    
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
    Résolution du KLR
    """
    lbd=lamb
    a = np.ones((n,1))
    ancien_a=43*a

    Y=np.expand_dims(y,1)
    while(np.linalg.norm(a-ancien_a)>tol):
        ancien_a = a
        m = K0@a
        P = np.diag(-lprimelog(-m*Y).ravel())
        W = np.sqrt(np.diag(lprime2log(m*Y).ravel()))
        z= m+Y/sigmoid(-m*Y)
        a = W@(np.linalg.inv(W@K0@W+n*lamb*np.eye(n)))@W@z

        print(np.linalg.norm(a-ancien_a))
    
    
    
    """
    Prédiction sur l'ensemble d'entraînement
    """
    a=a.squeeze()
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
        val_main.append(np.dot(a,K1[:,i]))
    val_main=sigmoid(np.array(val_main))
    val_main=np.where(val_main<0.5,np.zeros(val_main.shape[0]),val_main) # to fit to 0/1 values
    val_main=np.where(val_main>0.5,np.ones(val_main.shape[0]),val_main) # to fit to 0/1 values    
    return np.mean(prediction==y) , np.mean(val_main) , val_main