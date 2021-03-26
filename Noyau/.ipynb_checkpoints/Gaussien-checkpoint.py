import scipy
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix,csc_matrix
import numpy as np
"""
def calcul_K(sigma2,donnees):
    pairwise_dists = squareform(pdist(donnees, 'euclidean'))
    K = np.exp(-pairwise_dists / (2*sigma2)) # parenthese was missing
    return K
"""




def calcul_K(sigma2,donnees1, donnees2):
 
    """
    #Calcule K pour un noyau gaussien de variance sigma2, donnees1: train donnees2: test. 
    """
    print(donnees2.shape)
    pairwise_dists = cdist(donnees1, donnees2, 'euclidean')
    
    K = np.exp(-pairwise_dists**2 / (2*sigma2))
    return K