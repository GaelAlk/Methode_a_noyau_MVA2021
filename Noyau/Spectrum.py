import numpy as np
import pandas as pd
from itertools import product

import scipy
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix,csc_matrix



def calculpremier(Toutescombinaisons):
    """
    Nous abordons une approche «semi-Tree»
    Nous utilisons la fonction calculpremier et calculsecond 
    pour determiner rapidement la position d'une occurence 
    dans la liste de toutes les possibilités 
    """
    i=0
    ACGT=[]
    while(Toutescombinaisons[i][0]=='A'):
        i=i+1
    ACGT.append(i)
    while(Toutescombinaisons[i][0]=='C'):
        i=i+1
    ACGT.append(i)
    while(Toutescombinaisons[i][0]=='G'):
        i=i+1
    ACGT.append(i)

    return ACGT

def calculsecond(Toutescombinaisons,ACG):
    i=0
    ACGT=[]
    ACGT.append(calculpremier(Toutescombinaisons[0:ACG[0]]))
    ACGT.append(calculpremier(Toutescombinaisons[ACG[0]:ACG[1]]))
    ACGT.append(calculpremier(Toutescombinaisons[ACG[1]:ACG[2]]))
    ACGT.append(calculpremier(Toutescombinaisons[ACG[2]:]))

    return ACGT

def trouvelindice(Toutescombinaisons, phrase,ACG,ACGT2,block=False):
    """
    Fonction qui trouve l'indice dans la cible plus rapidement pour un grand k.
    """
    n=len(Toutescombinaisons)
    taille=2
    if(phrase[0]=='A'):
        #base=ACG[0]
        phrasee=phrase[1:]
        ACGT=ACGT2[0]
        if(phrasee[0]=='A'):

            return Toutescombinaisons[0:ACGT[0]].index(phrase)
        if(phrasee[0]=='C'):
            return ACGT[0]+Toutescombinaisons[ACGT[0]:ACGT[1]].index(phrase)
        if(phrasee[0]=='G'):
            return ACGT[1]+Toutescombinaisons[ACGT[1]:ACGT[2]].index(phrase)
        if(phrasee[0]=='T'):
            return ACGT[2]+Toutescombinaisons[ACGT[2]:ACG[0]].index(phrase)

        
    if(phrase[0]=='C'):
        base=ACG[0]
        phrasee=phrase[1:]
        ACGT=ACGT2[1]
        if(phrasee[0]=='A'):

            return base+Toutescombinaisons[base:base+ACGT[0]].index(phrase)
        if(phrasee[0]=='C'):
            return base+ACGT[0]+Toutescombinaisons[base+ACGT[0]:base+ACGT[1]].index(phrase)
        if(phrasee[0]=='G'):
            return base+ACGT[1]+Toutescombinaisons[base+ACGT[1]:base+ACGT[2]].index(phrase)
        if(phrasee[0]=='T'):
            return base+ACGT[2]+Toutescombinaisons[base+ACGT[2]:base+ACG[1]].index(phrase)
        
    if(phrase[0]=='G'):
        base=ACG[1]
        phrasee=phrase[1:]
        ACGT=ACGT2[2]
        if(phrasee[0]=='A'):

            return base+Toutescombinaisons[base:base+ACGT[0]].index(phrase)
        if(phrasee[0]=='C'):
            return base+ACGT[0]+Toutescombinaisons[base+ACGT[0]:base+ACGT[1]].index(phrase)
        if(phrasee[0]=='G'):
            return base+ACGT[1]+Toutescombinaisons[base+ACGT[1]:base+ACGT[2]].index(phrase)
        if(phrasee[0]=='T'):
            return base+ACGT[2]+Toutescombinaisons[base+ACGT[2]:base+ACG[2]].index(phrase)
    
    
    if(phrase[0]=='T'):
        base=ACG[2]
        phrasee=phrase[1:]
        ACGT=ACGT2[3]
        if(phrasee[0]=='A'):

            return base+Toutescombinaisons[base:base+ACGT[0]].index(phrase)
        if(phrasee[0]=='C'):
            return base+ACGT[0]+Toutescombinaisons[base+ACGT[0]:base+ACGT[1]].index(phrase)
        if(phrasee[0]=='G'):
            return base+ACGT[1]+Toutescombinaisons[base+ACGT[1]:base+ACGT[2]].index(phrase)
        if(phrasee[0]=='T'):
            return base+ACGT[2]+Toutescombinaisons[base+ACGT[2]:n].index(phrase)


def DistributionOccurence(n,donnees,Toutescombinaisons):
    '''
    Calcule les représentations dans l'espace associé à ce noyau.
    '''

    spectrum= list(zip(*[donnees[i:] for i in range(n)]))
    occurence = np.zeros(taille)

    for string in spectrum:
        index = trouvelindice(Toutescombinaisons,string,ACGT,ACGT2)
        occurence[index] = occurence[index]+1
        decoupe_llist = list(string)
        for ind,actuel in enumerate(decoupe_llist):
            for ai in Alphabet:
                if ai!=actuel:
                    templist = list(decoupe_llist)
                    templist[ind]= ai
                    mismatch_ngram = tuple(templist)
                    index = trouvelindice(Toutescombinaisons,mismatch_ngram,ACGT,ACGT2)
                    occurence[index] = occurence[index]+0.1

    return occurence

"""
Calculs préliminaires. Il suffit de changer k_taille pour changer le k de k spectrum. 
"""
Alphabet=['A','C','G','T']
k_taille = 6 
combinatoireADN = list(product(Alphabet,repeat=k_taille))
taille=len(combinatoireADN)

ToutescombARRAY=np.array(combinatoireADN)

ACGT=calculpremier(combinatoireADN)
ACGT2=calculsecond(ToutescombARRAY[:,1:],ACGT)

def calcul_K_spectrum(donnees_tr, donnees_te, train=True):
    'donnees_tr= train, donnees_te: test, train: are we building test gram matrix or train gram matrix'
    
    X_histo_0= np.empty([len(donnees_tr),len(combinatoireADN)])

    for i in range(len(donnees_tr)):
        X_histo_0[i,:] = DistributionOccurence(k_taille,donnees_tr[i],combinatoireADN)
        
    if train==True:
        K= X_histo_0@X_histo_0.T
        
    else:
        X_histo_1= np.empty([len(donnees_te),len(combinatoireADN)])
        for i in range(len(donnees_te)):
            X_histo_1[i,:] = DistributionOccurence(k_taille,donnees_te[i],combinatoireADN)
        K=X_histo_0@X_histo_1.T
        
    return K.astype(float)