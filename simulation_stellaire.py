import scipy.constants as sp
import numpy as np
import matplotlib.pyplot as plt
import sys


## Constantes ## 
 # Constantes du Soleil #
M_Sol = 1.988e30 # en kg 
R_Sol = 6.957e8 # en m 
L_Sol = 3.828e26 
 # Pi #
pi = np.pi
 # Boltzmann en J K⁻¹#
kb = 1.38e-23 
 # célérité de la lumière #
c=sp.c 
 # Constante de gravitation N.m²/kg²#
G = sp.G
# Constante de radiation #
a = 7.57e-16
 # Coefficient de Laplace #
gamma=5/3
 # Masse d'un atome d'hydrogene en kg #
mH = 1.67e-27

## éléments composants une étoile (ces proportions peuvent être variables) ##
 # proportion d'hydrogène #
X = 0.70 
 # proportion d'hélium #
Y = 0.28
 # proportion d'éléments plus lourd	#				  
Z = 0.02 
# poids moyen moléculaire #
mu = 4 / (3 + 5 * X - Z)  
# ou autre eq : 1.0 / (2.0 * X + 0.75 * Y + 0.5 * Z)

# Taux de génération d'énergie pour une chaine PP (proton-proton)
epsilon_pp = 2.6e-37 * X**2
# contribution du cycle CNO négligé
epsilon_CNO = 0
# Constante de l'opacité de Kramer
kappa0 = 4.3e21 * Z * (X + 1)

## Conditions limites du modèle stellaire ##
 # Pression moyenne au centre #
P_c = 1.0e15
 # Pression à la surface #
P_s = 1e8
 # Température moyenne au centre
T_c = 1.0e7
 # Température à la surface #
T_s = 5600.0
 # Luminosité
L_s = L_Sol
#L_s = L_Sol*(M_Et/M_Sol)


# tolerance de convergence acceptée pour les solutions  
tol = 1.0e-3

# paramètres de l'étoile #
#M_Et = input("Rentrez la masse de l'étoile sachant que celle du soleil est de 1.988e30 kg")
#M_Et=float(M_Et)
M_Et = M_Sol
# Relation masse-rayon des étoiles de la séquence principale # 
if M_Et < M_Sol:
    R_Et = R_Sol * (M_Et / M_Sol) ** 0.8
else:
    R_Et = R_Sol * (M_Et / M_Sol) ** 0.57

 # Rayon stellaire
R_s = 1.0 * R_Et




## nombre de points à évaluer et le nombre d'itérations pour la simulation ##
 # Nombre de points de la grille # 
dim = 10000
extra=0
#dim=input("Rentrez le nombre de points de la grille")
 # Nombre maximum d'itérations # 
numIter = 200 


# tableau des différences entre les itérations succesives
k_valeurs = np.zeros(numIter)
R_diff = np.zeros(numIter)
P_diff = np.zeros(numIter)
T_diff = np.zeros(numIter)
L_diff = np.zeros(numIter)


## Stockage des valeurs des variables (Masse : M, Rayon : R, Température : T, Pression : P, Luminosité : L, masse volumique : rho, opacité de Krammer : kappa, génération d'énergie : epsilon) ##
#Ici on intègre depuis le centre jusqu'au demi rayon, c'est-à-dire de l'intérieur vers l'extérieur #
M = np.zeros(dim)
R = np.zeros(dim)
T = np.zeros(dim)
P = np.zeros(dim)
L = np.zeros(dim)
rho = np.zeros(dim)
kappa = np.zeros(dim)
epsilon = np.zeros(dim)

# Ici on intègre depuis la Surface jusqu'au demi rayon, c'est-à-dire de l'extérieur vers l'intérieur #
MS = np.zeros(dim+extra)
RS = np.zeros(dim+extra)
TS = np.zeros(dim+extra)
PS = np.zeros(dim+extra)
LS = np.zeros(dim+extra)
rhoS = np.zeros(dim+extra)
kappaS = np.zeros(dim+extra)
epsilonS = np.zeros(dim+extra)

#Ici on intègre depuis le centre les variables après une variations des conditions au centre pour faire converger le système # 
Mv = np.zeros(dim)
Rv = np.zeros(dim)
Tv = np.zeros(dim)
Pv = np.zeros(dim)
Lv = np.zeros(dim)
rhov = np.zeros(dim)
kappav = np.zeros(dim)
epsilonv = np.zeros(dim)


# Ici on intègre depuis la surface les variables après une variation des conditions à la surface pour faire converger le système  #
MSv = np.zeros(dim)
RSv = np.zeros(dim)
TSv = np.zeros(dim)
PSv = np.zeros(dim)
LSv = np.zeros(dim)
rhoSv = np.zeros(dim)
kappaSv = np.zeros(dim)
epsilonSv = np.zeros(dim)

 # Masse infinitésimale de la couche centrale, il vaut mieux prendre une très faible valeur plutôt que 0 # 
M[0] = 1.0e-06 * M_Et
 # Masse de chaque couche, dM représente la variation infinitésimale utilisée lors de l'intégration # 
dM = (M_Et - M[0]) / (dim - 1)

## Résolution ##
 
for k in range (0, numIter):
    P[0]=P_c
    T[0]=T_c
    rho[0] = P[0] * mu * mH / (kb * T[0])
    R[0] = (3.0 * M[0] / (4.0 * pi * rho[0])) ** (1.0 / 3.0)
    epsilon[0] = epsilon_pp * rho[0] * T[0] ** 4.0 + epsilon_CNO * rho[0] * T[0] ** 19.9
    L[0] = epsilon[0] * M[0]
    kappa[0] = kappa0 * rho[0] * T[0] ** (-3.5)
    
    # intégration à partir du centre jusqu'au demi rayon #
    for i in range(1, int(dim // 2)):
        R[i] = R[i - 1] + dM / (4.0 * pi * R[i - 1] ** 2 * rho[i - 1])
        P[i] = P[i - 1] - dM * G * M[i - 1] / (4.0 * pi * R[i - 1] ** 4)
        L[i] = L[i - 1] + dM * epsilon[i - 1]
        nabla_rad = (3.0* kappa[i - 1]* L[i - 1]* P[i - 1]/ (16.0 * pi * a * c * T[i - 1] ** 4 * G * M[i - 1])) 
    # Cette condition détermine si T se trouve dans la zone radiative ou convective et calcule la température appropriée #
        if nabla_rad < (gamma - 1) / gamma:
            # on définit la température de radiation #
            T[i] = T[i - 1] - (dM* 3.0* kappa[i - 1]* L[i - 1]/ (16.0 * pi * a * c * R[i - 1] ** 2 * T[i - 1] ** 3)/ (4.0 * pi * R[i - 1] ** 2))
        else:
            # on définit la température de convection #
            T[i] = T[i - 1] + (dM* (gamma - 1.0)/ gamma* T[i - 1]/ P[i - 1]* (P[i] - P[i - 1])/ dM)  
             
        # on vérifie que T et P > 0 #
        if T[i] <= 0.0 or P[i] <= 0.0:
            print("Température ou pression négative")
            break 
               
        M[i] = M[i - 1] + dM
        rho[i] = P[i] * mu * mH / (kb * T[i])
        epsilon[i] = (epsilon_pp * rho[i] * T[i] ** 4.0 + epsilon_CNO * rho[i] * T[i] ** 19.9)
        kappa[i] = kappa0 * rho[i] * T[i] ** (-3.5)

    # Conditions limites au niveau la surface
    MS[dim - 1] = M_Et
    LS[dim - 1] = L_s
    RS[dim - 1] = R_s
    TS[dim - 1] = T_s
    PS[dim - 1] = P_s
    rhoS[dim - 1] = PS[dim - 1] * mu * mH / (kb * TS[dim - 1])
    kappaS[dim - 1] = kappa0 * rhoS[dim - 1] * TS[dim - 1] ** (-3.5)
    epsilonS[dim - 1] = (epsilon_pp * rhoS[dim - 1] * TS[dim - 1] ** 4+ epsilon_CNO * rhoS[dim - 1] * TS[dim - 1] ** 19.9)
    
    

    dr_Et = dM / (4.0 * pi * RS[dim - 1] ** 3 * rhoS[dim - 1])
    if dr_Et / R_s > 1.0e-02:
        print("Fin de l'exécution car la taille de pas radiale est trop grande")
        exit()

    # integration depuis la surface jusqu'au demi rayon #
    for j in range(dim - 2, int(dim // 2) - 2, -1):
        RS[j] = RS[j + 1] - dM / (4.0 * pi * RS[j + 1] ** 2 * rhoS[j + 1])
        PS[j] = PS[j + 1] + dM * G * MS[j + 1] / (4.0 * pi * RS[j + 1] ** 4)
        LS[j] = LS[j + 1] - dM * epsilonS[j + 1]
        nabla_rad = (3.0* kappaS[j + 1]* LS[j + 1]* PS[j + 1]/ (16.0 * pi * a * c * TS[j + 1] ** 4 * G * MS[j + 1]))
        if nabla_rad < (gamma - 1) / gamma:
            TS[j] = TS[j + 1] + (dM* 3.0* kappaS[j + 1]* LS[j + 1]/ (16.0 * pi * a * c * RS[j + 1] ** 2 * TS[j + 1] ** 3)/ (4.0 * pi * RS[j + 1] ** 2))
        else:
            TS[j] = TS[j + 1] - (dM* (gamma - 1.0)/ gamma* TS[j + 1]/ PS[j + 1]* (PS[j + 1] - PS[j])/ dM)
        
        # vérifiaction que le rayon et la luminosité reste bien positifs #
        if RS[j] <= 0.0 or LS[j] <= 0.0:
            print("Le rayon ou la luminosité est négatif")
            break

        MS[j] = MS[j + 1] - dM
        rhoS[j] = PS[j] * mu * mH / (kb * TS[j])
        epsilonS[j] = (epsilon_pp * rhoS[j] * TS[j] ** 4.0 + epsilon_CNO * rhoS[j] * TS[j] ** 19.9)
        kappaS[j] = kappa0 * rhoS[j] * TS[j] ** (-3.5)

# Valeurs initiales au demi rayon. Les intégrations depuis le centre et depuis la surface doivent converger ici.
    Rmi0 = R[int(dim // 2) - 1]
    RSmi0 = RS[int(dim // 2) - 1]
    Pmi0 = P[int(dim // 2) - 1]
    PSmi0 = PS[int(dim // 2) - 1]
    Tmi0 = T[int(dim // 2) - 1]
    TSmi0 = TS[int(dim // 2) - 1]
    Lmi0 = L[int(dim // 2) - 1]
    LSmi0 = LS[int(dim // 2) - 1]
    k_valeurs[k] = k
    R_diff[k] = Rmi0 - RSmi0
    P_diff[k] = Pmi0 - PSmi0
    T_diff[k] = Tmi0 - TSmi0
    L_diff[k] = Lmi0 - LSmi0
    
    Rmilieu = (R[int(dim // 2) - 1] + RS[int(dim // 2) - 1]) / 2.0
    Pmilieu = (P[int(dim // 2) - 1] + PS[int(dim // 2) - 1]) / 2.0
    Tmilieu = (T[int(dim // 2) - 1] + TS[int(dim // 2) - 1]) / 2.0
    Lmilieu = (L[int(dim // 2) - 1] + LS[int(dim // 2) - 1]) / 2.0
    

    # La condition d'arrêt si les équations de structure stellaire peuvent être résolues avec nos conditions limites #
    if abs(R_diff[k]) / Rmilieu < tol:
        if abs(P_diff[k]) / Pmilieu < tol:
            if abs(T_diff[k]) / Tmilieu < tol:
                if abs(L_diff[k]) / Lmilieu < tol:
                    print("Convergence atteinte")
                    break
                
    # on fait varier légèrement les paramètres afin de recoller les 2 courbes trouvées #
    # variation de x pourcent pour la pression #  
    Mv[0] = M[0]
    Pv[0] = 1.01 * P_c
    Tv[0] = T_c
    rhov[0] = Pv[0] * mu * mH / (kb * Tv[0])
    Rv[0] = (3.0 * Mv[0] / (4.0 * pi * rhov[0])) ** (1.0 / 3.0)
    epsilonv[0] = (epsilon_pp * rhov[0] * Tv[0] ** 4.0 + epsilon_CNO * rhov[0] * Tv[0] ** 19.9)
    Lv[0] = epsilonv[0] * Mv[0]
    kappav[0] = kappa0 * rhov[0] * Tv[0] ** (-3.5)
    # intégration depuis le centre 
    for i in range(1, dim // 2):
        Rv[i] = Rv[i - 1] + dM / (4.0 * pi * Rv[i - 1] ** 2 * rhov[i - 1])
        Pv[i] = Pv[i - 1] - dM * G * Mv[i - 1] / (4.0 * pi * Rv[i - 1] ** 4)
        Lv[i] = Lv[i - 1] + dM * epsilonv[i - 1]
        nabla_rad = (3.0* kappav[i - 1]* Lv[i - 1]* Pv[i - 1]/ (16.0 * pi * a * c * Tv[i - 1] ** 4 * G * Mv[i - 1]))
        if nabla_rad < (gamma - 1) / gamma:
            Tv[i] = Tv[i - 1] - (dM* 3.0* kappav[i - 1]* Lv[i - 1]/ (16.0 * pi * a * c * Rv[i - 1] ** 2 * Tv[i - 1] ** 3)/ (4.0 * pi * Rv[i - 1] ** 2))
        else:
            Tv[i] = Tv[i - 1] + (dM* (gamma - 1.0)/ gamma* Tv[i - 1]/ Pv[i - 1]* (Pv[i] - Pv[i - 1])/ dM)

        if Tv[i] <= 0.0 or Pv[i] <= 0.0:
            print("La température ou la pression est négative")
            break

        Mv[i] = Mv[i - 1] + dM
        rhov[i] = Pv[i] * mu * mH / (kb * Tv[i])
        epsilonv[i] = (epsilon_pp * rhov[i] * Tv[i] ** 4.0 + epsilon_CNO * rhov[i] * Tv[i] ** 19.9)
        kappav[i] = kappa0 * rhov[i] * Tv[i] ** (-3.5)
        
    # Prend le valeur au demi rayon de la solution après variation et on calcule la différence #
    Rmi1 = Rv[int(dim // 2) - 1]
    Pmi1 = Pv[int(dim // 2) - 1]
    Tmi1 = Tv[int(dim // 2) - 1]
    Lmi1 = Lv[int(dim // 2) - 1]
    Rdiff1 = Rmi1 - RSmi0
    Pdiff1 = Pmi1 - PSmi0
    Tdiff1 = Tmi1 - TSmi0
    Ldiff1 = Lmi1 - LSmi0

    # Conditions limites de la première couche avec une variation de température de x pour cent
    Mv[0] = M[0]
    Pv[0] = P_c
    Tv[0] = T_c * 1.01
    rhov[0] = Pv[0] * mu * mH / (kb * Tv[0])
    Rv[0] = (3.0 * Mv[0] / (4.0 * pi * rhov[0])) ** (1.0 / 3.0)
    epsilonv[0] = (
        epsilon_pp * rhov[0] * Tv[0] ** 4.0 + epsilon_CNO * rhov[0] * Tv[0] ** 19.9)
    Lv[0] = epsilonv[0] * Mv[0]
    kappav[0] = kappa0 * rhov[0] * Tv[0] ** (-3.5)


    # Intégrer depuis le centre avec une variation de température 
    for i in range(1, int(dim // 2)):
        Rv[i] = Rv[i - 1] + dM / (4.0 * pi * Rv[i - 1] ** 2 * rhov[i - 1])
        Pv[i] = Pv[i - 1] - dM * G * Mv[i - 1] / (4.0 * pi * Rv[i - 1] ** 4)
        Lv[i] = Lv[i - 1] + dM * epsilonv[i - 1]
        nabla_rad = (3.0* kappav[i - 1]* Lv[i - 1]* Pv[i - 1]/ (16.0 * pi * a * c * Tv[i - 1] ** 4 * G * Mv[i - 1]))

        if nabla_rad < (gamma - 1) / gamma:
            Tv[i] = Tv[i - 1] - (dM* 3.0* kappav[i - 1]* Lv[i - 1]/ (16.0 * pi * a * c * Rv[i - 1] ** 2 * Tv[i - 1] ** 3)/ (4.0 * pi * Rv[i - 1] ** 2))

        else:
            Tv[i] = Tv[i - 1] + (dM* (gamma - 1.0)/ gamma* Tv[i - 1]/ Pv[i - 1]* (Pv[i] - Pv[i - 1])/ dM)

        if Tv[i] <= 0.0 or Pv[i] <= 0.0:
            print("La température ou la pression est négative ")
            break

        Mv[i] = Mv[i - 1] + dM
        rhov[i] = Pv[i] * mu * mH / (kb * Tv[i])
        epsilonv[i] = (epsilon_pp * rhov[i] * Tv[i] ** 4.0 + epsilon_CNO * rhov[i] * Tv[i] ** 19.9)
        kappav[i] = kappa0 * rhov[i] * Tv[i] ** (-3.5)

    Rmi2 = Rv[int(dim // 2) - 1]
    Pmi2 = Pv[int(dim // 2) - 1]
    Tmi2 = Tv[int(dim // 2) - 1]
    Lmi2 = Lv[int(dim // 2) - 1]
    Rdiff2 = Rmi2 - RSmi0
    Pdiff2 = Pmi2 - PSmi0
    Tdiff2 = Tmi2 - TSmi0
    Ldiff2 = Lmi2 - LSmi0
    
    # Calcul avec variation de la luminosité #
    MSv[dim - 1] = M_Et
    LSv[dim - 1] = L_s * 1.01
    RSv[dim - 1] = R_s
    TSv[dim - 1] = T_s
    PSv[dim - 1] = P_s
    rhoSv[dim - 1] = PSv[dim - 1] * mu * mH / (kb * TSv[dim - 1])
    kappaSv[dim - 1] = kappa0 * rhoSv[dim - 1] * TSv[dim - 1] ** (-3.5)
    epsilonSv[dim - 1] = (epsilon_pp * rhoSv[dim - 1] * TSv[dim - 1] ** 4+ epsilon_CNO * rhoSv[dim - 1] * TSv[dim - 1] ** 19.9)

    dr_Et = dM / (4.0 * pi * RS[dim - 1] ** 3 * rhoS[dim - 1])
    if dr_Et / R_s > 1.0e-02:
        print("fin à l'exécution car la taille de l'étape radiale est trop grande")
        exit()
        

    for j in range(dim - 2, int(dim // 2) - 2, -1):
        RSv[j] = RSv[j + 1] - dM / (4.0 * pi * RSv[j + 1] ** 2 * rhoSv[j + 1])
        PSv[j] = PSv[j + 1] + dM * G * MSv[j + 1] / (4.0 * pi * RSv[j + 1] ** 4)
        LSv[j] = LSv[j + 1] - dM * epsilonSv[j + 1]
        nabla_rad = (3.0* kappaSv[j + 1]* LSv[j + 1]* PSv[j + 1]/ (16.0 * pi * a * c * TSv[j + 1] ** 4 * G * MSv[j + 1]))
        if nabla_rad < (gamma - 1) / gamma:
            TSv[j] = TSv[j + 1] + (dM* 3.0* kappaSv[j + 1]* LSv[j + 1]/ (16.0 * pi * a * c * RSv[j + 1] ** 2 * TSv[j + 1] ** 3)/ (4.0 * pi * RSv[j + 1] ** 2))
        else:
            TSv[j] = TSv[j + 1] - (dM* (gamma - 1.0)/ gamma* TSv[j + 1]/ PSv[j + 1]* (PSv[j + 1] - PSv[j])/ dM)

        if RSv[j] <= 0.0 or LSv[j] <= 0.0:
            print("Rayon ou Luminosité négatif ")
            break

        MSv[j] = MSv[j + 1] - dM
        rhoSv[j] = PSv[j] * mu * mH / (kb * TSv[j])
        epsilonSv[j] = (epsilon_pp * rhoSv[j] * TSv[j] ** 4.0+ epsilon_CNO * rhoSv[j] * TSv[j] ** 19.9)
        kappaSv[j] = kappa0 * rhoSv[j] * TSv[j] ** (-3.5)


    RSmi1 = RSv[int(dim // 2) - 1]
    PSmi1 = PSv[int(dim // 2) - 1]
    TSmi1 = TSv[int(dim // 2) - 1]
    LSmi1 = LSv[int(dim // 2) - 1]
    Rdiff3 = Rmi0 - RSmi1
    Pdiff3 = Pmi0 - PSmi1
    Tdiff3 = Tmi0 - TSmi1
    Ldiff3 = Lmi0 - LSmi1
    
    # maintenant nous allons faire varier le rayon #
    MSv[dim - 1] = M_Et
    LSv[dim - 1] = L_s
    RSv[dim - 1] = R_s * 1.01
    PSv[dim - 1] = P_s
    TSv[dim - 1] = T_s

    rhoSv[dim - 1] = PSv[dim - 1] * mu * mH / (kb * TSv[dim - 1])
    kappaSv[dim - 1] = kappa0 * rhoSv[dim - 1] * TSv[dim - 1] ** (-3.5)
    epsilonSv[dim - 1] = (epsilon_pp * rhoSv[dim - 1] * TSv[dim - 1] ** 4+ epsilon_CNO * rhoSv[dim - 1] * TSv[dim - 1] ** 19.9)

    dr_Et= dM / (4.0 * pi * RS[dim - 1] ** 3 * rhoS[dim - 1])
    if dr_Et / R_s > 1.0e-02:
        print(" fin à l'exécution car la taille de l'étape radiale est trop grande")
  
    # intègre le rayon après variation depuis la surface
    for j in range(dim - 2, int(dim // 2) - 2, -1):
        RSv[j] = RSv[j + 1] - dM / (4.0 * pi * RSv[j + 1] ** 2 * rhoSv[j + 1])
        PSv[j] = PSv[j + 1] + dM * G * MSv[j + 1] / (4.0 * pi * RSv[j + 1] ** 4)
        LSv[j] = LSv[j + 1] - dM * epsilonSv[j + 1]
        nabla_rad = (3.0* kappaSv[j + 1]* LSv[j + 1]* PSv[j + 1]/ (16.0 * pi * a * c * TSv[j + 1] ** 4 * G * MSv[j + 1]))

        if nabla_rad < (gamma - 1) / gamma:
            TSv[j] = TSv[j + 1] + (dM* 3.0* kappaSv[j + 1]* LSv[j + 1]/ (16.0 * pi * a * c * RSv[j + 1] ** 2 * TSv[j + 1] ** 3)/ (4.0 * pi * RSv[j + 1] ** 2))

        else:
            TSv[j] = TSv[j + 1] - (dM* (gamma - 1.0)/ gamma* TSv[j + 1]/ PSv[j + 1]* (PSv[j + 1] - PSv[j])/ dM)
        # vérification du dépassement de R et L
        if RSv[j] <= 0.0 or LSv[j] <= 0.0:
            print("Le rayon ou la luminosité est négatif")
            break

        MSv[j] = MSv[j + 1] - dM
        rhoSv[j] = PSv[j] * mu * mH / (kb * TSv[j])
        epsilonSv[j] = (epsilon_pp * rhoSv[j] * TSv[j] ** 4.0+ epsilon_CNO * rhoSv[j] * TSv[j] ** 19.9)
        kappaSv[j] = kappa0 * rhoSv[j] * TSv[j] ** (-3.5)


    # Détermine le point médian à partir de la surface après perturbation
    RSmi2 = RSv[int(dim // 2) - 1]
    PSmi2 = PSv[int(dim // 2) - 1]
    TSmi2 = TSv[int(dim // 2) - 1]
    LSmi2 = LSv[int(dim // 2) - 1]
    Rdiff4 = Rmi0 - RSmi2
    Pdiff4 = Pmi0 - PSmi2
    Tdiff4 = Tmi0 - TSmi2
    Ldiff4 = Lmi0 - LSmi2

    # Ici, nous construisons une matrice dont les éléments sont les dérivées des changements au demi rayon par rapport aux limites
    # Ainsi, nous résoudrons les 4 équations de structure stellaire comme un problème de valeurs propres

    # Dérivées des différences de rayon
    d_deltaR_dP_c = (Rdiff1 - R_diff[k]) / (0.01 * P_c)
    d_deltaR_dT_c = (Rdiff2 - R_diff[k]) / (0.01 * T_c)
    d_deltaR_dLs = (Rdiff3 - R_diff[k]) / (0.01 * L_s)
    d_deltaR_dRs = (Rdiff4 - R_diff[k]) / (0.01 * R_s)

    # Dérivées des différences de pression
    d_deltaP_dP_c = (Pdiff1 - P_diff[k]) / (0.01 * P_c)
    d_deltaP_dT_c = (Pdiff2 - P_diff[k]) / (0.01 * T_c)
    d_deltaP_dLs = (Pdiff3 - P_diff[k]) / (0.01 * L_s)
    d_deltaP_dRs = (Pdiff4 - P_diff[k]) / (0.01 * R_s)

    # Dérivées des différences de Températures
    d_deltaT_dP_c = (Tdiff1 - T_diff[k]) / (0.01 * P_c)
    d_deltaT_dT_c = (Tdiff2 - T_diff[k]) / (0.01 * T_c)
    d_deltaT_dLs = (Tdiff3 - T_diff[k]) / (0.01 * L_s)
    d_deltaT_dRs = (Tdiff4 - T_diff[k]) / (0.01 * R_s)

    # Dérivées des différences de Luminiosité
    d_deltaL_dP_c = (Ldiff1 - L_diff[k]) / (0.01 * P_c)
    d_deltaL_dT_c = (Ldiff2 - L_diff[k]) / (0.01 * T_c)
    d_deltaL_dLs = (Ldiff3 - L_diff[k]) / (0.01 * L_s)
    d_deltaL_dRs = (Ldiff4 - L_diff[k]) / (0.01 * R_s)

    # Les entrées de la matrice A
    A = np.zeros(16).reshape(4, 4)
    A[0, 0] = d_deltaR_dP_c
    A[0, 1] = d_deltaR_dT_c
    A[0, 2] = d_deltaR_dLs
    A[0, 3] = d_deltaR_dRs
    A[1, 0] = d_deltaP_dP_c
    A[1, 1] = d_deltaP_dT_c
    A[1, 2] = d_deltaP_dLs
    A[1, 3] = d_deltaP_dRs
    A[2, 0] = d_deltaT_dP_c
    A[2, 1] = d_deltaT_dT_c
    A[2, 2] = d_deltaT_dLs
    A[2, 3] = d_deltaT_dRs
    A[3, 0] = d_deltaL_dP_c
    A[3, 1] = d_deltaL_dT_c
    A[3, 2] = d_deltaL_dLs
    A[3, 3] = d_deltaL_dRs

    y = np.zeros(4)
    y[0] = 0.1 * R_diff[k]
    y[1] = 0.1 * P_diff[k]
    y[2] = 0.1 * T_diff[k]
    y[3] = 0.1 * L_diff[k]

    x = np.linalg.solve(A, y)

    P_c = P_c - x[0]
    T_c = T_c - x[1]
    L_s = L_s - x[2]
    R_s = R_s - x[3]


# intégration au delà 
for j in range (dim-1,dim +extra-1):
    RS[j+1] = RS[j] + dM / (4.0 * pi * RS[j + 1] ** 2 * rhoS[j + 1])
    PS[j+1] = PS[j] - dM * G * MS[j + 1] / (4.0 * pi * RS[j + 1] ** 4)
    LS[j+1] = LS[j ] + dM * epsilonS[j + 1]
    nabla_rad = (3.0* kappaS[j ]* LS[j ]* PS[j ]/ (16.0 * pi * a * c * TS[j ] ** 4 * G * MS[j ]))
    if nabla_rad < (gamma - 1) / gamma:
        TS[j+1] = TS[j ] - (dM* 3.0* kappaS[j ]* LS[j ]/ (16.0 * pi * a * c * RS[j] ** 2 * TS[j] ** 3)/ (4.0 * pi * RS[j ] ** 2))
    else:
        TS[j+1] = TS[j ] + (dM* (gamma - 1.0)/ gamma* TS[j ]/ PS[j ]* (PS[j ] - PS[j-1])/ dM)
        
        # vérifiaction que le rayon et la luminosité reste bien positifs #
        if RS[j+1] <= 0.0 or LS[j+1] <= 0.0:
            print("Le rayon ou la luminosité est négatif")
            break

        MS[j+1] = MS[j ] + dM
        rhoS[j+1] = PS[j+1] * mu * mH / (kb * TS[j+1])
        epsilonS[j+1] = (epsilon_pp * rhoS[j+1] * TS[j+1] ** 4.0 + epsilon_CNO * rhoS[j+1] * TS[j+1] ** 19.9)
        kappaS[j+1] = kappa0 * rhoS[j+1] * TS[j+1] ** (-3.5)


print("le nombre d'itération est %s" % int(max(k_valeurs)))

print("Masse totale:  %s kg" % round(MS[dim - 1], 3))
print("Rayon: %s metres" % round(RS[dim - 1], 3))
print("Surface Température: %s K" % round(TS[dim - 1], 3))
print("Surface Pression: %s" % round(PS[dim - 1], 3))
print("Surface Luminosité: %s Watts" % round(LS[dim - 1], 2))
print("Masse volumique: %s kg/m^3" % round(rhoS[dim - 1], 2))

plt.figure(1, figsize=(8.5, 11))


plt.figure(1)
plt.plot([0 : int(dim / 2) - 1], P[0 : int(dim // 2) - 1], "b-")
plt.plot(RS[int(dim / 2) : dim +extra- 1], PS[int(dim // 2) : dim + extra- 1], "c-")
plt.title("Pression vs Rayon")
plt.xlabel("Rayon (m)")
plt.ylabel("Pression (KPa)")

plt.figure(2)
plt.plot(R[0 : int(dim / 2) - 1], rho[0 : int(dim // 2) - 1], "b-")
plt.plot(RS[int(dim / 2) : dim +extra- 1], rhoS[int(dim // 2) : dim + extra- 1], "c-")
plt.title("Masse volumique vs Rayon")
plt.xlabel("Rayon (m)")
plt.ylabel("Masse volumique (kg/m^3)")


plt.figure(3)
plt.plot(R[0 : int(dim // 2) - 1], L[0 : int(dim // 2) - 1], "b")
plt.plot(RS[int(dim // 2) : dim +extra- 1], LS[int(dim // 2) : dim +extra-1], "c-")
plt.title("Luminosité vs Rayon")
plt.xlabel("Rayon (m)")
plt.ylabel("Luminosité (Watts)")

plt.figure(4)
plt.plot(R[0 : int(dim // 2) - 1], M[0 : int(dim // 2) - 1], "b-")
plt.plot(RS[int(dim // 2) : dim +extra- 1], MS[int(dim // 2) : dim + extra - 1], "c-")
plt.title("Masse vs Rayon")
plt.xlabel("Rayon (m)")
plt.ylabel("Masse (kg)")

plt.figure(5)
plt.plot(R[0 : int(dim // 2) - 1], T[0 : int(dim // 2) - 1], "b-")
plt.plot(RS[int(dim // 2) : dim +extra- 1], TS[int(dim // 2) : dim + extra - 1], "c-")
plt.title("Température vs Rayon")
plt.xlabel("Rayon (m)")
plt.ylabel("Température (K)")

plt.show()       

    

 

 





    
