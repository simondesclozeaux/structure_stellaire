import numpy as np
from sympy import symbols, Matrix

####Constantes
pi = np.pi
# Boltzmann
k_b = 1.38e-23
# vitesse lumière
c = 3e8
G = 6.67e-11
gamma = 5.0 / 3.0
# masse atome hydrogene kg
m_H = 1.67e-27
X = 0.70  # proportion H
Y = 0.28  # proportion HE
Z = 0.02  # proportion éléments plus lourd
mu = 4 / (3 + 5 * X - Z)
# Taux de génération d'énergie pour PP (proton-proton), CNO (cycle CNO) et l'opacité de Kramer
epsilon_pp = 2.6e-37 * X**2
# contribution du cycle CNO négligé
epsilon_CNO = 0


def Stabilité(m0, T0, L0, P0, r0):
    # Définir les paramètres comme symboles
    m0, T0, L0, P0, r0 = symbols("m0 T0 L0 P0 r0")

    # Définir la matrice en fonction des paramètres
    matrix = Matrix(
        [
            [
                (-2 * k_b * T0) / (4 * pi * (r0**3) * mu * m_H * P0),
                (-k_b * T0) / (4 * pi * (r0**2) * mu * m_H * (P0**2)),
                0,
                (2 * k_b) / (4 * pi * (r0**2) * mu * m_H * P0),
            ],
            [G * m0 / (pi * (r0) ** 5), 0, 0, 0],
            [
                0,
                (epsilon_pp * (T0**3) * mu * m_H) / k_b,
                (3 * epsilon_pp * (T0**2) * mu * m_H * P0) / k_b,
                0,
            ],
            [
                ((gamma - 1) * T0 * 4 * G * m0) / (gamma * P0 * r0**5),
                ((gamma - 1) * T0 * G * m0) / (gamma * (P0**2) * r0**4),
                0,
                -((gamma - 1) * G * m0) / (gamma * P0 * r0**4),
            ],
        ]
    )

    # Calculer les valeurs propres
    eigenvalues = matrix.eigenvals()

    return eigenvalues


# Exemple d'utilisation avec des valeurs spécifiques pour les paramètres
param_m0 = 1
param_T0 = 2
param_L0 = 3
param_P0 = 4
param_r0 = 5

eigenvalues_result = Stabilité(param_m0, param_T0, param_L0, param_P0, param_r0)

print("Les valeurs propres de la matrice sont :", eigenvalues_result)
