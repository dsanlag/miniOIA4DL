# --- INICIO BLOQUE GENERADO CON IA ---
# Optimización de la capa Dense.
# Se sustituye la multiplicación matricial basada en bucles por una operación vectorizada: A @ B + bias.
# A es la entrada de la capa, B los pesos de la red y bias el vector que se suma al resultado. 
# C se deja para no romper el codigo de dense.py
# Esto permite aprovechar implementaciones optimizadas de NumPy y mejorar el rendimiento.
import numpy as np  # Importa NumPy para usar operaciones vectorizadas eficientes

def matmul_biasses(A, B, C, bias):
    return (A @ B + bias).astype(np.float32)  # Multiplicación matricial + bias optimizada
# --- FIN BLOQUE GENERADO CON IA ---
