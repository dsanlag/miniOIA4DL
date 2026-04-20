# --- INICIO BLOQUE GENERADO CON IA ---
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as cnp  # Permite usar tipos de NumPy en Cython (mejora rendimiento)

def dense_forward_cython(cnp.ndarray[cnp.float32_t, ndim=2] A,
                        cnp.ndarray[cnp.float32_t, ndim=2] B,
                        cnp.ndarray[cnp.float32_t, ndim=1] bias):

    # A: [m, p] -> entrada (batch_size x in_features)
    # B: [p, n] -> pesos (in_features x out_features)
    # bias: [n] -> bias de cada neurona de salida

    cdef int m = A.shape[0]  # Número de muestras (batch_size)
    cdef int p = A.shape[1]  # Número de features de entrada
    cdef int n = B.shape[1]  # Número de neuronas de salida

    cdef cnp.ndarray[cnp.float32_t, ndim=2] C = np.zeros((m, n), dtype=np.float32) # Matriz de salida: [batch_size, out_features]

    cdef int i, j, k # Variables para recorrer matrices

    cdef float val # Acumulador para la suma del producto escalar

    for i in range(m):  # Recorre cada fila de A (cada muestra del batch)
        for j in range(n):  # Recorre cada columna de B (cada neurona de salida)
            val = 0.0  # Inicializa el acumulador para el producto escalar
            for k in range(p):  # Recorre las features
                val += A[i, k] * B[k, j]   # Producto escalar entre fila i de A y columna j de B

            C[i, j] = val + bias[j]   # Guarda el resultado y suma el bias correspondiente

    return C  # Devuelve la salida final

# --- FIN BLOQUE GENERADO CON IA ---