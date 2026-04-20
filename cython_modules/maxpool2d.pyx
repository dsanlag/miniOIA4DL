# --- INICIO BLOQUE GENERADO CON IA ---
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as cnp  # Importa los tipos de NumPy para poder tipar arrays en Cython

def maxpool_forward_cython(cnp.ndarray[cnp.float32_t, ndim=4] input_data,
                           int kernel_size,
                           int stride):
    cdef int B = input_data.shape[0] # Número de imágenes en el batch
    cdef int C = input_data.shape[1] # Número de canales
    cdef int H = input_data.shape[2] # Alto de la entrada
    cdef int W = input_data.shape[3] # Ancho de la entrada

    cdef int KH = kernel_size # Alto de la ventana de pooling
    cdef int KW = kernel_size # Ancho de la ventana de pooling
    cdef int SH = stride # Salto vertical
    cdef int SW = stride # Salto horizontal

    cdef int out_h = (H - KH) // SH + 1  # Alto de la salida
    cdef int out_w = (W - KW) // SW + 1  # Ancho de la salida

    # Reserva la matriz de salida donde irá el máximo de cada ventana. 
    cdef cnp.ndarray[cnp.float32_t, ndim=4] output = np.zeros((B, C, out_h, out_w), dtype=np.float32)

    # Guarda las posiciones [fila, columna] del valor máximo de cada ventana
    # La última dimensión de tamaño 2 almacena: [max_h, max_w]
    cdef cnp.ndarray[cnp.int32_t, ndim=5] max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=np.int32)

    cdef int b, c, i, j, h, w # Variables enteras para recorrer batch, canales, salida y ventana
    cdef int h_start, w_start # Coordenadas de inicio de cada ventana de pooling
    cdef float max_val, val # max_val = máximo actual encontrado en la ventana, val = valor actual que se está comparando
    cdef int max_h, max_w # Coordenadas dentro de la entrada donde se encontró el máximo

    for b in range(B): # Recorre cada imagen del batch
        for c in range(C): # Recorre cada canal
            for i in range(out_h): # Recorre las filas de la salida
                for j in range(out_w): # Recorre las columnas de la salida
                    h_start = i * SH # Fila inicial de la ventana actual
                    w_start = j * SW # Columna inicial de la ventana actual

                    max_val = input_data[b, c, h_start, w_start] # Inicializa el máximo con el primer valor de la ventana
                    max_h = h_start  # Inicializa las coordenadas del máximo en esa misma posición
                    max_w = w_start

                    for h in range(KH): # Recorre las filas dentro de la ventana
                        for w in range(KW): # Recorre las columnas dentro de la ventana
                            val = input_data[b, c, h_start + h, w_start + w] # Lee el valor actual de la ventana

                            if val > max_val: # Si encuentra uno mayor, actualiza el máximo
                                max_val = val
                                max_h = h_start + h
                                max_w = w_start + w

                    output[b, c, i, j] = max_val  # Guarda el valor máximo de la ventana en la salida

                    # Guarda también dónde estaba ese máximo en la entrada
                    max_indices[b, c, i, j, 0] = max_h
                    max_indices[b, c, i, j, 1] = max_w

    #Devuelve output: salida del MaxPool y max_indices: posiciones de los máximos, necesarias para backward
    return output, max_indices 

# --- FIN BLOQUE GENERADO CON IA ---