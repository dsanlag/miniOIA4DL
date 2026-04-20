# --- INICIO BLOQUE GENERADO CON IA ---
# cython: boundscheck=False  
# Desactiva comprobaciones de límites (más rápido)    
# cython: wraparound=False 
# Desactiva índices negativos tipo Python (más rápido)      
# cython: nonecheck=False         
# No comprueba si variables son None (más rápido)
# cython: initializedcheck=False  
# No comprueba inicialización de variables (más rápido)

import numpy as np
cimport numpy as cnp  # Permite usar tipos de NumPy en Cython

def im2col_cython(cnp.ndarray[cnp.float32_t, ndim=4] input_data,
                  int k_h,
                  int k_w,
                  int stride,
                  int padding):
    
    # input_data: [B, C, H, W]
    cdef int B = input_data.shape[0]  # Batch size
    cdef int C = input_data.shape[1]  # Número de canales
    cdef int H = input_data.shape[2]  # Alto
    cdef int W = input_data.shape[3]  # Ancho

    cdef int out_h
    cdef int out_w
    cdef int N
    cdef int K

    cdef int b, c, i, j, kh, kw # Variables para recorrer batch, canales, salida y kernel

    cdef int row, col  # Índices en la matriz final cols

    cdef cnp.ndarray[cnp.float32_t, ndim=4] x # x = entrada (con padding si aplica)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] cols # cols = matriz im2col final

    if padding > 0:
        x = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        ).astype(np.float32) # Aplica padding a la entrada (añade ceros)
    else:
        x = np.ascontiguousarray(input_data, dtype=np.float32) # Asegura que el array está en memoria contigua (mejor rendimiento)

    # Actualiza dimensiones tras el padding
    H = x.shape[2]
    W = x.shape[3]
    

    out_h = (H - k_h) // stride + 1  # Alto de la salida
    out_w = (W - k_w) // stride + 1  # Ancho de la salida

    N = B * out_h * out_w  # Número total de ventanas de convolución (cada una será una fila)

    K = C * k_h * k_w  # Tamaño de cada ventana (cada una será una fila con K elementos)

    cols = np.empty((N, K), dtype=np.float32)
    # Matriz final im2col:
    # filas = ventanas
    # columnas = valores dentro de la ventana

    for b in range(B):  # Recorre cada imagen del batch
        for i in range(out_h):  # Recorre posiciones verticales de la salida
            for j in range(out_w):  # Recorre posiciones horizontales
                row = b * out_h * out_w + i * out_w + j  
                # Calcula el índice de fila en cols (cada ventana → una fila)

                col = 0  # Reinicia índice de columna

                for c in range(C):  # Recorre canales
                    for kh in range(k_h):  # Recorre alto del kernel
                        for kw in range(k_w):  # Recorre ancho del kernel
                            cols[row, col] = x[b, c, i * stride + kh, j * stride + kw]
                            # Copia el valor de la ventana correspondiente

                            col += 1  # Avanza en la fila

    return cols, out_h, out_w

# --- FIN BLOQUE GENERADO CON IA ---