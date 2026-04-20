from modules.layer import Layer
from cython_modules.maxpool2d import maxpool_forward_cython # Se importa la versión optimizada en Cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    # --- INICIO BLOQUE GENERADO CON IA ---
    # Se sustituye la implementación manual de MaxPool por una versión optimizada en Cython.
    # El objetivo es reducir el uso de bucles en Python y mejorar el rendimiento del forward.
    # Además, se fuerza el tipo float32 para garantizar compatibilidad con el módulo Cython.
    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input.astype(np.float32)  # Asegura el tipo correcto para el módulo Cython

        output, max_indices = maxpool_forward_cython( # Llamada a la implementación en Cython
            self.input, 
            self.kernel_size, # Tamaño de la ventana de pooling
            self.stride # Paso con el que se desplaza la ventana
        )

        # Guarda las posiciones de los valores máximos encontrados en cada ventana.
        # Esto es necesario para el backward, ya que el gradiente solo se propaga por esas posiciones.
        self.max_indices = max_indices 
        return output # Devuelve la salida del max pooling ya calculada en Cython
    # --- FIN BLOQUE GENERADO CON IA ---

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input