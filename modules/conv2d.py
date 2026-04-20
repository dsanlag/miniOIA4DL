from modules.layer import Layer
from modules.utils import *
from cython_modules.im2col import im2col_cython

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
         # Selección del algoritmo de convolución
         # El cambio realizado permite seleccionar el algoritmo de convolución mediante 
         # conv_algo==1, añadiendo im2col para mejorar el rendimiento frente al método directo.
         # conv_algo==2, añadiendo cython al metodo im2col para mejorar el rendimiento.
        if conv_algo == 0:
            self.mode = 'direct'
        elif conv_algo == 1:
            self.mode = 'im2col'
        elif conv_algo == 2:
            self.mode = 'im2col_cython'
        else:
            print(f"Algoritmo {conv_algo} no soportado aún, usando direct")
            self.mode = 'direct'

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        elif self.mode == 'im2col_cython':
            return self._forward_im2col_cython(input)
        else:
            raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython ")

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c,
                                           i * self.stride:i * self.stride + k_h,
                                           j * self.stride:j * self.stride + k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]

        return output

    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input
    
    # --- INICIO BLOQUE GENERADO CON IA ---
    # Construcción de im2col.
    # Función que transforma la entrada en una matriz para poder hacer la convolución como una multiplicación matricial
    def _im2col(self, input_data, k_h, k_w, stride, padding):

        if padding > 0: # Comprueba si hay que añadir padding alrededor de la imagen
            input_data = np.pad( # Añade ceros alrededor de la entrada
                input_data,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), # No añade en batch ni canales; sí en alto y ancho
                mode='constant' # El padding se rellena con ceros
            )

        B, C, H, W = input_data.shape # Obtiene dimensiones de la entrada: batch, canales, alto y ancho

        out_h = (H - k_h) // stride + 1 # Calcula el alto de la salida de la convolución
        out_w = (W - k_w) // stride + 1 # Calcula el ancho de la salida de la convolución

        cols = np.zeros((B, C, k_h, k_w, out_h, out_w), dtype=np.float32) # Reserva memoria para guardar todas las ventanas extraídas de la entrada

        for y in range(k_h): # Recorre las filas del kernel
            y_max = y + stride * out_h # Calcula hasta dónde hay que leer en vertical para cubrir todas las posiciones
            for x in range(k_w): # Recorre las columnas del kernel
                x_max = x + stride * out_w # Calcula hasta dónde hay que leer en horizontal para cubrir todas las posiciones
                cols[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride] # Extrae todos los valores de la entrada que corresponden a esa posición (y,x) del kernel

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1) # Reordena la matriz para que cada ventana de convolución quede convertida en una fila

        return cols, out_h, out_w # Devuelve la matriz im2col y las dimensiones espaciales de la salida
    # --- FIN BLOQUE GENERADO CON IA ---

    # --- INICIO BLOQUE GENERADO CON IA ---
    # Calcula la convolución usando im2col con NumPy.
    def _forward_im2col(self, input): # Obtiene el número de imágenes del batch
        batch_size = input.shape[0] # Convierte la entrada en formato im2col (cada ventana es una fila)

        cols, out_h, out_w = self._im2col(
            input,
            self.kernel_size, # Alto del kernel
            self.kernel_size, # Ancho del kernel
            self.stride, # Stride de la convolución
            self.padding  # Padding aplicado
        )

        # Convierte cada filtro en una fila (flatten)
        # Antes: [out_channels, in_channels, k_h, k_w]
        # Después: [out_channels, C * k_h * k_w]
        kernels_col = self.kernels.reshape(self.out_channels, -1)

        # Multiplicación matricial:
        # cols:        [B*out_h*out_w, C*k_h*k_w]
        # kernels_col: [out_channels, C*k_h*k_w]
        # Resultado:   [B*out_h*out_w, out_channels]
        # Cada fila es una posición de la imagen y cada columna un filtro
        output = cols @ kernels_col.T

        output += self.biases # Suma el bias a cada filtro

        output = output.reshape(batch_size, out_h, out_w, self.out_channels) # Reorganiza la salida plana a formato espacial: [B, out_h, out_w, out_channels]
        output = output.transpose(0, 3, 1, 2).astype(np.float32)   # Cambia el orden a formato estándar de CNN:[batch, canales, alto, ancho]

        return output # Devuelve el resultado final de la convolución
    # --- FIN BLOQUE GENERADO CON IA ---

    # --- INICIO BLOQUE GENERADO CON IA ---
    # Construcción de im2col acelerada con Cython.
    def _im2col_cython(self, input_data, k_h, k_w, stride, padding): # Función que prepara la entrada para la convolución usando una rutina Cython

        input_data = input_data.astype(np.float32, copy=False)  # Asegura que la entrada tenga tipo float32 sin copiar si no hace falta

        if padding > 0: # Comprueba si hay que añadir padding
            input_data = np.pad( # Añade ceros alrededor de la imagen
                input_data,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), # No añade padding en batch ni canales; sí en alto y ancho
                mode='constant' # El padding se rellena con ceros
            )

        cols, out_h, out_w = im2col_cython( # Llama a la función Cython que construye la matriz im2col de forma más rápida
            input_data, 
            k_h, # Alto del kernel
            k_w, # Ancho del kernel
            stride, # Stride de la convolución
            0  # Se pasa 0 porque el padding ya se ha aplicado antes en Python
        )

        return cols, out_h, out_w # Devuelve la matriz im2col y las dimensiones espaciales de salida
    # --- FIN BLOQUE GENERADO CON IA ---

    # --- INICIO BLOQUE GENERADO CON IA ---
    # Calcula la convolución usando im2col acelerado con Cython.
    def _forward_im2col_cython(self, input):
        batch_size = input.shape[0]  # Obtiene el número de imágenes del batch

        cols, out_h, out_w = self._im2col_cython(
            input,
            self.kernel_size, # Alto del kernel
            self.kernel_size, # Ancho del kernel
            self.stride, # Stride de la convolución
            self.padding # Padding de la convolución
        )

        kernels_col = self.kernels.reshape(self.out_channels, -1) # Aplana cada filtro para convertirlo en una fila

        output = cols @ kernels_col.T # Realiza la convolución como una multiplicación matricial
        output += self.biases # Suma el bias de cada filtro al resultado

        output = output.reshape(batch_size, out_h, out_w, self.out_channels)  # Reorganiza la salida plana a formato espacial
        output = output.transpose(0, 3, 1, 2).astype(np.float32) # Cambia el orden a [batch, canales, alto, ancho]

        return output # Devuelve la salida final de la convolución
    # --- FIN BLOQUE GENERADO CON IA ---