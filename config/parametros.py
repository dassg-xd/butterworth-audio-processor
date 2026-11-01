"""
Configuración de parámetros por defecto para el Laboratorio 1.
Procesamiento de Señales e Imágenes - Ingeniería de Sistemas.
"""

# Frecuencia de muestreo (sugerida en el laboratorio)
DEFAULT_FS = 44100  # Hz

# Frecuencias de corte por defecto (BW = 3000 Hz, fc2 = 5000 Hz <= 5000 Hz)
DEFAULT_FC1 = 2000.0  # Hz
DEFAULT_FC2 = 5000.0  # Hz

# Rango permitido de Ancho de Banda (BW) según enunciado: 2000 Hz a 3000 Hz
MIN_BW = 2000.0  # Hz
MAX_BW = 3000.0  # Hz

# Límite superior de frecuencia de corte no superior a 5000 Hz
MAX_FC2 = 5000.0  # Hz

# Tamaño de bloque y búfer para procesamiento de audio en tiempo real
BUFFER_SIZE = 4096
FFT_SIZE = 4096
