"""
core/difference_equation.py

Módulo para la aplicación directa de la Ecuación en Diferencias del filtro IIR.
Implementación 100% fiel a los apuntes del profesor (Página 19, 23, 25).
"""

import numpy as np


class DifferenceEquationFilter:
    """
    Ejecutor de la Ecuación en Diferencias para un filtro digital IIR de orden 4
    (resultado de la transformación pasabandas del prototipo N=2).

    Fórmula matemática explícita de los apuntes:
    ---------------------------------------------
    y(n) = (1/a0) * [ b0*x(n) + b1*x(n-1) + b2*x(n-2) + b3*x(n-3) + b4*x(n-4)
                      - a1*y(n-1) - a2*y(n-2) - a3*y(n-3) - a4*y(n-4) ]

    Si los coeficientes b y a están normalizados (a0 = 1):
    y(n) = b0*x(n) + b1*x(n-1) + b2*x(n-2) + b3*x(n-3) + b4*x(n-4)
           - a1*y(n-1) - a2*y(n-2) - a3*y(n-3) - a4*y(n-4)
    """

    def __init__(self, b: np.ndarray, a: np.ndarray):
        """
        b: Coeficientes del numerador [b0, b1, b2, b3, b4]
        a: Coeficientes del denominador [a0, a1, a2, a3, a4] (normalizados con a0=1)
        """
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.a = np.ascontiguousarray(a, dtype=np.float64)

        # Estados pasados para x(n-1)..x(n-4) y y(n-1)..y(n-4)
        self.x_state = np.zeros(4, dtype=np.float64)
        self.y_state = np.zeros(4, dtype=np.float64)

    def reiniciar_estados(self):
        """ Reinicia los búferes de estado a cero. """
        self.x_state.fill(0.0)
        self.y_state.fill(0.0)

    def filtrar_muestra(self, x_n: float) -> float:
        """
        Filtra una sola muestra x(n) aplicando la ecuación en diferencias directa.
        """
        b = self.b
        a = self.a

        # Ecuacion en diferencias explicita del profesor:
        y_n = (b[0] * x_n +
               b[1] * self.x_state[0] +
               b[2] * self.x_state[1] +
               b[3] * self.x_state[2] +
               b[4] * self.x_state[3] -
               a[1] * self.y_state[0] -
               a[2] * self.y_state[1] -
               a[3] * self.y_state[2] -
               a[4] * self.y_state[3])

        # Actualizar los estados (desplazamiento temporal)
        self.x_state[3] = self.x_state[2]
        self.x_state[2] = self.x_state[1]
        self.x_state[1] = self.x_state[0]
        self.x_state[0] = x_n

        self.y_state[3] = self.y_state[2]
        self.y_state[2] = self.y_state[1]
        self.y_state[1] = self.y_state[0]
        self.y_state[0] = y_n

        return y_n

    def filtrar_bloque(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filtra un bloque continuo de muestras en tiempo real o en procesamiento por lotes,
        manteniendo la continuidad de estado entre bloques.
        """
        x = np.ascontiguousarray(chunk, dtype=np.float64)
        n_samples = len(x)
        y = np.zeros(n_samples, dtype=np.float64)

        b = self.b
        a = self.a
        x_st = self.x_state
        y_st = self.y_state

        for i in range(n_samples):
            xi = x[i]
            yi = (b[0] * xi +
                  b[1] * x_st[0] +
                  b[2] * x_st[1] +
                  b[3] * x_st[2] +
                  b[4] * x_st[3] -
                  a[1] * y_st[0] -
                  a[2] * y_st[1] -
                  a[3] * y_st[2] -
                  a[4] * y_st[3])

            y[i] = yi

            x_st[3] = x_st[2]
            x_st[2] = x_st[1]
            x_st[1] = x_st[0]
            x_st[0] = xi

            y_st[3] = y_st[2]
            y_st[2] = y_st[1]
            y_st[1] = y_st[0]
            y_st[0] = yi

        return y

    def filtrar_senal_completa(self, signal: np.ndarray) -> np.ndarray:
        """
        Filtra una señal completa desde el inicio (reinicia los estados antes de filtrar).
        """
        self.reiniciar_estados()
        return self.filtrar_bloque(signal)
