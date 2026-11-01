"""
core/butterworth_bp.py

Núcleo matemático: diseño del Filtro Butterworth Pasabandas de orden 2
por Transformación Bilineal, siguiendo LITERALMENTE el Ejercicio 4 de los
apuntes de clase (páginas 26 y 27, "continuarlo").

Ruta de diseño de los apuntes
-----------------------------
1) Prototipo Butterworth normalizado de orden N = 2 (pág. 22):
       H(s) = 1 / (s² + √2·s + 1)

2) Transformación analógica pasabajas -> pasabandas (pág. 21):
       s ← (s² + w0²) / (BW·s),      w0 = √(wc1·wc2),   BW = wc2 − wc1
   que conduce a (pág. 26):
       H_BP(s) = (BW)²·s² /
                 [ s⁴ + √2·BW·s³ + (2w0² + (BW)²)·s² + √2·BW·w0²·s + w0⁴ ]

3) Prewarping y Bilineal (pág. 24):
       wd1 = (2/T)·tan(π·fc1/fs)
       wd2 = (2/T)·tan(π·fc2/fs)
       s   = (2/T)·(1 − z⁻¹)/(1 + z⁻¹),      T = 1/fs

4) Sustitución en H_BP(s) (pág. 26-27, el "continuarlo"):
       H_BP(z) = (BW)²·(2/T)²·((1−z⁻¹)/(1+z⁻¹))²
                 ---------------------------------------------------------------
                 (2/T)⁴((1−z⁻¹)/(1+z⁻¹))⁴
                 + √2·BW·(2/T)³((1−z⁻¹)/(1+z⁻¹))³
                 + (2w0²+(BW)²)(2/T)²((1−z⁻¹)/(1+z⁻¹))²
                 + √2·BW·w0²·(2/T)((1−z⁻¹)/(1+z⁻¹))
                 + w0⁴

   Multiplicando numerador y denominador por (1+z⁻¹)⁴ y dividiendo todo
   por (2/T)⁴ aparecen las variables de clase:
       C1 = tan(π·fc1/fs),   C2 = tan(π·fc2/fs)
       β  = C2 − C1          (= (T/2)·BW)
       γ  = C1·C2            (= (T/2)²·w0²)
   de donde:
       b = β²·[1, 0, −2, 0, 1]
       a0 = 1 + √2β + 2γ + β² + √2βγ + γ²
       a1 = −4 − 2√2β + 2√2βγ + 4γ²
       a2 = 6 − 4γ − 2β² + 6γ²
       a3 = −4 + 2√2β − 2√2βγ + 4γ²
       a4 = 1 − √2β + 2γ + β² − √2βγ + γ²

El módulo calcula AMBAS formas: la expansión literal con (2/T) y la forma
compacta en β, γ. Se usan los coeficientes compactos (mejor condicionados
numéricamente) y las pruebas verifican que ambas coinciden.
"""

import numpy as np

from config.parametros import MIN_BW, MAX_BW, MAX_FC2


class ButterworthBandpassCore:
    """Diseño del filtro Butterworth pasabandas de orden 2 (digital de orden 4)."""

    def __init__(self, fc1: float, fc2: float, fs: float):
        self.fc1 = float(fc1)
        self.fc2 = float(fc2)
        self.fs = float(fs)
        self.T = 1.0 / self.fs
        self.K = 2.0 / self.T  # = 2·fs

        self.validar_parametros()

        # --- Paso 3: prewarping (pág. 24 y 26) -------------------------
        self.C1 = np.tan(np.pi * self.fc1 / self.fs)          # tan(π·fc1/fs)
        self.C2 = np.tan(np.pi * self.fc2 / self.fs)          # tan(π·fc2/fs)
        self.wd1 = self.K * self.C1                           # (2/T)·tan(π·fc1/fs)
        self.wd2 = self.K * self.C2                           # (2/T)·tan(π·fc2/fs)
        self.w0 = np.sqrt(self.wd1 * self.wd2)                # √(wd1·wd2)
        self.BW = self.wd2 - self.wd1                         # BW analógico [rad/s]

        # --- Variables de la forma compacta ----------------------------
        self.beta = self.C2 - self.C1                         # β = (T/2)·BW
        self.gamma = self.C1 * self.C2                        # γ = (T/2)²·w0²

        # --- Coeficientes -------------------------------------------------
        self.b_literal, self.a_literal = self._expansion_literal_apuntes()
        b, a = self._forma_compacta_beta_gamma()

        self.a0 = a[0]
        self.b = b / self.a0
        self.a = a / self.a0

    # ==================================================================
    # Validación del enunciado
    # ==================================================================
    def validar_parametros(self):
        """BW entre 2000 y 3000 Hz y fc2 ≤ 5000 Hz (enunciado del laboratorio)."""
        if self.fc1 <= 0 or self.fc2 <= 0:
            raise ValueError("Las frecuencias de corte deben ser mayores que 0 Hz.")
        if self.fc1 >= self.fc2:
            raise ValueError("fc1 debe ser estrictamente menor que fc2.")
        if self.fc2 > MAX_FC2:
            raise ValueError(
                f"fc2 = {self.fc2:g} Hz supera el límite de {MAX_FC2:g} Hz del enunciado.")
        if self.fc2 >= self.fs / 2:
            raise ValueError(
                f"fc2 = {self.fc2:g} Hz debe ser menor que fs/2 = {self.fs / 2:g} Hz.")

        bw_hz = self.fc2 - self.fc1
        if not (MIN_BW <= bw_hz <= MAX_BW):
            raise ValueError(
                f"BW = {bw_hz:g} Hz está fuera del rango permitido "
                f"[{MIN_BW:g}, {MAX_BW:g}] Hz.")

    # ==================================================================
    # Forma compacta en β y γ (resultado del "continuarlo")
    # ==================================================================
    def _forma_compacta_beta_gamma(self):
        beta, gamma = self.beta, self.gamma
        r2 = np.sqrt(2.0)
        b2 = beta ** 2
        g2 = gamma ** 2

        b = np.array([b2, 0.0, -2.0 * b2, 0.0, b2], dtype=np.float64)

        a = np.array([
            1.0 + r2 * beta + 2.0 * gamma + b2 + r2 * beta * gamma + g2,
            -4.0 - 2.0 * r2 * beta + 2.0 * r2 * beta * gamma + 4.0 * g2,
            6.0 - 4.0 * gamma - 2.0 * b2 + 6.0 * g2,
            -4.0 + 2.0 * r2 * beta - 2.0 * r2 * beta * gamma + 4.0 * g2,
            1.0 - r2 * beta + 2.0 * gamma + b2 - r2 * beta * gamma + g2,
        ], dtype=np.float64)

        return b, a

    # ==================================================================
    # Expansión literal de la página 26-27 (factores (2/T))
    # ==================================================================
    def _expansion_literal_apuntes(self):
        """
        Construye H_BP(z) exactamente como queda escrita en los apuntes,
        multiplicando por (1+z⁻¹)⁴ y expandiendo los polinomios en z⁻¹.
        """
        K, BW, w0 = self.K, self.BW, self.w0
        r2 = np.sqrt(2.0)

        u = np.array([1.0, -1.0])                 # (1 − z⁻¹)
        v = np.array([1.0, 1.0])                  # (1 + z⁻¹)
        u2 = np.polymul(u, u)                     # (1 − z⁻¹)²
        u3 = np.polymul(u2, u)                    # (1 − z⁻¹)³
        u4 = np.polymul(u3, u)                    # (1 − z⁻¹)⁴
        v2 = np.polymul(v, v)
        v3 = np.polymul(v2, v)
        v4 = np.polymul(v3, v)

        num = (BW ** 2) * (K ** 2) * np.polymul(u2, v2)

        den = (K ** 4) * u4
        den = den + r2 * BW * (K ** 3) * np.polymul(u3, v)
        den = den + (2.0 * w0 ** 2 + BW ** 2) * (K ** 2) * np.polymul(u2, v2)
        den = den + r2 * BW * (w0 ** 2) * K * np.polymul(u, v3)
        den = den + (w0 ** 4) * v4

        return num, den

    # ==================================================================
    # Consultas
    # ==================================================================
    def obtener_coeficientes_normalizados(self):
        """Devuelve (b, a) normalizados por a0, listos para la ecuación en diferencias."""
        return self.b, self.a

    def resumen_apuntes(self) -> dict:
        """Valores intermedios tal como se calculan en clase (para mostrar en la vista)."""
        return {
            "T": self.T,
            "C1": self.C1,
            "C2": self.C2,
            "wd1": self.wd1,
            "wd2": self.wd2,
            "w0": self.w0,
            "BW": self.BW,
            "BW_hz": self.fc2 - self.fc1,
            "beta": self.beta,
            "gamma": self.gamma,
            "a0": self.a0,
        }

    def respuesta_en_frecuencia(self, n_points: int = 4096):
        """
        H(e^{jω}) evaluada en el círculo unitario.
        Devuelve (frecuencias_Hz, |H|, |H|_dB, fase_rad).
        """
        w = np.linspace(0.0, np.pi, n_points)
        z_inv = np.exp(-1j * w)

        num = np.zeros_like(z_inv)
        den = np.zeros_like(z_inv)
        for k in range(5):
            num += self.b[k] * z_inv ** k
            den += self.a[k] * z_inv ** k

        H = num / den
        freqs = (w / (2.0 * np.pi)) * self.fs
        mag = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        return freqs, mag, mag_db, np.angle(H)
