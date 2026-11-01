"""
core/audio_processor.py

Módulo de adquisición y procesamiento de audio.
- Captura de voz en tiempo real desde el micrófono (sounddevice).
- Filtrado muestra a muestra con la ecuación en diferencias.
- Cálculo de espectros por FFT para entrada y salida.
"""

import threading

import numpy as np
import sounddevice as sd

from core.butterworth_bp import ButterworthBandpassCore
from core.difference_equation import DifferenceEquationFilter
from config.parametros import (
    DEFAULT_FS,
    DEFAULT_FC1,
    DEFAULT_FC2,
    BUFFER_SIZE,
    FFT_SIZE,
)


class AudioProcessor:
    """
    Orquesta el flujo del laboratorio:
    micrófono -> x(n) -> ecuación en diferencias -> y(n) -> FFT.
    """

    def __init__(self, fs: float = DEFAULT_FS, fc1: float = DEFAULT_FC1, fc2: float = DEFAULT_FC2):
        self.fs = float(fs)
        self.fc1 = float(fc1)
        self.fc2 = float(fc2)

        self.filter_core = ButterworthBandpassCore(self.fc1, self.fc2, self.fs)
        self.diff_eq = DifferenceEquationFilter(*self.filter_core.obtener_coeficientes_normalizados())

        self.lock = threading.RLock()
        self.stream = None
        self.is_streaming = False

        # Búfer circular para la gráfica en vivo
        self.buffer_in = np.zeros(FFT_SIZE, dtype=np.float64)
        self.buffer_out = np.zeros(FFT_SIZE, dtype=np.float64)
        self.fft_window = np.hanning(FFT_SIZE)

        self.ultimo_error = ""

    # ------------------------------------------------------------------
    # Diseño del filtro
    # ------------------------------------------------------------------
    def actualizar_filtro(self, fc1: float, fc2: float):
        """Recalcula los coeficientes y reinicia los estados del filtro."""
        with self.lock:
            core = ButterworthBandpassCore(fc1, fc2, self.fs)
            b, a = core.obtener_coeficientes_normalizados()
            self.filter_core = core
            self.diff_eq = DifferenceEquationFilter(b, a)
            self.fc1, self.fc2 = float(fc1), float(fc2)

    @property
    def coeficientes(self):
        return self.diff_eq.b, self.diff_eq.a

    @property
    def bw(self) -> float:
        return self.fc2 - self.fc1

    # ------------------------------------------------------------------
    # Micrófono
    # ------------------------------------------------------------------
    def iniciar_microfono(self):
        if self.is_streaming:
            return
        with self.lock:
            self.diff_eq.reiniciar_estados()
            self.buffer_in.fill(0.0)
            self.buffer_out.fill(0.0)

        def callback(indata, frames, time_info, status):
            if status:
                self.ultimo_error = str(status)
            x = indata[:, 0].astype(np.float64)

            with self.lock:
                y = self.diff_eq.filtrar_bloque(x)

                n = min(len(x), FFT_SIZE)
                self.buffer_in[:-n] = self.buffer_in[n:]
                self.buffer_in[-n:] = x[-n:]

                self.buffer_out[:-n] = self.buffer_out[n:]
                self.buffer_out[-n:] = y[-n:]

        try:
            self.stream = sd.InputStream(
                samplerate=int(self.fs),
                channels=1,
                dtype="float32",
                blocksize=BUFFER_SIZE,
                callback=callback,
            )
            self.stream.start()
            self.is_streaming = True
            self.ultimo_error = ""
        except Exception as exc:
            self.stream = None
            self.is_streaming = False
            raise RuntimeError(f"No fue posible abrir el micrófono: {exc}") from exc

    def detener_microfono(self):
        with self.lock:
            self.is_streaming = False
            stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Análisis espectral
    # ------------------------------------------------------------------
    def espectro_tiempo_real(self):
        """FFT en vivo del búfer de entrada y del de salida."""
        with self.lock:
            x = self.buffer_in * self.fft_window
            y = self.buffer_out * self.fft_window
        return self._espectro(x, y)

    def _espectro(self, x: np.ndarray, y: np.ndarray):
        n = min(len(x), len(y))
        if n < 16:
            freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / self.fs)
            vacio = np.full(freqs.shape, -120.0)
            return freqs, vacio, vacio, vacio, vacio

        x = x[:n]
        y = y[:n]
        ventana = np.hanning(n)
        escala = 2.0 / np.sum(ventana)

        mx = np.abs(np.fft.rfft(x * ventana)) * escala
        my = np.abs(np.fft.rfft(y * ventana)) * escala
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)

        db = lambda m: 20.0 * np.log10(np.maximum(m, 1e-9))
        return freqs, db(mx), db(my), mx, my
