"""
view/main_window.py

Capa de VISTA (interfaz gráfica) del Laboratorio 1.
No contiene matemáticas del filtro: captura parámetros, invoca al núcleo (core)
y dibuja resultados con Matplotlib embebido en Tkinter.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.audio_processor import AudioProcessor
from config.parametros import DEFAULT_FS, DEFAULT_FC1, DEFAULT_FC2, MIN_BW, MAX_BW, MAX_FC2

# Paleta clara
BG = "#f3f5f8"
PANEL = "#ffffff"
FG = "#1f2933"
MUTED = "#6b7785"
BORDER = "#dbe2ea"
ACCENT = "#2563eb"
ACCENT_2 = "#ea580c"
OK = "#15803d"
ERR = "#b91c1c"


class MainWindow(tk.Tk):
    """Ventana principal del laboratorio."""

    def __init__(self):
        super().__init__()
        self.title("Laboratorio 1 — Filtro Butterworth Pasabandas")
        self.geometry("1280x760")
        self.minsize(1040, 660)
        self.configure(bg=BG)

        self.proc = AudioProcessor(DEFAULT_FS, DEFAULT_FC1, DEFAULT_FC2)
        self._after_id = None
        self.marcas_corte = []

        self._construir_estilos()
        self._construir_layout()
        self._actualizar_diseno(forzar=True)
        self._programar_actualizacion()

    # ==================================================================
    # Estilos
    # ==================================================================
    def _construir_estilos(self):
        estilo = ttk.Style(self)
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass

        estilo.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        estilo.configure("TFrame", background=BG)
        estilo.configure("Panel.TFrame", background=PANEL)
        estilo.configure("TLabel", background=BG, foreground=FG)
        estilo.configure("Panel.TLabel", background=PANEL, foreground=FG)
        estilo.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                         font=("Segoe UI", 9))
        estilo.configure("Title.TLabel", background=BG, foreground=FG,
                         font=("Segoe UI Semibold", 16))
        estilo.configure("Section.TLabel", background=PANEL, foreground=FG,
                         font=("Segoe UI Semibold", 10))

        estilo.configure("TButton", background="#e8edf3", foreground=FG,
                         borderwidth=0, focusthickness=0, padding=(10, 7),
                         font=("Segoe UI", 10))
        estilo.map("TButton",
                   background=[("active", "#dbe4ee"), ("disabled", "#eef1f5")],
                   foreground=[("disabled", "#a3adb9")])

        estilo.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                         font=("Segoe UI Semibold", 10))
        estilo.map("Accent.TButton",
                   background=[("active", "#1d4ed8"), ("disabled", "#a8bff5")],
                   foreground=[("disabled", "#f2f5ff")])

        estilo.configure("Danger.TButton", background="#fde8e8", foreground=ERR)
        estilo.map("Danger.TButton", background=[("active", "#fbd5d5")])

        estilo.configure("TEntry", fieldbackground="#ffffff", foreground=FG,
                         insertcolor=FG, borderwidth=1, padding=6)
        estilo.configure("TSeparator", background=BORDER)

    # ==================================================================
    # Layout
    # ==================================================================
    def _construir_layout(self):
        cabecera = ttk.Frame(self, padding=(20, 16, 20, 4))
        cabecera.pack(fill="x")
        ttk.Label(cabecera, text="Filtro Butterworth Pasabandas · Orden 2",
                  style="Title.TLabel").pack(anchor="w")

        cuerpo = ttk.Frame(self, padding=(16, 10, 16, 16))
        cuerpo.pack(fill="both", expand=True)

        self.panel = ttk.Frame(cuerpo, style="Panel.TFrame", width=300, padding=16)
        self.panel.pack(side="left", fill="y", padx=(0, 14))
        self.panel.pack_propagate(False)

        self._construir_panel_parametros()

        grafico = ttk.Frame(cuerpo)
        grafico.pack(side="left", fill="both", expand=True)
        self._construir_graficos(grafico)

    # ------------------------------------------------------------------
    def _construir_panel_parametros(self):
        ttk.Label(self.panel, text="Diseño", style="Section.TLabel").pack(anchor="w")

        marco = ttk.Frame(self.panel, style="Panel.TFrame")
        marco.pack(fill="x", pady=(10, 0))
        marco.columnconfigure(1, weight=1)

        self.var_fs = tk.StringVar(value=str(int(DEFAULT_FS)))
        self.var_fc1 = tk.StringVar(value=f"{DEFAULT_FC1:g}")
        self.var_fc2 = tk.StringVar(value=f"{DEFAULT_FC2:g}")

        for fila, (etiqueta, var) in enumerate([
            ("fs (Hz)", self.var_fs),
            ("fc1 (Hz)", self.var_fc1),
            ("fc2 (Hz)", self.var_fc2),
        ]):
            ttk.Label(marco, text=etiqueta, style="Panel.TLabel").grid(
                row=fila, column=0, sticky="w", pady=3)
            ttk.Entry(marco, textvariable=var, width=12).grid(
                row=fila, column=1, sticky="ew", padx=(10, 0), pady=3)

        self.lbl_bw = ttk.Label(self.panel, text="", style="Muted.TLabel")
        self.lbl_bw.pack(anchor="w", pady=(10, 0))

        ttk.Button(self.panel, text="Aplicar", style="Accent.TButton",
                   command=self._actualizar_diseno).pack(fill="x", pady=(10, 0))

        self.lbl_estado = ttk.Label(self.panel, text="", style="Muted.TLabel",
                                    wraplength=260, justify="left")
        self.lbl_estado.pack(anchor="w", pady=(8, 0))

        ttk.Separator(self.panel).pack(fill="x", pady=14)

    # ------------------------------------------------------------------
    def _construir_graficos(self, contenedor):
        self.figura = Figure(figsize=(9.2, 6.8), dpi=100, facecolor=BG)
        self.figura.subplots_adjust(left=0.075, right=0.98, top=0.93, bottom=0.09,
                                    hspace=0.45, wspace=0.22)

        grid = self.figura.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

        self.ax_espectro = self.figura.add_subplot(grid[0, :])
        self.ax_respuesta = self.figura.add_subplot(grid[1, 0])
        self.ax_tiempo = self.figura.add_subplot(grid[1, 1])

        for ax in (self.ax_espectro, self.ax_respuesta, self.ax_tiempo):
            self._estilizar_ejes(ax)

        self.ax_espectro.set_title("Espectro de magnitud", color=FG, fontsize=11,
                                   loc="left", pad=8)
        self.ax_espectro.set_xlabel("Frecuencia [Hz]")
        self.ax_espectro.set_ylabel("Magnitud [dB]")
        self.ax_espectro.set_xlim(0, min(12000, DEFAULT_FS / 2))
        self.ax_espectro.set_ylim(-90, 10)

        self.linea_entrada, = self.ax_espectro.plot([], [], color="#94a3b8", lw=0.9,
                                                    label="Original")
        self.linea_filtrada, = self.ax_espectro.plot([], [], color=ACCENT, lw=1.2,
                                                     label="Filtrada")
        self.banda_paso = self.ax_espectro.axvspan(DEFAULT_FC1, DEFAULT_FC2,
                                                   color=ACCENT, alpha=0.08, lw=0)
        self.ax_espectro.legend(loc="upper right", frameon=False,
                                labelcolor=FG, fontsize=9)

        self.ax_respuesta.set_title(r"Respuesta en frecuencia $|H(e^{j\omega})|$",
                                    color=FG, fontsize=11, loc="left", pad=8)
        self.ax_respuesta.set_xlabel("Frecuencia [Hz]")
        self.ax_respuesta.set_ylabel("|H| [dB]")
        self.ax_respuesta.set_ylim(-80, 6)

        self.linea_h, = self.ax_respuesta.plot([], [], color=ACCENT_2, lw=1.4)
        self.ax_respuesta.axhline(-3.0, color=ERR, ls="--", lw=0.9, alpha=0.6)

        self.ax_tiempo.set_title("Forma de onda", color=FG, fontsize=11,
                                 loc="left", pad=8)
        self.ax_tiempo.set_xlabel("Tiempo [ms]")
        self.ax_tiempo.set_ylabel("Amplitud")
        self.ax_tiempo.set_xlim(0, 50)
        self.ax_tiempo.set_ylim(-1.1, 1.1)
        self.linea_t_in, = self.ax_tiempo.plot([], [], color="#94a3b8", lw=0.9,
                                               label="Original")
        self.linea_t_out, = self.ax_tiempo.plot([], [], color=ACCENT, lw=1.0,
                                                label="Filtrada")
        self.ax_tiempo.legend(loc="upper right", frameon=True, facecolor=PANEL,
                              edgecolor=BORDER, framealpha=0.9,
                              labelcolor=FG, fontsize=8)

        self.canvas = FigureCanvasTkAgg(self.figura, master=contenedor)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        barra = NavigationToolbar2Tk(self.canvas, contenedor, pack_toolbar=False)
        barra.configure(background=BG)
        barra.update()
        barra.pack(fill="x")

        self._dibujar_respuesta()

    def _estilizar_ejes(self, ax):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors=MUTED, labelsize=8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(BORDER)
        ax.grid(True, color="#e8edf3", lw=0.7)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    # ==================================================================
    # Acciones
    # ==================================================================
    def _leer_parametros(self):
        fs = float(self.var_fs.get().replace(",", "."))
        fc1 = float(self.var_fc1.get().replace(",", "."))
        fc2 = float(self.var_fc2.get().replace(",", "."))
        return fs, fc1, fc2

    def _actualizar_diseno(self, forzar: bool = False):
        try:
            fs, fc1, fc2 = self._leer_parametros()
            if forzar or abs(fs - self.proc.fs) > 1e-9:
                if self.proc.is_streaming:
                    self.proc.detener_microfono()
                    self._refrescar_boton_mic()
                self.proc = AudioProcessor(fs, fc1, fc2)
            else:
                self.proc.actualizar_filtro(fc1, fc2)

        except Exception as exc:
            self.lbl_bw.configure(text="")
            self.lbl_estado.configure(text=str(exc), foreground=ERR)
            return

        self.lbl_bw.configure(
            text=f"BW = {self.proc.bw:,.0f} Hz  ·  {MIN_BW:.0f} ≤ BW ≤ {MAX_BW:.0f}  ·  "
                 f"fc2 ≤ {MAX_FC2:.0f}",
            foreground=OK,
        )
        self.lbl_estado.configure(text="", foreground=MUTED)
        self._dibujar_respuesta()
        self._dibujar_espectro()
        self.canvas.draw_idle()

    # ==================================================================
    # Dibujo
    # ==================================================================
    def _dibujar_respuesta(self):
        freqs, mag, mag_db, _ = self.proc.filter_core.respuesta_en_frecuencia(3000)
        self.linea_h.set_data(freqs, mag_db)

        tope = min(12000, self.proc.fs / 2)
        self.ax_respuesta.set_xlim(0, tope)
        self.ax_espectro.set_xlim(0, tope)

        for marca in self.marcas_corte:
            marca.remove()
        self.marcas_corte = []
        for fc in (self.proc.fc1, self.proc.fc2):
            if fc <= tope:
                self.marcas_corte.append(
                    self.ax_respuesta.axvline(fc, color=ERR, ls=":", lw=0.9, alpha=0.7))
                self.marcas_corte.append(
                    self.ax_espectro.axvline(fc, color=ERR, ls=":", lw=0.9, alpha=0.7))

        self.banda_paso.remove()
        self.banda_paso = self.ax_espectro.axvspan(
            self.proc.fc1, self.proc.fc2, color=ACCENT, alpha=0.08, lw=0)
        self.ax_espectro.legend(loc="upper right", frameon=False,
                                labelcolor=FG, fontsize=9)

    def _dibujar_espectro(self):
        if self.proc.is_streaming:
            freqs, db_in, db_out, _, _ = self.proc.espectro_tiempo_real()
        else:
            freqs, db_in, db_out, _, _ = self.proc.espectro_senal_completa()

        self.linea_entrada.set_data(freqs, db_in)
        self.linea_filtrada.set_data(freqs, db_out)
        self.ax_espectro.set_xlim(0, min(12000, self.proc.fs / 2))

    def _dibujar_tiempo(self):
        with self.proc.lock:
            x = self.proc.buffer_in[-2048:].copy()
            y = self.proc.buffer_out[-2048:].copy()

        if x.size == 0:
            return

        t_ms = np.arange(x.size) / self.proc.fs * 1000.0
        escala = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-9)
        self.linea_t_in.set_data(t_ms, x / escala)
        self.linea_t_out.set_data(t_ms, y / escala)
        self.ax_tiempo.set_xlim(0, max(t_ms[-1], 1e-3))
        self.ax_tiempo.set_ylim(-1.1, 1.1)

    # ------------------------------------------------------------------
    def _programar_actualizacion(self):
        self._actualizar_en_vivo()
        self._after_id = self.after(70, self._programar_actualizacion)

    def _actualizar_en_vivo(self):
        try:
            if self.proc.is_streaming:
                self._dibujar_espectro()
                self._dibujar_tiempo()
                self.canvas.draw_idle()
        except Exception:
            pass

    def destroy(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self.proc.detener_microfono()
        self.proc.detener_reproduccion()
        super().destroy()
