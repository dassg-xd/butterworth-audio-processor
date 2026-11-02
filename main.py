"""
main.py

Punto de entrada del Laboratorio 1.
Arquitectura: main (arranque) -> view (interfaz) -> core (matemáticas y audio).

Ejecución:
    python main.py
"""

import sys


def main() -> int:
    try:
        from view.main_window import MainWindow
    except ImportError as exc:
        print("Falta una dependencia. Instale con:  pip install -r requirements.txt", file=sys.stderr)
        print(f"Detalle: {exc}", file=sys.stderr)
        return 1

    try:
        app = MainWindow()
        app.mainloop()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
