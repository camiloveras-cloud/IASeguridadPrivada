# Demo: Detección de Intrusión en Zona Restringida

Aplicación local que detecta **personas** en un video o webcam, permite marcar
una zona restringida con el mouse, y genera una alerta visual, sonora y una
captura de evidencia cuando una persona entra en esa zona.

**Nota conceptual:** el modelo solo detecta personas. No identifica intención
ni clasifica a nadie como amenaza. Los estados que verás en pantalla son
`NORMAL` / `ALERTA`, y en consola `person detected in restricted zone`.

---

## 1. Requisitos

- Python 3.10 u 11 (CPU, sin GPU).
- ~700 MB libres para dependencias (`torch`/`opencv`/`ultralytics`).
- macOS, Windows o Linux con webcam opcional.

## 2. Instalación

### macOS / Linux / WSL

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Este proyecto ya trae un `.venv` creado con `uv` (Python 3.11.15) y las
> dependencias instaladas. Si vas a moverlo a otra máquina, borra `.venv/` y
> repite los pasos de arriba.

## 3. Preparar la demo (antes de presentar)

Corre siempre esto antes de la presentación — valida Python, dependencias,
video, audio, descarga/precarga el modelo YOLO, crea las carpetas de salida y
ejecuta una inferencia de prueba real sobre el video incluido:

```bash
python prepare_demo.py
```

Si todo está en orden verás `[SUMMARY] Everything looks ready.` al final.

## 4. Ejecutar la demo

Con el video incluido, en loop (se reinicia solo al terminar):

```bash
python app.py --source "Test Videos/Pencuri.mp4" --device cpu --loop
```

Con la webcam:

```bash
python app.py --source 0 --device cpu
```

### Flags disponibles

| Flag           | Default              | Descripción                                             |
|----------------|-----------------------|----------------------------------------------------------|
| `--source`     | `0`                   | Ruta a un video, o índice de cámara (`0`, `1`, ...).      |
| `--device`     | `cpu`                 | Dispositivo de inferencia (solo CPU en esta demo).        |
| `--confidence` | `0.45`                | Confianza mínima (0–1) para aceptar una detección.        |
| `--cooldown`   | `5.0`                 | Segundos entre alertas/capturas consecutivas.              |
| `--loop`       | desactivado           | Reinicia el video al llegar al final (ignorado en webcam). |
| `--model`      | `yolov8n.pt`          | Ruta al modelo YOLO (se descarga solo si falta).           |
| `--alarm`      | `assets/alarm.wav`    | Sonido de alerta. Si falta, avisa y continúa sin audio.    |
| `--mute`       | desactivado           | Desactiva el audio aunque exista el archivo.                |

## 5. Controles en la ventana

| Acción              | Control          |
|---------------------|------------------|
| Agregar punto a la zona | Clic izquierdo |
| Borrar el polígono actual | Clic derecho |
| Confirmar la zona (mín. 3 puntos) | Enter |
| Reiniciar la zona | R |
| Salir | Q o Escape |

Flujo típico: haz 4-6 clics izquierdos para dibujar el polígono de la zona
restringida → Enter para confirmarla → la detección de personas se activa
automáticamente a partir de ese momento.

## 6. Evidencia generada

Cada incidente (persona detectada dentro de la zona, respetando el cooldown)
guarda una captura en:

```
outputs/incidents/incident_YYYYMMDD_HHMMSS_mmm.jpg
```

## 7. Solución de errores comunes

- **`[ERROR] Could not open webcam`**: no hay cámara conectada, está en uso
  por otra app, o el sistema no dio permisos de cámara a la terminal/IDE
  (en macOS: *Preferencias del Sistema → Privacidad y Seguridad → Cámara*).
- **`[ERROR] Video file not found`**: revisa la ruta pasada a `--source`
  (usa comillas si el nombre tiene espacios, como `"Test Videos/Pencuri.mp4"`).
- **`[WARN] Alarm file not found ... Continuing without audio alerts`**: la
  demo sigue funcionando, solo sin sonido. Verifica que exista
  `assets/alarm.wav`, o pasa otro archivo con `--alarm`.
- **La ventana no aparece / se congela**: confirma que estás corriendo el
  script en una sesión con acceso a pantalla (no en SSH puro sin forwarding
  de GUI), y que no hay otra ventana `Restricted Zone Monitor` abierta.
- **Inferencia muy lenta**: baja la resolución del video de entrada, o usa un
  video más corto para la demo; el modelo (`yolov8n`) ya es la variante más
  ligera de YOLOv8.
- **`ModuleNotFoundError`**: el entorno virtual no está activado, o falta
  correr `pip install -r requirements.txt`.

## 8. Guion sugerido para la presentación (5-7 min)

1. **Contexto (30s):** "Esta demo muestra detección de personas en tiempo
   real sobre CPU, aplicada a seguridad física: definimos una zona restringida
   y el sistema avisa cuando alguien la cruza."
2. **Preparación en vivo (30s):** correr `python prepare_demo.py` con la
   audiencia viendo la consola — refuerza que todo corre localmente, sin nube.
3. **Arrancar la demo (10s):**
   `python app.py --source "Test Videos/Pencuri.mp4" --device cpu --loop`
4. **Dibujar la zona (30s):** clics izquierdos formando un polígono sobre una
   zona del video (ej. una puerta o pasillo), Enter para confirmar.
5. **Mostrar detección normal (1 min):** señalar el bounding box verde y el
   estado `NORMAL` mientras una persona camina fuera de la zona.
6. **Mostrar la alerta (1-2 min):** cuando una persona entra a la zona:
   recuadro rojo, estado `ALERTA`, sonido, contador de incidentes subiendo, y
   mencionar que se guardó la captura en `outputs/incidents/`.
7. **Cooldown (30s):** explicar que el sistema no satura de alertas ni
   capturas repetidas — hay un tiempo mínimo configurable (`--cooldown`)
   entre incidentes.
8. **Cierre (30s):** aclarar que el sistema **solo detecta personas**, no
   identifica quiénes son ni juzga intenciones — la decisión y la respuesta
   siguen siendo humanas; esto es una herramienta de apoyo a seguridad física.
9. **Q&A:** mostrar `outputs/incidents/` con las capturas generadas durante
   la demo como evidencia tangible.

## 9. Estructura del proyecto

```
intrusion-detection-demo/
├── app.py                  # Aplicación principal
├── prepare_demo.py         # Chequeo previo a la demo
├── requirements.txt
├── README_DEMO.md
├── .gitignore
├── assets/
│   └── alarm.wav           # Tono de alerta generado (sin dependencias externas)
├── Test Videos/
│   └── Pencuri.mp4         # Video de prueba, personas caminando (CC BY 4.0)
├── outputs/
│   └── incidents/          # Capturas de evidencia (se crea/llena en runtime)
└── yolov8n.pt               # Pesos del modelo (se descargan solos, ignorado por git)
```

## 10. Créditos y licencias

- Video de prueba `Test Videos/Pencuri.mp4`: adaptado de
  `one-by-one-person-detection.mp4`, repositorio
  [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos)
  (licencia CC BY 4.0).
- Modelo de detección: YOLOv8n (Ultralytics), descargado automáticamente la
  primera vez que se ejecuta `prepare_demo.py` o `app.py`.
- Tono de alerta `assets/alarm.wav`: generado localmente por síntesis de
  onda (sin dependencias ni licencias de terceros).
