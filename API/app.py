import cv2
import mediapipe as mp
import csv
import os
import threading
import time
from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI(title="HELP AI - Reconhecimento de Gestos")

# Montar a pasta de arquivos estáticos (serve /static)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CSV de gestos
csv_path = "dados.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dedos,mensagem\n")

# Carregar gestos do CSV
gestos_csv = {}
with open(csv_path, encoding="utf-8") as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        dedos = tuple(sorted(int(d) for d in linha["dedos"].split("|") if d))
        gestos_csv[dedos] = linha["mensagem"]

# Variáveis globais
current_frame_jpeg = None
current_gesture = ("Nenhum", False)  # (mensagem, crítico)
_frame_lock = threading.Lock()
_running = True

# Buffer para estabilizar gestos
gesture_buffer = []
BUFFER_SIZE = 5  # frames consecutivos para confirmar o gesto

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- Funções de reconhecimento ----------------

def detectar_gesto(landmarks):
    dedos_estendidos = []

    # Dedos: indicador, médio, anelar, mindinho
    for tip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if landmarks[tip].y < landmarks[dip].y:
            dedos_estendidos.append(tip)

    # Polegar (ajuste para mão direita/esquerda)
    if landmarks[17].x < landmarks[5].x:  # mão direita
        if landmarks[4].x > landmarks[3].x:
            dedos_estendidos.append(4)
    else:  # mão esquerda
        if landmarks[4].x < landmarks[3].x:
            dedos_estendidos.append(4)

    chave = tuple(sorted(dedos_estendidos))
    mensagem = gestos_csv.get(chave, "Gesto não reconhecido")
    critico = any(word in mensagem.lower() for word in ["emergência", "socorro", "pânico", "help"])
    return mensagem, critico

def atualizar_gesto_stavel(novo_gesto):
    """Mantém um buffer dos últimos N gestos e retorna o mais frequente"""
    gesture_buffer.append(novo_gesto)
    if len(gesture_buffer) > BUFFER_SIZE:
        gesture_buffer.pop(0)
    # retorna o gesto mais frequente nos últimos frames
    return max(set(gesture_buffer), key=gesture_buffer.count)

# ---------------- Loop de captura ----------------

def capture_loop():
    global current_frame_jpeg, current_gesture, _running
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    
    while _running:
        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        gesture_text = "Nenhum"
        critico = False

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                g, c = detectar_gesto(handLms.landmark)
                gesture_text = atualizar_gesto_stavel(g)
                critico = c

        ret, jpeg = cv2.imencode('.jpg', img)
        if ret:
            with _frame_lock:
                current_frame_jpeg = jpeg.tobytes()
                current_gesture = (gesture_text, critico)
        time.sleep(0.03)

    cap.release()
    hands.close()

threading.Thread(target=capture_loop, daemon=True).start()

# ---------------- Gerador de MJPEG ----------------

def mjpeg_generator():
    global current_frame_jpeg
    while True:
        with _frame_lock:
            frame = current_frame_jpeg
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype="uint8")
            ret, blank_jpg = cv2.imencode('.jpg', blank)
            chunk = blank_jpg.tobytes()
        else:
            chunk = frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
        time.sleep(0.03)

# ---------------- Rotas FastAPI ----------------

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
async def events(request: Request):
    def event_generator():
        last_sent = None
        while True:
            with _frame_lock:
                gesto, critico = current_gesture
            if gesto != last_sent:
                last_sent = gesto
                yield f'data: {{"gesto": "{gesto}", "critical": {str(critico).lower()}}}\n\n'
            time.sleep(0.15)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/save_gesture")
async def save_gesture(dedos: str = Form(...), mensagem: str = Form(...)):
    key = tuple(sorted(int(d) for d in dedos.split("|") if dedos.strip()))
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f'{ "|".join(map(str, key)) },"{mensagem}"\n')
    gestos_csv[key] = mensagem
    return JSONResponse({"status": "ok", "dedos": "|".join(map(str, key)), "mensagem": mensagem})

@app.get("/export")
async def export():
    out = "gestos_exportados.txt"
    with open(out, "w", encoding="utf-8") as f:
        for dedos, mensagem in gestos_csv.items():
            f.write(f"{'|'.join(map(str, dedos))}: {mensagem}\n")
    return FileResponse(out, media_type="text/plain", filename=out)

@app.get("/dados.csv")
def get_csv():
    return FileResponse(csv_path)

@app.on_event("shutdown")
def shutdown_event():
    global _running
    _running = False

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("static/index.html")
