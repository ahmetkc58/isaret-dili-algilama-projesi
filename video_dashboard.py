import os
import cv2
import json
import time
import difflib
import threading
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

K = keras.backend

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "R", "S", "T", "U", "V", "Y", "Z", "del", "nothing", "space"
]

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.h5")
WORDLIST_PATH = os.getenv("WORDLIST_PATH", "turizm_sozluk.json")
VIDEO_PATH = os.getenv("VIDEO_PATH", "video.mp4")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
COMMIT_CONFIDENCE_THRESHOLD = float(os.getenv("COMMIT_CONFIDENCE_THRESHOLD", "0.20"))
TOP1_TOP2_MARGIN_MIN = float(os.getenv("TOP1_TOP2_MARGIN_MIN", "0.02"))
REPEAT_LABEL_GAP_SEC = float(os.getenv("REPEAT_LABEL_GAP_SEC", "0.80"))
STABLE_REQUIRED = int(os.getenv("STABLE_REQUIRED", "4"))
PROCESS_EVERY_N_FRAME = int(os.getenv("PROCESS_EVERY_N_FRAME", "2"))
TOP_K = int(os.getenv("TOP_K", "5"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))
DOMINANT_MIN_COUNT = int(os.getenv("DOMINANT_MIN_COUNT", "2"))
MIN_COMMIT_INTERVAL_SEC = float(os.getenv("MIN_COMMIT_INTERVAL_SEC", "0.45"))
NOTHING_RELEASE_REQUIRED = int(os.getenv("NOTHING_RELEASE_REQUIRED", "2"))


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / K.sqrt(s_squared_norm + K.epsilon())


class PrimaryCapsule(keras.layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super().__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = keras.layers.Conv2D(
            filters=dim_capsule * n_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )

    def call(self, inputs):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = K.reshape(outputs, [batch_size, -1, self.dim_capsule])
        return squash(outputs)


class CapsuleLayer(keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[self.num_capsule, 144, self.dim_capsule, 8],
            initializer="glorot_uniform",
            name="W",
        )
        self.built = True

    def call(self, inputs):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tile = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = tf.einsum("abci,bcji->abcj", inputs_tile, self.W)
        b = tf.zeros([tf.shape(inputs_hat)[0], self.num_capsule, tf.shape(inputs)[1]])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.einsum("abc,abcd->abd", c, inputs_hat)
            outputs = squash(s)
            if i < self.routings - 1:
                b += tf.einsum("abd,abcd->abc", outputs, inputs_hat)
        return outputs


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    input_layer = keras.layers.Input(shape=(224, 224, 3))
    base_model = keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_tensor=input_layer,
    )
    x = base_model.output
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    primary_caps = PrimaryCapsule(8, 16, 3, 2, "valid")(x)
    digit_caps = CapsuleLayer(len(CLASSES), 16, 3)(primary_caps)
    out_caps = keras.layers.Lambda(
        lambda z: K.sqrt(K.sum(K.square(z), axis=-1) + K.epsilon())
    )(digit_caps)

    model = keras.models.Model(inputs=input_layer, outputs=out_caps)
    model.load_weights(MODEL_PATH)
    return model


try:
    with open(WORDLIST_PATH, "r", encoding="utf-8") as f:
        WORDS = json.load(f).get("kelimeler", [])
except Exception:
    WORDS = []


def get_suggestion(text):
    if not text or not WORDS:
        return ""
    parts = text.split()
    if not parts:
        return ""
    last_word = parts[-1].upper()
    matches = difflib.get_close_matches(last_word, WORDS, n=1, cutoff=0.45)
    return matches[0] if matches else ""


def get_word_match(token, cutoff=0.40):
    if not token or not WORDS:
        return ""
    token = token.upper()
    matches = difflib.get_close_matches(token, WORDS, n=1, cutoff=cutoff)
    return matches[0] if matches else ""


def finalize_last_word(text):
    base = text.rstrip()
    if not base:
        return text

    has_trailing_space = text.endswith(" ")
    parts = base.split(" ")
    last_token = parts[-1]
    matched = get_word_match(last_token)
    if matched:
        parts[-1] = matched

    rebuilt = " ".join(parts)
    if has_trailing_space:
        rebuilt += " "
    return rebuilt


app = FastAPI(title="Sessiz Kelimeler Video Dashboard")
model = load_model()

state = {
    "running": False,
    "video_path": VIDEO_PATH,
    "frame_index": 0,
    "label": "",
    "confidence": 0.0,
    "sentence": "",
    "suggestion": "",
    "top_predictions": [],
    "history": [],
    "updated_at": 0.0,
    "error": "",
}

frame_lock = threading.Lock()
latest_jpeg = None
history_queue = deque(maxlen=15)


def apply_label(text, label):
    if label == "space":
        if text and not text.endswith(" "):
            return text + " "
        return text
    if label == "del":
        return text[:-1] if text else text
    if label == "nothing":
        return text
    return text + label


def top_k_predictions(preds):
    values = preds[0]
    order = np.argsort(values)[::-1][:TOP_K]
    return [
        {"label": CLASSES[int(i)], "score": float(values[int(i)])}
        for i in order
    ]


def process_video_loop():
    global latest_jpeg

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        state["error"] = f"Video acilamadi: {VIDEO_PATH}"
        state["running"] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps and fps > 0 else 0.03

    vote_window = deque(maxlen=max(1, SMOOTHING_WINDOW))
    last_committed = ""
    last_commit_time = 0.0
    nothing_release_count = 0
    ready_for_new_commit = True

    state["running"] = True
    state["error"] = ""

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        state["frame_index"] = frame_index

        if frame_index % PROCESS_EVERY_N_FRAME == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            inp = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

            preds = model.predict(inp, verbose=0)
            idx = int(np.argmax(preds[0]))
            label = CLASSES[idx]
            confidence = float(np.max(preds[0]))
            sorted_scores = np.sort(preds[0])
            top2 = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
            margin = confidence - top2

            state["label"] = label
            state["confidence"] = confidence
            state["top_predictions"] = top_k_predictions(preds)

            if confidence >= COMMIT_CONFIDENCE_THRESHOLD or margin >= TOP1_TOP2_MARGIN_MIN:
                vote_window.append(label)
                # Son penceredeki en sık etiketi alarak titreşimi azalt.
                dominant_label = max(set(vote_window), key=vote_window.count)
                dominant_count = vote_window.count(dominant_label)

                if dominant_label == "nothing":
                    nothing_release_count += 1
                    if nothing_release_count >= max(1, NOTHING_RELEASE_REQUIRED):
                        ready_for_new_commit = True
                else:
                    nothing_release_count = 0

                now_ts = time.time()
                enough_time_passed = (now_ts - last_commit_time) >= MIN_COMMIT_INTERVAL_SEC
                same_as_last = dominant_label == last_committed
                can_repeat = (
                    same_as_last
                    and ready_for_new_commit
                    and ((now_ts - last_commit_time) >= REPEAT_LABEL_GAP_SEC)
                )
                should_commit = (
                    dominant_label != "nothing"
                    and dominant_count >= max(1, DOMINANT_MIN_COUNT)
                    and enough_time_passed
                    and ((not same_as_last) or can_repeat)
                )

                if should_commit:
                    state["sentence"] = apply_label(state["sentence"], dominant_label)

                    if dominant_label == "space":
                        state["sentence"] = finalize_last_word(state["sentence"])

                    state["suggestion"] = get_suggestion(state["sentence"])
                    history_queue.appendleft(
                        {
                            "t": int(now_ts),
                            "label": dominant_label,
                            "confidence": round(confidence, 3),
                            "sentence": state["sentence"],
                        }
                    )
                    state["history"] = list(history_queue)
                    last_committed = dominant_label
                    last_commit_time = now_ts
                    ready_for_new_commit = False
                    nothing_release_count = 0
            else:
                vote_window.clear()
                nothing_release_count += 1
                if nothing_release_count >= max(1, NOTHING_RELEASE_REQUIRED):
                    ready_for_new_commit = True

        overlay = frame.copy()
        cv2.putText(overlay, f"Label: {state['label']} ({state['confidence']:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Cumle: {state['sentence']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if state["suggestion"]:
            cv2.putText(overlay, f"Oneri: {state['suggestion']}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ok, encoded = cv2.imencode(".jpg", overlay)
        if ok:
            with frame_lock:
                latest_jpeg = encoded.tobytes()

        state["updated_at"] = time.time()
        time.sleep(frame_delay)

    cap.release()
    state["running"] = False


@app.on_event("startup")
def startup_event():
    worker = threading.Thread(target=process_video_loop, daemon=True)
    worker.start()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Sessiz Kelimeler Dashboard</title>
<style>
:root { --bg: #091018; --card: #142232; --txt: #eaf2ff; --muted: #9fb4cc; --ok: #45e0a1; --warn: #ffd166; --line: #294661; }
body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: radial-gradient(circle at 20% 20%, #18304a, var(--bg) 45%); color: var(--txt); }
.wrap { max-width: 1200px; margin: 20px auto; padding: 0 14px; }
.card { background: var(--card); border-radius: 16px; padding: 14px; box-shadow: 0 8px 24px rgba(0,0,0,.28); }
.grid { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; align-items: start; }
img { width: 100%; border-radius: 12px; border: 1px solid #27415e; }
.kv { font-size: 15px; color: var(--muted); margin: 8px 0; }
.big { font-size: 22px; color: var(--txt); font-weight: 600; min-height: 34px; }
.badge { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #243b52; color: #cde2ff; font-size: 13px; }
.status-ok { color: var(--ok); }
.status-warn { color: var(--warn); }
.section { margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--line); }
.row { display: grid; grid-template-columns: 72px 1fr 52px; gap: 8px; margin: 6px 0; align-items: center; }
.bar-wrap { height: 10px; border-radius: 999px; background: #0f1a27; border: 1px solid #2b455f; overflow: hidden; }
.bar { height: 100%; background: linear-gradient(90deg, #4ad7b2, #6fa9ff); }
.mono { font-family: Consolas, Monaco, monospace; font-size: 12px; color: #cfe1ff; }
.history { max-height: 180px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: #0f1b29; }
.item { padding: 6px 0; border-bottom: 1px dashed #2c4763; }
.item:last-child { border-bottom: 0; }
@media (max-width: 850px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class=\"wrap\">
  <h2>Sessiz Kelimeler - Video Dashboard</h2>
  <div class=\"grid\">
    <div class=\"card\">
      <img src=\"/video_feed\" alt=\"Video feed\" />
    </div>
    <div class=\"card\">
      <div class=\"kv\">Durum: <span id=\"running\" class=\"badge\">bekleniyor</span></div>
            <div class="kv">Video: <span id="video_path" class="mono">-</span></div>
            <div class="kv">Frame: <span id="frame_index" class="mono">0</span></div>
      <div class=\"kv\">Son Etiket:</div>
      <div id=\"label\" class=\"big\">-</div>
      <div class=\"kv\">Güven:</div>
      <div id=\"conf\" class=\"big\">0.00</div>
      <div class=\"kv\">Cümle:</div>
      <div id=\"sentence\" class=\"big\">-</div>
      <div class=\"kv\">Öneri:</div>
      <div id=\"sugg\" class=\"big\">-</div>
            <div class="section">
                <div class="kv">Canli Top Tahminler</div>
                <div id="topk"></div>
            </div>
            <div class="section">
                <div class="kv">Islenen Etiket Gecmisi</div>
                <div id="history" class="history"></div>
            </div>
      <div id=\"err\" class=\"kv status-warn\"></div>
    </div>
  </div>
</div>
<script>
function fmtTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
}

function renderTopK(items) {
    const host = document.getElementById('topk');
    if (!items || !items.length) {
        host.innerHTML = '<div class="mono">Tahmin bekleniyor...</div>';
        return;
    }
    host.innerHTML = items.map((it) => {
        const p = Math.max(0, Math.min(100, it.score * 100));
        return '<div class="row">'
            + '<div class="mono">' + it.label + '</div>'
            + '<div class="bar-wrap"><div class="bar" style="width:' + p.toFixed(1) + '%"></div></div>'
            + '<div class="mono">' + p.toFixed(1) + '%</div>'
            + '</div>';
    }).join('');
}

function renderHistory(items) {
    const host = document.getElementById('history');
    if (!items || !items.length) {
        host.innerHTML = '<div class="mono">Henuz islenen etiket yok.</div>';
        return;
    }
    host.innerHTML = items.map((it) => (
        '<div class="item">'
        + '<div class="mono">' + fmtTime(it.t) + ' | ' + it.label + ' | ' + (it.confidence * 100).toFixed(1) + '%</div>'
        + '<div>' + (it.sentence || '-') + '</div>'
        + '</div>'
    )).join('');
}

async function tick() {
  try {
    const res = await fetch('/state');
    const s = await res.json();
    const running = document.getElementById('running');
    running.textContent = s.running ? 'calisiyor' : 'durdu';
    running.className = 'badge ' + (s.running ? 'status-ok' : 'status-warn');
        document.getElementById('video_path').textContent = s.video_path || '-';
        document.getElementById('frame_index').textContent = s.frame_index || 0;
    document.getElementById('label').textContent = s.label || '-';
    document.getElementById('conf').textContent = (s.confidence || 0).toFixed(2);
    document.getElementById('sentence').textContent = s.sentence || '-';
    document.getElementById('sugg').textContent = s.suggestion || '-';
        renderTopK(s.top_predictions || []);
        renderHistory(s.history || []);
    document.getElementById('err').textContent = s.error || '';
  } catch (e) {
    document.getElementById('err').textContent = 'Baglanti hatasi';
  }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
    """


@app.get("/state")
def get_state():
    return JSONResponse(state)


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.03)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print(f"Dashboard baslatiliyor. Video: {VIDEO_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8010)
