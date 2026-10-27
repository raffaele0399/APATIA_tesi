import csv
import os
import time
import numpy as np
from collections import Counter, deque, defaultdict
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope



WINDOW_SECONDS = 5.0
CALIBRATION_WINDOWS = 12
FRENETIC_PHASE_DURATION = 120.0
CALM_PHASE_NOTICE_AFTER = 120.0
ROLLING_LEN = 3

MULT_BASE = 0.5
MULT_RANGE = 1.5



STATE_MIN_SAMPLES = 8
STATE_BUFFER_MAXLEN = 60 

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


SESSION_LOG_PATH = None
EVENTS_LOG_PATH = None



STRESS_FEATURES = [
    "key_press_count", # numero di pressioni in 5 secondi
    "key_release_count", #numero di rilasci in 5 secondi
    "unique_keys", # tasti diversi premuti in 5 secondi
    "key_frequency", # frequenza di pressione dei tasti in 5 secondi
    "avg_inter_press", # intervallo medio tra le varie pressioni
    "std_inter_press", # irregolarità di tempo tra le varie pressioni
    "burstiness", # tendenza a premere a "raffiche"
    "stick_mean", # posizione media dello stick
    "stick_std", # variabilità di oscillazione dello stick
    "stick_max", # inclinazione massima dello stick
    "stick_count", # numero di muovimenti dello stick registrati in tutta la sessione
    "stick_range", # differenza di escursione, dal valore minimo a quello massimo dello stick
    "movement_energy", # energia del movimento di uno stick
    "sign_changes", # numero di cambi di direzione
    "action_intensity", # numero delle pressioni e dei tasti premuti 
    "key_diversity_rate", # tasso di diversità di pressioni, se è basso allora il giocatore sta spammando lo stesso tasto
    "key_entropy", # entropia nel premere i tasti, quindi se questo valore è basso vuol dire che c'è un tasto che viene premuto maggiormente
    "delta_key_press", # differenza numerica di tasti premuti in più o in meno rispetto alla finestra di 5 secondi precedente 
    "delta_key_freq", # differenza che indica l'aumento o la diminuzione del ritmo di pressione
    "delta_stick_std", # differenza di quanto è cambiata l'instabilità del movimento
    "rolling_key_freq", # media della frequenza di pressione sulle ultime finestre, se è basso stai mantenendo un ritmo basso per un attimo
    "rolling_stick_variability", # media dell'instabilità dello stick sulle ultime finestre 
]


REDUCED_FEATURES_FOR_MAHALANOBIS = ["key_frequency", "burstiness", "stick_std", "movement_energy", "unique_keys"]

MODEL_FEATURE_SETS = {
    "isolation_forest": STRESS_FEATURES,
    "one_class_svm": STRESS_FEATURES,
    "local_outlier_factor": STRESS_FEATURES,
    "mahalanobis": REDUCED_FEATURES_FOR_MAHALANOBIS,
}


OFFICIAL_MODEL_NAME = "isolation_forest"


def build_candidate_models():
    return {
        "isolation_forest": IsolationForest(n_estimators=150, contamination="auto", random_state=42),
        "one_class_svm": OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
        "local_outlier_factor": LocalOutlierFactor(n_neighbors=5, novelty=True, contamination="auto"),
        "mahalanobis": EllipticEnvelope(contamination=0.1, random_state=42),
    }

app = FastAPI()


def fresh_state():
    return {
        "prev_key_press": 0,
        "prev_key_freq": 0.0,
        "prev_stick_std": 0.0,
        "key_freq_history": deque(maxlen=ROLLING_LEN),
        "stick_std_history": deque(maxlen=ROLLING_LEN),

        "calibrated": False,
        "calibration_buffer": [],   

        "models": {},              
        "baseline_mean": None,
        "baseline_std": None,

        "state_buffers": defaultdict(lambda: deque(maxlen=STATE_BUFFER_MAXLEN)),
        "state_models": {},        

        "session_start_time": time.time(),

        "marker_started": False,
        "calm_start_session_time": None,
        "frenetic_start_session_time": None,
        "calm_notice_printed": False,
    }


state = fresh_state()


def start_new_session_logs():
    """Crea due nuovi file di log con un nome unico per QUESTA sessione (basato su
    data e ora di avvio). Cosi' giocatori diversi in un playtest finiscono automaticamente
    in file separati, invece di mescolarsi tutti nello stesso CSV. Chiamata all'avvio del
    server e ogni volta che arriva un /reset (nuovo giocatore)."""
    global SESSION_LOG_PATH, EVENTS_LOG_PATH
    stamp = time.strftime("%Y%m%d_%H%M%S")
    SESSION_LOG_PATH = os.path.join(LOG_DIR, f"session_{stamp}.csv")
    EVENTS_LOG_PATH = os.path.join(LOG_DIR, f"events_{stamp}.csv")

    with open(SESSION_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "session_time_s", "phase",
            "official_score", "top_contributing_feature", "top_feature_zscore",
            "baseline_used", "game_state",
        ] + [f"score_{name}" for name in build_candidate_models().keys()])

    with open(EVENTS_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "session_time_s", "event_name", "value"])

    print(f"[nuova sessione] log in: {os.path.basename(SESSION_LOG_PATH)}")


start_new_session_logs()


class ControllerEvent(BaseModel):
    type: Literal["key", "absolute"]
    event: str
    state: float
    t: float


class WindowPayload(BaseModel):
    events: List[ControllerEvent]
    game_state: Optional[str] = "unknown"


class GameEvent(BaseModel):
    event_name: str
    value: Optional[float] = None


def compute_features(events: List[ControllerEvent]) -> dict:
    keys = [e for e in events if e.type == "key"]
    abss = [e for e in events if e.type == "absolute"]

    key_press_count = sum(1 for e in keys if e.state == 1)
    key_release_count = sum(1 for e in keys if e.state == 0)
    unique_keys = len(set(e.event for e in keys))
    key_frequency = key_press_count / WINDOW_SECONDS

    press_times = sorted(e.t for e in keys if e.state == 1)
    if len(press_times) >= 2:
        intervals = np.diff(press_times)
        avg_inter_press = float(np.mean(intervals))
        std_inter_press = float(np.std(intervals)) if len(intervals) > 1 else 0.0
    else:
        avg_inter_press = 0.0
        std_inter_press = 0.0

    if abss:
        vals = np.array([e.state for e in abss])
        stick_mean = float(vals.mean())
        stick_std = float(vals.std()) if len(vals) > 1 else 0.0
        stick_max = float(np.abs(vals).max())
        stick_count = len(vals)
        stick_range = float(vals.max() - vals.min())
        movement_energy = float(np.sum(vals ** 2))
        sign_changes = int(np.sum(np.diff(np.sign(vals)) != 0)) if len(vals) > 2 else 0
    else:
        stick_mean = stick_std = stick_max = stick_range = 0.0
        stick_count = 0
        movement_energy = 0.0
        sign_changes = 0

    action_intensity = key_press_count + stick_count
    key_diversity_rate = unique_keys / (key_press_count + 1)

    if keys:
        counts = Counter(e.event for e in keys)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        key_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
    else:
        key_entropy = 0.0

    burstiness = (std_inter_press / (avg_inter_press + 1e-6)) if avg_inter_press > 0 else 0.0

    delta_key_press = key_press_count - state["prev_key_press"]
    delta_key_freq = key_frequency - state["prev_key_freq"]
    delta_stick_std = stick_std - state["prev_stick_std"]

    state["key_freq_history"].append(key_frequency)
    state["stick_std_history"].append(stick_std)
    rolling_key_freq = float(np.mean(state["key_freq_history"]))
    rolling_stick_variability = float(np.mean(state["stick_std_history"]))

    state["prev_key_press"] = key_press_count
    state["prev_key_freq"] = key_frequency
    state["prev_stick_std"] = stick_std

    return {
        "key_press_count": key_press_count,
        "key_release_count": key_release_count,
        "unique_keys": unique_keys,
        "key_frequency": key_frequency,
        "avg_inter_press": avg_inter_press,
        "std_inter_press": std_inter_press,
        "burstiness": burstiness,
        "stick_mean": stick_mean,
        "stick_std": stick_std,
        "stick_max": stick_max,
        "stick_count": stick_count,
        "stick_range": stick_range,
        "movement_energy": movement_energy,
        "sign_changes": sign_changes,
        "action_intensity": action_intensity,
        "key_diversity_rate": key_diversity_rate,
        "key_entropy": key_entropy,
        "delta_key_press": delta_key_press,
        "delta_key_freq": delta_key_freq,
        "delta_stick_std": delta_stick_std,
        "rolling_key_freq": rolling_key_freq,
        "rolling_stick_variability": rolling_stick_variability,
    }


def run_calibration():
    buf = state["calibration_buffer"]
    X = np.array([[row[f] for f in STRESS_FEATURES] for row in buf])

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1e-6
    state["baseline_mean"] = mean
    state["baseline_std"] = std

    candidates = build_candidate_models()
    for name, model in candidates.items():
        feat_list = MODEL_FEATURE_SETS[name]
        X_model = np.array([[row[f] for f in feat_list] for row in buf])
        model.fit(X_model)
        calib_scores = -model.score_samples(X_model) if hasattr(model, "score_samples") else -model.decision_function(X_model)
        abn_mean = float(calib_scores.mean())
        abn_std = float(calib_scores.std()) if calib_scores.std() > 0 else 1.0
        state["models"][name] = {"model": model, "abn_mean": abn_mean, "abn_std": abn_std}

    state["calibrated"] = True


def raw_anomaly_score(model, x: np.ndarray) -> float:
    if hasattr(model, "score_samples"):
        return float(-model.score_samples(x)[0])
    return float(-model.decision_function(x)[0])


def score_all_models(feats: dict) -> dict:
    """Calcola il punteggio normalizzato 0-1 per OGNI modello di confronto.
    Ritorna un dict {nome_modello: punteggio}."""
    scores = {}
    for name, bundle in state["models"].items():
        feat_list = MODEL_FEATURE_SETS[name]
        x = np.array([[feats[f] for f in feat_list]])
        abnormality = raw_anomaly_score(bundle["model"], x)
        normalized = (abnormality - bundle["abn_mean"]) / bundle["abn_std"]
        scores[name] = float(np.clip(normalized, 0.0, 3.0) / 3.0)
    return scores


def direction_and_top_feature(feats: dict):
    """Ritorna (composite_direction, nome_feature_piu_anomala, suo_zscore).
    Usato sia per decidere la direzione (sopra/sotto la norma) sia per la spiegabilita'."""
    mean = state["baseline_mean"]
    std = state["baseline_std"]
    x = np.array([feats[f] for f in STRESS_FEATURES])
    z = (x - mean) / std

    composite_direction = float(np.mean(z))
    top_idx = int(np.argmax(np.abs(z)))
    top_feature = STRESS_FEATURES[top_idx]
    top_zscore = float(z[top_idx])

    return composite_direction, top_feature, top_zscore


def clamp(v, lo=1.0, hi=5.0):
    return max(lo, min(hi, v))


def update_state_buffer_and_maybe_fit(game_state: str, feats: dict):
    """Accumula le feature per QUESTO specifico stato di gioco (es. 'combattimento',
    'esplorazione'). Appena ci sono abbastanza campioni, addestra la baseline dedicata
    a quello stato UNA SOLA VOLTA: da quel momento resta fissa per il resto della sessione,
    non viene piu' riaddestrata. Questo e' intenzionale: se la baseline si aggiornasse
    continuamente, un comportamento sostenuto ed elevato (es. un lungo combattimento
    intenso) finirebbe per essere assorbito come 'nuovo normale' entro un minuto o due,
    smettendo di essere segnalato — l'opposto di quello che vogliamo su una sessione lunga."""
    if game_state in state["state_models"]:
        return  

    buf = state["state_buffers"][game_state]
    buf.append(feats)

    if len(buf) < STATE_MIN_SAMPLES:
        return  

    X = np.array([[row[f] for f in STRESS_FEATURES] for row in buf])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1e-6

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(X)
    calib_scores = -model.score_samples(X)
    abn_mean = float(calib_scores.mean())
    abn_std = float(calib_scores.std()) if calib_scores.std() > 0 else 1.0

    state["state_models"][game_state] = {
        "model": model, "mean": mean, "std": std,
        "abn_mean": abn_mean, "abn_std": abn_std, "n_samples": len(buf),
    }


def score_with_bundle(bundle: dict, feats: dict):
    """Calcola punteggio 0-1 e feature responsabile usando una baseline specifica
    (puo' essere quella globale o quella dedicata a uno stato)."""
    x_vec = np.array([feats[f] for f in STRESS_FEATURES])
    x = x_vec.reshape(1, -1)

    abnormality = raw_anomaly_score(bundle["model"], x)
    normalized = (abnormality - bundle["abn_mean"]) / bundle["abn_std"]
    
    magnitude = float(np.clip(normalized, 0.0, 2.0) / 2.0)

    z = (x_vec - bundle["mean"]) / bundle["std"]
    composite_direction = float(np.mean(z))
    top_idx = int(np.argmax(np.abs(z)))
    top_feature = STRESS_FEATURES[top_idx]
    top_zscore = float(z[top_idx])

    score = magnitude if composite_direction > 0 else magnitude * 0.3
    return score, top_feature, top_zscore


def score_official(game_state: str, feats: dict):
    """Decide quale baseline usare per pilotare davvero il gioco:
    - se lo stato attuale ha gia' abbastanza campioni -> usa la SUA baseline dedicata
    - altrimenti -> usa la baseline GLOBALE di sessione come riserva
    Ritorna anche una stringa che dice quale delle due e' stata usata, utile per
    il log e per discuterne in tesi."""
    state_bundle = state["state_models"].get(game_state)
    if state_bundle is not None:
        score, top_feature, top_zscore = score_with_bundle(state_bundle, feats)
        baseline_used = f"per_stato:{game_state}(n={state_bundle['n_samples']})"
        return score, top_feature, top_zscore, baseline_used

    global_bundle = {
        "model": state["models"][OFFICIAL_MODEL_NAME]["model"],
        "mean": state["baseline_mean"],
        "std": state["baseline_std"],
        "abn_mean": state["models"][OFFICIAL_MODEL_NAME]["abn_mean"],
        "abn_std": state["models"][OFFICIAL_MODEL_NAME]["abn_std"],
    }
    score, top_feature, top_zscore = score_with_bundle(global_bundle, feats)
    return score, top_feature, top_zscore, "globale(riserva)"


def log_window(phase: str, official_score: float, all_scores: dict, top_feature: str,
               top_zscore: float, baseline_used: str = "", game_state: str = ""):
    session_time = time.time() - state["session_start_time"]
    with open(SESSION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        row = [time.time(), round(session_time, 2), phase, official_score, top_feature,
               round(top_zscore, 3), baseline_used, game_state]
        row += [all_scores.get(name, "") for name in build_candidate_models().keys()]
        writer.writerow(row)


@app.post("/predict")
def predict(payload: WindowPayload):
    feats = compute_features(payload.events)

    if not state["calibrated"]:
        state["calibration_buffer"].append(feats)
        progress = len(state["calibration_buffer"])

        if progress >= CALIBRATION_WINDOWS:
            run_calibration()
            print("\n" + "=" * 60)
            print(">>> CALIBRAZIONE COMPLETATA (60s) — puoi premere F5 (calm_start) <<<")
            print("=" * 60 + "\n")
            log_window("calibration_complete", 0.0, {}, "", 0.0)
            return {"phase": "calibration_complete", "stress_raw": 1.0}

        log_window("calibrating", 0.0, {}, "", 0.0)
        return {
            "phase": "calibrating",
            "progress": progress,
            "target": CALIBRATION_WINDOWS,
            "stress_raw": 1.0,
        }

    game_state = payload.game_state or "unknown"
    update_state_buffer_and_maybe_fit(game_state, feats)

    all_scores = score_all_models(feats)

    official_score, top_feature, top_zscore, baseline_used = score_official(game_state, feats)

    stress_raw = clamp(1.0 + 4.0 * official_score)

    log_window("active", official_score, all_scores, top_feature, top_zscore, baseline_used, game_state)

    session_time_now = time.time() - state["session_start_time"]

    if (state["calm_start_session_time"] is not None
            and not state["calm_notice_printed"]
            and (session_time_now - state["calm_start_session_time"]) >= CALM_PHASE_NOTICE_AFTER):
        print("\n" + "=" * 60)
        print(">>> 2 MINUTI DI FASE CALMA COMPLETATI — puoi premere F6 (frenetic_start) <<<")
        print("=" * 60 + "\n")
        state["calm_notice_printed"] = True


    if (not state["marker_started"]
            and state["frenetic_start_session_time"] is not None
            and (session_time_now - state["frenetic_start_session_time"]) >= FRENETIC_PHASE_DURATION):
        state["marker_started"] = True
        print("\n" + "=" * 60)
        print(">>> 2 MINUTI DI FASE FRENETICA COMPLETATI — la barra e' ora attiva <<<")
        print("=" * 60 + "\n")

    bar_stress_raw = stress_raw if state["marker_started"] else 1.0

    return {
        "phase": "active",
        "stress_raw": bar_stress_raw,
        "stress_score_debug": official_score,
        "top_contributing_feature": top_feature,
        "top_feature_zscore": round(top_zscore, 3),
        "comparison_scores": all_scores,
        "baseline_used": baseline_used,
        "game_state": game_state,
    }


@app.post("/log_event")
def log_event(event: GameEvent):
    """Endpoint per registrare eventi di gioco (danno subito, cambio stanza, ecc.) con
    timestamp, cosi' da poterli confrontare a posteriori con la curva di stress stimata."""
    session_time = time.time() - state["session_start_time"]
    with open(EVENTS_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), round(session_time, 2), event.event_name, event.value])

    if event.event_name == "calm_start":
        state["key_freq_history"].clear()
        state["stick_std_history"].clear()
        state["prev_key_press"] = 0
        state["prev_key_freq"] = 0.0
        state["prev_stick_std"] = 0.0
        state["calm_start_session_time"] = session_time

    if event.event_name == "frenetic_start":
        state["frenetic_start_session_time"] = session_time

    return {"status": "logged"}


@app.post("/reset")
def reset():
    global state
    state = fresh_state()
    start_new_session_logs() 
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "ok", "calibrated": state["calibrated"]}


# Comando per avviare l'ascolto
# uvicorn realtime_anomaly_server:app --host 127.0.0.1 --port 8000