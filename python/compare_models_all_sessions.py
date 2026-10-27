"""
compare_models_all_sessions.py

Calcola TPR e FPR per ciascuno dei quattro modelli di anomaly detection
(Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope),
usando i punteggi di confronto a 3-sigma già presenti nei CSV di sessione
(colonne score_isolation_forest, score_one_class_svm,
score_local_outlier_factor, score_mahalanobis), aggregando su tutte le
sessioni con marker calm_start/frenetic_start disponibili in logs/.

Stessa logica di individuazione delle fasi calma/frenetica usata in
evaluate_all_sessions.py, applicata però ai quattro modelli di confronto
invece che al solo punteggio ufficiale (official_score).
"""

import csv
import os
import glob
import statistics

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTION_THRESHOLD = 0.4 
FRENETIC_PHASE_DURATION_DEFAULT = 120.0

MODELS = [
    "score_isolation_forest",
    "score_one_class_svm",
    "score_local_outlier_factor",
    "score_mahalanobis",
]

MODEL_LABELS = {
    "score_isolation_forest": "Isolation Forest",
    "score_one_class_svm": "One-Class SVM",
    "score_local_outlier_factor": "Local Outlier Factor",
    "score_mahalanobis": "Elliptic Envelope",
}


def load_windows(session_path):
    """Carica le finestre 'active' di una sessione, con i punteggi dei quattro modelli."""
    windows = []
    with open(session_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "active":
                continue
            w = {"t": float(row["session_time_s"])}
            for m in MODELS:
                v = row.get(m, "")
                w[m] = float(v) if v not in (None, "") else None
            windows.append(w)
    return windows


def load_events(events_path):
    events = []
    if not os.path.exists(events_path):
        return events
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            events.append({
                "t": float(row["session_time_s"]),
                "name": row["event_name"],
            })
    return events


all_session_files = sorted(glob.glob(os.path.join(LOG_DIR, "session_*.csv")))
if not all_session_files:
    print("Nessuna sessione trovata in logs/.")
    exit()


results = {m: {"tpr": [], "fpr": []} for m in MODELS}
sessions_used = []

for session_path in all_session_files:
    stamp = os.path.basename(session_path).replace("session_", "").replace(".csv", "")
    events_path = os.path.join(LOG_DIR, f"events_{stamp}.csv")

    windows = load_windows(session_path)
    events = load_events(events_path)

    if not windows:
        continue

    calm_start = next((e["t"] for e in events if e["name"] == "calm_start"), None)
    frenetic_start = next((e["t"] for e in events if e["name"] == "frenetic_start"), None)

    if calm_start is None or frenetic_start is None:
        continue

    frenetic_end = frenetic_start + FRENETIC_PHASE_DURATION_DEFAULT

    calm_phase = [w for w in windows if calm_start <= w["t"] < frenetic_start]
    frenetic_phase = [w for w in windows if frenetic_start <= w["t"] < frenetic_end]

    if not calm_phase or not frenetic_phase:
        continue

    sessions_used.append(stamp)

    for m in MODELS:
        calm_scores = [w[m] for w in calm_phase if w[m] is not None]
        frenetic_scores = [w[m] for w in frenetic_phase if w[m] is not None]
        if not calm_scores or not frenetic_scores:
            continue

        false_positives = sum(1 for s in calm_scores if s > DETECTION_THRESHOLD)
        true_positives = sum(1 for s in frenetic_scores if s > DETECTION_THRESHOLD)
        fpr = false_positives / len(calm_scores)
        tpr = true_positives / len(frenetic_scores)

        results[m]["tpr"].append(tpr)
        results[m]["fpr"].append(fpr)


print("=" * 100)
print(f"CONFRONTO TRA I QUATTRO MODELLI DI ANOMALY DETECTION — su {len(sessions_used)} sessioni")
print("=" * 100)
print(f"Sessioni incluse: {', '.join(sessions_used)}\n")

header = f"{'Modello':<24}{'N sessioni':>11}{'TPR medio':>12}{'TPR std':>10}{'FPR medio':>12}{'FPR std':>10}"
print(header)
print("-" * len(header))

rows_for_csv = []
for m in MODELS:
    tprs = results[m]["tpr"]
    fprs = results[m]["fpr"]
    n = len(tprs)
    if n == 0:
        continue
    tpr_mean = statistics.mean(tprs)
    tpr_std = statistics.stdev(tprs) if n > 1 else 0.0
    fpr_mean = statistics.mean(fprs)
    fpr_std = statistics.stdev(fprs) if n > 1 else 0.0
    print(f"{MODEL_LABELS[m]:<24}{n:>11}{tpr_mean:>12.1%}{tpr_std:>10.1%}{fpr_mean:>12.1%}{fpr_std:>10.1%}")
    rows_for_csv.append([MODEL_LABELS[m], n, f"{tpr_mean:.4f}", f"{tpr_std:.4f}", f"{fpr_mean:.4f}", f"{fpr_std:.4f}"])

csv_path = os.path.join(OUTPUT_DIR, "confronto_modelli_aggregato.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["modello", "n_sessioni", "tpr_medio", "tpr_std", "fpr_medio", "fpr_std"])
    for row in rows_for_csv:
        writer.writerow(row)

print(f"\nTabella salvata in: {csv_path}")