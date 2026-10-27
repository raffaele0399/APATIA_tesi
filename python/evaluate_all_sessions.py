import csv
import os
import glob
import statistics



LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTION_THRESHOLD = 0.4

FRENETIC_PHASE_DURATION_DEFAULT = 120.0


all_session_files = sorted(glob.glob(os.path.join(LOG_DIR, "session_*.csv")))
if not all_session_files:
    print("Nessuna sessione trovata in logs/.")
    exit()


def load_windows(session_path):
    """Carica solo le finestre 'active' di una sessione, stesso formato di analyze_session.py."""
    windows = []
    with open(session_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "active":
                continue
            windows.append({
                "t": float(row["session_time_s"]),
                "official_score": float(row["official_score"]),
            })
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



results = []

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

    calm_scores = [w["official_score"] for w in calm_phase]
    frenetic_scores = [w["official_score"] for w in frenetic_phase]

    calm_mean = sum(calm_scores) / len(calm_scores)
    frenetic_mean = sum(frenetic_scores) / len(frenetic_scores)
    delta = frenetic_mean - calm_mean

    false_positives = sum(1 for s in calm_scores if s > DETECTION_THRESHOLD)
    true_positives = sum(1 for s in frenetic_scores if s > DETECTION_THRESHOLD)
    fpr = false_positives / len(calm_scores)
    tpr = true_positives / len(frenetic_scores)

    first_detection = next(
        (w["t"] for w in frenetic_phase if w["official_score"] > DETECTION_THRESHOLD), None
    )
    latency = (first_detection - frenetic_start) if first_detection is not None else None

    results.append({
        "stamp": stamp,
        "n_calm": len(calm_phase),
        "n_frenetic": len(frenetic_phase),
        "calm_mean": calm_mean,
        "frenetic_mean": frenetic_mean,
        "delta": delta,
        "tpr": tpr,
        "fpr": fpr,
        "latency": latency,
    })



print("=" * 100)
print(f"TABELLA AGGREGATA — PROTOCOLLO CALMA/FRENETICA su {len(results)} sessioni con marker")
print("=" * 100)

if not results:
    print("Nessuna sessione con marker calm_start/frenetic_start trovata.")
    print("Rilancia questo script dopo aver raccolto sessioni con il protocollo attivato.")
    exit()

header = f"{'Sessione':<18}{'N calma':>9}{'N frenet':>10}{'Media calma':>13}{'Media frenet':>14}{'Delta':>9}{'TPR':>8}{'FPR':>8}{'Latenza(s)':>12}"
print(header)
print("-" * len(header))
for r in results:
    lat_str = f"{r['latency']:.1f}" if r["latency"] is not None else "n/d"
    print(f"{r['stamp']:<18}{r['n_calm']:>9}{r['n_frenetic']:>10}"
          f"{r['calm_mean']:>13.3f}{r['frenetic_mean']:>14.3f}{r['delta']:>9.3f}"
          f"{r['tpr']:>8.1%}{r['fpr']:>8.1%}{lat_str:>12}")



def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s

calm_m, calm_s = mean_std([r["calm_mean"] for r in results])
frenetic_m, frenetic_s = mean_std([r["frenetic_mean"] for r in results])
delta_m, delta_s = mean_std([r["delta"] for r in results])
tpr_m, tpr_s = mean_std([r["tpr"] for r in results])
fpr_m, fpr_s = mean_std([r["fpr"] for r in results])
lat_m, lat_s = mean_std([r["latency"] for r in results])

print("\n" + "=" * 60)
print(f"STATISTICHE AGGREGATE (media ± deviazione standard, n={len(results)} sessioni)")
print("=" * 60)
print(f"Media punteggio fase calma:     {calm_m:.3f} ± {calm_s:.3f}")
print(f"Media punteggio fase frenetica: {frenetic_m:.3f} ± {frenetic_s:.3f}")
print(f"Delta (frenetica - calma):      {delta_m:.3f} ± {delta_s:.3f}")
print(f"TPR:                            {tpr_m:.1%} ± {tpr_s:.1%}")
print(f"FPR:                            {fpr_m:.1%} ± {fpr_s:.1%}")
if lat_m is not None:
    print(f"Latenza di rilevamento:         {lat_m:.1f}s ± {lat_s:.1f}s")
else:
    print("Latenza di rilevamento: nessuna sessione ha superato la soglia in fase frenetica")

print("\nQueste cifre aggregate (media ± dev. standard su piu' sessioni/tester) sono piu'")
print("difendibili in sede di discussione tesi rispetto a un singolo valore da una sola sessione.")



csv_path = os.path.join(OUTPUT_DIR, "tabella_calma_frenetica_aggregata.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sessione", "n_calma", "n_frenetica", "media_calma", "media_frenetica",
                      "delta", "tpr", "fpr", "latenza_s"])
    for r in results:
        writer.writerow([r["stamp"], r["n_calm"], r["n_frenetic"],
                          f"{r['calm_mean']:.4f}", f"{r['frenetic_mean']:.4f}", f"{r['delta']:.4f}",
                          f"{r['tpr']:.4f}", f"{r['fpr']:.4f}",
                          f"{r['latency']:.2f}" if r["latency"] is not None else ""])
    writer.writerow([])
    writer.writerow(["MEDIA_AGGREGATA", "", "", f"{calm_m:.4f}", f"{frenetic_m:.4f}", f"{delta_m:.4f}",
                      f"{tpr_m:.4f}", f"{fpr_m:.4f}", f"{lat_m:.2f}" if lat_m is not None else ""])
    writer.writerow(["DEV_STD", "", "", f"{calm_s:.4f}", f"{frenetic_s:.4f}", f"{delta_s:.4f}",
                      f"{tpr_s:.4f}", f"{fpr_s:.4f}", f"{lat_s:.2f}" if lat_s is not None else ""])

print(f"\nTabella salvata in: {csv_path}")