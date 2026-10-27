import csv
import os
import sys
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTION_THRESHOLD = 0.4  # soglia che rileva l'anomalia

FRENETIC_PHASE_DURATION_DEFAULT = 120.0

MODEL_NAMES = ["isolation_forest", "one_class_svm", "local_outlier_factor", "mahalanobis"]
MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "local_outlier_factor": "Local Outlier Factor",
    "mahalanobis": "Mahalanobis (Elliptic Envelope)",
}


all_session_files = sorted(glob.glob(os.path.join(LOG_DIR, "session_*.csv")))
if not all_session_files:
    print("Nessuna sessione trovata in logs/. Gioca almeno una sessione prima di lanciare l'analisi.")
    exit()

print("=" * 60)
print(f"SESSIONI DISPONIBILI ({len(all_session_files)})")
print("=" * 60)
for path in all_session_files:
    stamp = os.path.basename(path).replace("session_", "").replace(".csv", "")
    print(f"  {stamp}")

if len(sys.argv) > 1:
    chosen_stamp = sys.argv[1]
    SESSION_LOG_PATH = os.path.join(LOG_DIR, f"session_{chosen_stamp}.csv")
    EVENTS_LOG_PATH = os.path.join(LOG_DIR, f"events_{chosen_stamp}.csv")
else:
    SESSION_LOG_PATH = all_session_files[-1]
    chosen_stamp = os.path.basename(SESSION_LOG_PATH).replace("session_", "").replace(".csv", "")
    EVENTS_LOG_PATH = os.path.join(LOG_DIR, f"events_{chosen_stamp}.csv")

print(f"\nAnalizzo in dettaglio: {chosen_stamp}")
print("(per analizzarne un'altra: python analyze_session.py <timestamp_sessione>)\n")



print("=" * 60)
print("RIEPILOGO COMPARATIVO — media punteggio ufficiale per ciascuna sessione")
print("=" * 60)
for path in all_session_files:
    stamp = os.path.basename(path).replace("session_", "").replace(".csv", "")
    scores_this_session = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] == "active":
                scores_this_session.append(float(row["official_score"]))
    if scores_this_session:
        avg = sum(scores_this_session) / len(scores_this_session)
        print(f"  {stamp}: {len(scores_this_session)} finestre attive, punteggio medio = {avg:.3f}, "
              f"picco massimo = {max(scores_this_session):.3f}")
    else:
        print(f"  {stamp}: nessuna finestra attiva (sessione troppo breve o solo calibrazione)")
print()



windows = []
with open(SESSION_LOG_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["phase"] != "active":
            continue
        entry = {
            "t": float(row["session_time_s"]),
            "official_score": float(row["official_score"]),
            "top_feature": row["top_contributing_feature"],
            "top_zscore": float(row["top_feature_zscore"]) if row["top_feature_zscore"] else 0.0,
            "baseline_used": row.get("baseline_used", ""),
            "game_state": row.get("game_state", ""),
        }
        for name in MODEL_NAMES:
            val = row.get(f"score_{name}", "")
            entry[name] = float(val) if val not in ("", None) else None
        windows.append(entry)

events = []
with open(EVENTS_LOG_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        events.append({
            "t": float(row["session_time_s"]),
            "name": row["event_name"],
        })

print(f"Finestre attive caricate: {len(windows)}")
print(f"Eventi caricati: {len(events)}")

if not windows:
    print("Nessuna finestra 'active' trovata nel log. Gioca una sessione completa (dopo la calibrazione) prima di analizzare.")
    exit()



from collections import Counter as _Counter
baseline_counts = _Counter(w["baseline_used"] for w in windows)
state_counts = _Counter(w["game_state"] for w in windows)

print("\n" + "=" * 60)
print("BASELINE USATA (globale di riserva vs dedicata per stato)")
print("=" * 60)
for label, count in baseline_counts.most_common():
    print(f"  {label or '(vuoto)'}: {count} finestre")

print("\nDistribuzione per stato di gioco rilevato:")
for label, count in state_counts.most_common():
    print(f"  {label or '(vuoto)'}: {count} finestre")



official_scores = [w["official_score"] for w in windows]
print("\n" + "=" * 60)
print("PUNTEGGIO UFFICIALE (quello che muove la barra in Godot)")
print("=" * 60)
print(f"media={sum(official_scores)/len(official_scores):.3f}, "
      f"min={min(official_scores):.3f}, max={max(official_scores):.3f}")
sopra_01 = sum(1 for s in official_scores if s > 0.1)
print(f"Finestre con punteggio > 0.1 (contribuiscono a far salire la barra): "
      f"{sopra_01}/{len(official_scores)} ({sopra_01/len(official_scores):.0%})")

times = [w["t"] for w in windows]



out_of_order = [(i, times[i], times[i + 1]) for i in range(len(times) - 1) if times[i + 1] < times[i]]
if out_of_order:
    print(f"\nATTENZIONE: {len(out_of_order)} righe NON sono in ordine cronologico nel file!")
    print("Questo spiega la 'diagonale' nei grafici: il disegno collega i punti nell'ordine")
    print("in cui appaiono nel file, non nell'ordine del tempo reale. Esempi:")
    for i, t1, t2 in out_of_order[:5]:
        print(f"  riga {i}: t={t1:.1f}s seguita da riga {i+1}: t={t2:.1f}s (va indietro nel tempo)")
    print("Possibile causa: due processi del server erano attivi insieme (es. non hai chiuso")
    print("del tutto uno 'uvicorn' precedente prima di lanciarne un altro), scrivendo entrambi")
    print("nello stesso file senza coordinarsi. Controlla di avere UN SOLO terminale con")
    print("uvicorn attivo prima del prossimo test.")
else:
    print("\nOrdine cronologico: OK, tutte le righe sono in sequenza corretta.")



from collections import Counter as _TimeCounter
time_counts = _TimeCounter(times)
duplicate_times = {t: c for t, c in time_counts.items() if c > 1}
if duplicate_times:
    print(f"\nATTENZIONE: {len(duplicate_times)} istanti compaiono PIU' VOLTE nel file (stesso timestamp):")
    for t, c in list(duplicate_times.items())[:5]:
        vals = [w["official_score"] for w in windows if w["t"] == t]
        print(f"  t={t:.1f}s compare {c} volte, con punteggi: {vals}")
    print("Anche questo puo' spiegare lo zig-zag: due scritture per lo stesso istante,")
    print("probabile causa: due processi server attivi in parallelo sullo stesso file di log.")



gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
big_gaps = [(times[i], times[i] + g) for i, g in enumerate(gaps) if g > 10.0]
if big_gaps:
    print(f"\nATTENZIONE: trovati {len(big_gaps)} buchi temporali > 10s tra finestre consecutive")
    print("(probabile causa: nessun input rilevato in quell'intervallo, la finestra non e' stata inviata)")
    for start, end in big_gaps:
        print(f"  buco tra t={start:.1f}s e t={end:.1f}s (durata: {end - start:.1f}s)")
    print("I grafici ora mostrano un pallino per ogni dato reale: i tratti tra due")
    print("pallini molto distanti sono questi buchi, non un trend continuo.")



fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(times, [w["official_score"] for w in windows], color="crimson", linewidth=1.5,
        marker="o", markersize=4, label="Stress stimato (Isolation Forest)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Tempo di sessione (secondi)")
ax.set_ylabel("Punteggio di anomalia (0-1)")
ax.set_title("Stress stimato nel tempo, con eventi di gioco annotati")

for ev in events:
    ax.axvline(ev["t"], color="gray", linestyle="--", alpha=0.6)
    ax.text(ev["t"], 1.02, ev["name"], rotation=90, fontsize=8, va="bottom", ha="center", color="dimgray")

ax.legend(loc="upper left")
fig.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "stress_nel_tempo.png")
fig.savefig(fig1_path, dpi=150)
print(f"\nSalvato: {fig1_path}")



def rolling_mean(values, window=5):
    """Media mobile semplice: serve a far emergere l'andamento sotto il rumore."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


ROLLING_WINDOW = 5
PANEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig2, axes2 = plt.subplots(len(MODEL_NAMES), 1, figsize=(12, 10), sharex=True, sharey=True)
if len(MODEL_NAMES) == 1:
    axes2 = [axes2]

for idx, (ax, name) in enumerate(zip(axes2, MODEL_NAMES)):
    color = PANEL_COLORS[idx % len(PANEL_COLORS)]
    ys = [w[name] for w in windows]
    xs_plot = [t for t, y in zip(times, ys) if y is not None]
    ys_plot = [y for y in ys if y is not None]

    if not ys_plot:
        ax.text(0.5, 0.5, f"{MODEL_LABELS[name]}: nessun dato valido",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(-0.05, 1.05)
        continue


    
    ax.plot(xs_plot, ys_plot, alpha=0.25, linewidth=0.8, color=color)
    ax.plot(xs_plot, rolling_mean(ys_plot, ROLLING_WINDOW), linewidth=2.2, color=color)

    media = sum(ys_plot) / len(ys_plot)
    ax.axhline(media, linestyle=":", linewidth=1.0, color="black", alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Punteggio")
    ax.set_title(f"{MODEL_LABELS[name]}  —  media = {media:.3f}",
                 fontsize=11, loc="left", fontweight="bold")



    for e in events:
        if e["name"] in ("calm_start", "frenetic_start"):
            ax.axvline(e["t"], linestyle="--", linewidth=1.0, color="gray", alpha=0.7)

axes2[-1].set_xlabel("Tempo di sessione (secondi)")



legend_handles = [
    Line2D([0], [0], color="gray", alpha=0.4, linewidth=0.8,
           label="Punteggio di ogni finestra (5s)"),
    Line2D([0], [0], color="gray", linewidth=2.2,
           label=f"Media mobile ({ROLLING_WINDOW} finestre = 25s)"),
    Line2D([0], [0], color="black", linestyle=":", linewidth=1.0, alpha=0.6,
           label="Media dell'algoritmo sull'intera sessione"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
           label="Marker di fase (calm_start / frenetic_start)"),
]
fig2.legend(handles=legend_handles, loc="upper right", fontsize=8,
            bbox_to_anchor=(0.995, 0.995), framealpha=0.9)

fig2.suptitle("Confronto tra algoritmi di anomaly detection", fontsize=13, x=0.06, ha="left")
fig2.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "confronto_algoritmi.png")
fig2.savefig(fig2_path, dpi=150)
print(f"Salvato: {fig2_path}")



print("\n" + "=" * 60)
print("STATISTICHE DI CONFRONTO TRA ALGORITMI")
print("=" * 60)
for name in MODEL_NAMES:
    vals = [w[name] for w in windows if w[name] is not None]
    if not vals:
        print(f"{MODEL_LABELS[name]}: MAI addestrato con successo in questa sessione "
              f"(probabile problema numerico con questa dimensionalita'/n. campioni)")
        continue
    mean = sum(vals) / len(vals)
    print(f"{MODEL_LABELS[name]}: media={mean:.3f}, min={min(vals):.3f}, max={max(vals):.3f}, "
          f"n_finestre_valide={len(vals)}/{len(windows)}")




from collections import Counter
top_features_during_high_stress = [
    w["top_feature"] for w in windows if w["official_score"] > DETECTION_THRESHOLD and w["top_feature"]
]
if top_features_during_high_stress:
    counts = Counter(top_features_during_high_stress)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    names_sorted = [n for n, _ in counts.most_common()]
    counts_sorted = [c for _, c in counts.most_common()]
    ax3.barh(names_sorted, counts_sorted, color="steelblue")
    ax3.set_xlabel("N. di finestre in cui questa feature ha causato il picco")
    ax3.set_title(f"Feature piu' spesso responsabili di stress rilevato (score > {DETECTION_THRESHOLD})")
    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "feature_responsabili.png")
    fig3.savefig(fig3_path, dpi=150)
    print(f"Salvato: {fig3_path}")
else:
    print("\nNessuna finestra ha superato la soglia di rilevamento: nessun grafico di spiegabilita' generato.")



calm_start = next((e["t"] for e in events if e["name"] == "calm_start"), None)
frenetic_start = next((e["t"] for e in events if e["name"] == "frenetic_start"), None)
game_start = next((e["t"] for e in events if e["name"] == "game_start"), None)

if calm_start is not None and frenetic_start is not None:
    print("\n" + "=" * 60)
    print("PROTOCOLLO DI VALUTAZIONE: FASE CALMA vs FASE FRENETICA")
    print("=" * 60)



    frenetic_end = game_start if game_start is not None else (frenetic_start + FRENETIC_PHASE_DURATION_DEFAULT)

    calm_phase = [w for w in windows if calm_start <= w["t"] < frenetic_start]
    frenetic_phase = [w for w in windows if frenetic_start <= w["t"] < frenetic_end]

    if calm_phase and frenetic_phase:
        calm_scores = [w["official_score"] for w in calm_phase]
        frenetic_scores = [w["official_score"] for w in frenetic_phase]

        calm_mean = sum(calm_scores) / len(calm_scores)
        frenetic_mean = sum(frenetic_scores) / len(frenetic_scores)

        false_positives = sum(1 for s in calm_scores if s > DETECTION_THRESHOLD)
        true_positives = sum(1 for s in frenetic_scores if s > DETECTION_THRESHOLD)

        fpr = false_positives / len(calm_scores)
        tpr = true_positives / len(frenetic_scores)


        first_detection = next(
            (w["t"] for w in frenetic_phase if w["official_score"] > DETECTION_THRESHOLD), None
        )
        latency = (first_detection - frenetic_start) if first_detection is not None else None

        print(f"Finestre fase calma:     {len(calm_phase)}  (media score: {calm_mean:.3f})")
        print(f"Finestre fase frenetica: {len(frenetic_phase)}  (media score: {frenetic_mean:.3f})")
        print(f"\nFalse Positive Rate (fase calma, score > {DETECTION_THRESHOLD}):     {fpr:.1%}")
        print(f"True Positive Rate (fase frenetica, score > {DETECTION_THRESHOLD}):    {tpr:.1%}")
        if latency is not None:
            print(f"Latenza di rilevamento: {latency:.1f} secondi dall'inizio della fase frenetica")
        else:
            print("Latenza di rilevamento: NESSUNA finestra della fase frenetica ha superato la soglia")

        print("\nQueste cifre sono direttamente citabili nel capitolo di valutazione della tesi.")


        print("\n" + "-" * 60)
        print("CONFRONTO TRA SOGLIE DI RILEVAMENTO (nessuna sessione nuova necessaria)")
        print("-" * 60)
        print(f"{'Soglia':>8} | {'FPR':>8} | {'TPR':>8}")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            fpr_t = sum(1 for s in calm_scores if s > thr) / len(calm_scores)
            tpr_t = sum(1 for s in frenetic_scores if s > thr) / len(frenetic_scores)
            marker = "  <- attuale" if thr == DETECTION_THRESHOLD else ""
            print(f"{thr:>8.1f} | {fpr_t:>7.1%} | {tpr_t:>7.1%}{marker}")
        print("\nScegli la soglia che minimizza l'FPR mantenendo un TPR ancora accettabile")
        print("per il tuo caso d'uso, poi aggiorna DETECTION_THRESHOLD in cima allo script.")


        import datetime
        report_path = os.path.join(OUTPUT_DIR, f"report_valutazione_{chosen_stamp}.txt")
        with open(report_path, "w", encoding="utf-8") as rep:
            rep.write("REPORT DI VALUTAZIONE — PROTOCOLLO CALMO/FRENETICO\n")
            rep.write("=" * 60 + "\n")
            rep.write(f"Sessione analizzata: {chosen_stamp}\n")
            rep.write(f"Report generato il: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rep.write(f"Soglia di rilevamento usata: {DETECTION_THRESHOLD}\n\n")
            rep.write(f"Finestre fase calma:     {len(calm_phase)}  (media score: {calm_mean:.3f})\n")
            rep.write(f"Finestre fase frenetica: {len(frenetic_phase)}  (media score: {frenetic_mean:.3f})\n\n")
            rep.write(f"False Positive Rate (fase calma):     {fpr:.1%}\n")
            rep.write(f"True Positive Rate (fase frenetica):  {tpr:.1%}\n")
            if latency is not None:
                rep.write(f"Latenza di rilevamento: {latency:.1f} secondi dall'inizio della fase frenetica\n")
            else:
                rep.write("Latenza di rilevamento: nessuna finestra della fase frenetica ha superato la soglia\n")
            rep.write(f"\nNota metodologica: campione di {len(frenetic_phase)} finestre in fase frenetica ")
            rep.write("e {} in fase calma. Con campioni cosi' ridotti, le percentuali vanno trattate ".format(len(calm_phase)))
            rep.write("come indicative, non come stime statisticamente definitive.\n")
        print(f"\nReport salvato in: {report_path}")
    else:
        print("Una delle due fasi non ha finestre sufficienti: assicurati che la sessione sia durata")
        print("abbastanza a lungo dopo ciascun marker (calm_start / frenetic_start).")
else:
    print("\nNessun marker 'calm_start'/'frenetic_start' trovato nel log eventi.")
    print("Per generare il report di valutazione, durante una sessione di test chiama:")
    print('  StressMonitor.log_game_event("calm_start")     (a inizio fase calma)')
    print('  StressMonitor.log_game_event("frenetic_start")  (dopo ~2 minuti, a inizio fase frenetica)')
    print("poi rilancia questo script.")