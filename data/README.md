# Dati sperimentali

Questa cartella è predisposta per i dati della sperimentazione (10 partecipanti).
Per impostazione predefinita `python/logs/` è escluso da Git (vedi `.gitignore`).

## Formato dei log (generati dal server)

- `session_<AAAAMMGG_HHMMSS>.csv`: una riga per finestra da 5 s.
  Colonne: `timestamp, session_time_s, phase, official_score, top_contributing_feature,
  top_feature_zscore, baseline_used, game_state, score_isolation_forest, score_one_class_svm,
  score_local_outlier_factor, score_mahalanobis`.
  `phase` vale `calibrating`, `calibration_complete` oppure `active`.
- `events_<AAAAMMGG_HHMMSS>.csv`: eventi marcati manualmente (`timestamp, session_time_s,
  event_name, value`), tra cui `calm_start` e `frenetic_start`.

