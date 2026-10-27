# Parte Godot

Questa cartella contiene solo `StressMonitor.gd`, lo script (autoload) che collega il gioco al server:

- cattura l'input (pulsanti e assi del controller, più alcuni tasti della tastiera);
- ogni 5 secondi invia la finestra di eventi a `POST /predict` del server;
- riceve `stress_raw` ed emette il segnale `stress_multiplier_updated`;
- applica l'inversione dei comandi sinistra/destra del "mondo psichedelico";
- in una versione esportata può avviare un server impacchettato (`server/apatia_server.exe`); dall'editor non lo fa.

Il progetto Godot completo (scene, sprite, musica, livelli) **non è pubblicato** in questo repository:
sono incluse le parti che realizzano il contributo della tesi. Il gioco è stato sviluppato con Godot 4.6.
