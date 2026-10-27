extends Node

const SERVER_URL := "http://127.0.0.1:8000/predict"
const RESET_URL := "http://127.0.0.1:8000/reset"
const LOG_EVENT_URL := "http://127.0.0.1:8000/log_event"
const HEALTH_URL := "http://127.0.0.1:8000/health"
# Server impacchettato: cartella "server" accanto all'eseguibile del gioco (solo versione esportata)
const BUNDLED_SERVER_RELATIVE_PATH := "server/apatia_server.exe"
const HEALTH_MAX_ATTEMPTS := 40
const WINDOW_SECONDS := 5.0


const TRACKED_BUTTONS := [
	JOY_BUTTON_A, JOY_BUTTON_B, JOY_BUTTON_X, JOY_BUTTON_Y,
	JOY_BUTTON_LEFT_SHOULDER, JOY_BUTTON_RIGHT_SHOULDER,
	JOY_BUTTON_DPAD_UP, JOY_BUTTON_DPAD_DOWN, JOY_BUTTON_DPAD_LEFT, JOY_BUTTON_DPAD_RIGHT,
]

const TRACKED_AXES := [
	JOY_AXIS_LEFT_X, JOY_AXIS_LEFT_Y, JOY_AXIS_RIGHT_X, JOY_AXIS_RIGHT_Y,
	JOY_AXIS_TRIGGER_LEFT, JOY_AXIS_TRIGGER_RIGHT,
]


const TRACKED_KEYS := [
	KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE, KEY_SHIFT, KEY_E, MOUSE_BUTTON_LEFT,
]


const MULT_BASE := 0.5
const MULT_K := 0.375

signal stress_multiplier_updated(multiplier: float, raw_value: float)


var current_game_state := "unknown"


const STATE_GROUPS := {
	"attack": "combat",
	"take_damage": "combat",
	"death": "combat",
	"jump": "traversal",
	"fall": "traversal",
	"dash": "traversal",
	"run": "traversal",
	"crouch": "traversal",
	"idle": "idle",
}


func set_game_state(raw_state_name: String) -> void:
	var key := raw_state_name.to_lower()
	var category: String = STATE_GROUPS.get(key, "unknown")

	if category == "combat":
		var boss_present := not get_tree().get_nodes_in_group("Boss").is_empty()
		category = "combat_boss" if boss_present else "combat_regular"

	current_game_state = category

var _window_start_time := 0.0
var _buffered_events := []
var _http := HTTPRequest.new()

var _server_pid := -1
var _server_ready := true
var _health_attempts := 0
var _health_http := HTTPRequest.new()

var _last_axis_value := {}

var _input_used_this_window := false


var inputs_inverted : bool = false


func _ready() -> void:
	add_child(_http)
	_http.request_completed.connect(_on_request_completed)
	_window_start_time = Time.get_ticks_msec() / 1000.0
	set_process_input(true)
	set_process(true)
	_start_bundled_server()


func _input(event: InputEvent) -> void:
	var t := (Time.get_ticks_msec() / 1000.0) - _window_start_time

	if event is InputEventJoypadButton:
		_input_used_this_window = true
		if event.button_index in TRACKED_BUTTONS:
			_buffered_events.append({
				"type": "key",
				"event": str(event.button_index),
				"state": 1.0 if event.pressed else 0.0,
				"t": t,
			})

	elif event is InputEventKey and not event.echo:
		if event.keycode in TRACKED_KEYS:
			_input_used_this_window = true
			_buffered_events.append({
				"type": "key",
				"event": "kb_" + str(event.keycode),
				"state": 1.0 if event.pressed else 0.0,
				"t": t,
			})

	elif event is InputEventJoypadMotion:
		if event.axis in TRACKED_AXES:
			# evita di floodare col rumore dello stick fermo: soglia minima di variazione
			var prev = _last_axis_value.get(event.axis, 0.0)
			if abs(event.axis_value - prev) > 0.02:
				_input_used_this_window = true
				_buffered_events.append({
					"type": "absolute",
					"event": str(event.axis),
					"state": event.axis_value,
					"t": t,
				})
				_last_axis_value[event.axis] = event.axis_value



func _process(_delta: float) -> void:
	var elapsed := (Time.get_ticks_msec() / 1000.0) - _window_start_time
	if elapsed >= WINDOW_SECONDS:
		_send_window()


func _send_window() -> void:
	var events_copy := _buffered_events.duplicate(true)
	_buffered_events.clear()
	_input_used_this_window = false
	_window_start_time = Time.get_ticks_msec() / 1000.0

	if events_copy.is_empty():
		return

	# il server impacchettato impiega qualche secondo ad avviarsi: finche' non risponde non si inviano finestre
	if not _server_ready:
		return

	var body := JSON.stringify({"events": events_copy, "game_state": current_game_state})
	var headers := ["Content-Type: application/json"]
	_http.request(SERVER_URL, headers, HTTPClient.METHOD_POST, body)


func _on_request_completed(_result, response_code, _headers, body) -> void:
	if response_code != 200:
		push_warning("StressMonitor: richiesta al server fallita (%s)" % response_code)
		return

	var json := JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK:
		push_warning("StressMonitor: risposta JSON non valida")
		return

	var data = json.get_data()
	var stress_raw: float = data.get("stress_raw", 1.0)

	var stress_mult := MULT_BASE + MULT_K * (stress_raw - 1.0)

	stress_multiplier_updated.emit(stress_mult, stress_raw)


func _start_bundled_server() -> void:
	# Dall'editor il server si avvia a mano (uvicorn), come prima.
	if OS.has_feature("editor"):
		return
	var server_exe := OS.get_executable_path().get_base_dir().path_join(BUNDLED_SERVER_RELATIVE_PATH)
	if not FileAccess.file_exists(server_exe):
		push_warning("StressMonitor: server non trovato (%s): il gioco parte senza adattivita'" % server_exe)
		return
	_server_pid = OS.create_process(server_exe, [], false)
	if _server_pid <= 0:
		push_warning("StressMonitor: impossibile avviare il server")
		_server_pid = -1
		return
	_server_ready = false
	_health_attempts = 0
	add_child(_health_http)
	_health_http.request_completed.connect(_on_health_completed)
	_health_http.request(HEALTH_URL)


func _on_health_completed(_result, response_code, _headers, _body) -> void:
	if response_code == 200:
		_server_ready = true
		return
	_health_attempts += 1
	if _health_attempts >= HEALTH_MAX_ATTEMPTS:
		push_warning("StressMonitor: il server non risponde: il gioco prosegue senza adattivita'")
		_server_ready = true
		return
	await get_tree().create_timer(1.0).timeout
	_health_http.request(HEALTH_URL)


func _stop_bundled_server() -> void:
	if _server_pid > 0:
		OS.kill(_server_pid)
		_server_pid = -1


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_stop_bundled_server()


func _exit_tree() -> void:
	_stop_bundled_server()


func reset_session() -> void:
	var http := HTTPRequest.new()
	add_child(http)
	http.request(RESET_URL, [], HTTPClient.METHOD_POST)
	_buffered_events.clear()
	_window_start_time = Time.get_ticks_msec() / 1000.0


func log_game_event(event_name: String, value: float = 0.0) -> void:
	var http := HTTPRequest.new()
	add_child(http)
	http.request_completed.connect(func(_r, _c, _h, _b): http.queue_free())
	var body := JSON.stringify({"event_name": event_name, "value": value})
	var headers := ["Content-Type: application/json"]
	http.request(LOG_EVENT_URL, headers, HTTPClient.METHOD_POST, body)


func set_input_inversion( active : bool ) -> void:
	if active == inputs_inverted:
		return
	_swap_input_actions( "left", "right" )
	inputs_inverted = active
	pass


func force_normal_controls() -> void:
	set_input_inversion( false )
	pass


func _swap_input_actions( action_a : String, action_b : String ) -> void:
	var events_a := InputMap.action_get_events( action_a )
	var events_b := InputMap.action_get_events( action_b )
	InputMap.action_erase_events( action_a )
	InputMap.action_erase_events( action_b )
	for e in events_b:
		InputMap.action_add_event( action_a, e )
	for e in events_a:
		InputMap.action_add_event( action_b, e )
	pass
