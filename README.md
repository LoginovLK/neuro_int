# neuro_int — курсор по указательному пальцу

Поднимаешь ☝ — курсор на ПК едет туда, куда указывает палец. Опускаешь (или
убираешь руку) — курсор замирает. Всё. Остальной функционал (жесты, клики,
клавиши, docker-стек, веб-UI) удалён.

## Запуск

```bash
make run        # окно камеры + управление курсором (q — выход)
make headless   # без окна
make smoke      # 12с самопроверка камеры/GPU/детекта (мышь не трогает)
make test       # юнит-тесты
```

Venv `.venv-edge` уже на месте (onnxruntime-directml, opencv, pynput, numpy).

## Настройки

`config.json` в корне. Сенса меняется на лету клавишами `+` / `-` в окне
(шаг 0.2, сохраняется сразу):

| ключ | дефолт | что делает |
|---|---|---|
| `sensitivity` | 1.0 | размер активной зоны кадра: больше = резче курсор |
| `min_cutoff` / `beta` | 1.2 / 0.01 | One Euro: плавность против дрожи |
| `camera_index` | 0 | какая вебка |
| `mirror` | true | зеркало кадра |
| `debug_window` | true | окно предпросмотра |
| `tracking` | true | ROI руки из ландмарков прошлого кадра (кость не съезжает) |
| `clahe` | true | выравнивание контраста (тёмная комната) |
| `palm_score_threshold` / `landmark_score_threshold` | 0.4 / 0.25 | пороги детекта |

Активная зона = центральная часть кадра (при сенсе 1.0 — средняя половина по
ширине и высоте). Палец в центре зоны → курсор в центре экрана; палец в углу
зоны → курсор в углу экрана. Палец вне зоны — курсор прижат к краю экрана.

## Как это работает (4 слоя)

1. **Захват** (`camera.py`) — `CAP_DSHOW` + MJPG + выделенный drain-поток,
   который непрерывно осушает камеру и отдаёт только свежайший кадр
   (с номером — цикл не обрабатывает дубликаты). Драйвер DShow держит
   внутреннюю очередь: без drain-потока из неё читаются устаревшие кадры
   и копится лаг (те самые 2-3с).
2. **Детект** (`hands.py`, `tracking.py`) — PINTO-пайплайн как ONNX:
   palm-детектор с NMS внутри графа (192×192) + landmark-модель
   (224×224, 21 точка) через `onnxruntime` DirectML на GPU. Архитектура
   **track-priority**: пока рука в треке, источник — трекуемый ROI
   (ландмарки прошлого кадра + экстраполяция скорости), palm работает
   разведчиком (дубликаты игнорирует, перехватывает только ДРУГУЮ руку,
   лучше на 0.15); разрывы (смаз/резкий свайп) мостятся palm-кандидатом
   той же руки без сброса фильтров. Кропы ограничены (пер-кадр + окно
   1.5с + потолок), статичные не-руки (полка, лицо в покое) уходят в
   карантин. Кадр и кроп прогоняются через CLAHE.
3. **Кадры → команда** (`pose.py`, `mapping.py`) — палец считается указательным,
   когда он прямой (tip дальше от запястья, чем PIP, и изгиб в PIP < 50°),
   а средний/безымянный/мизинец согнуты. Ладонь или кулак — не считаются.
   Подтверждение 3 кадра подряд (debounce). Позиция кончика (landmark 8)
   маппится из активной зоны в экран абсолютно + One Euro сглаживание.
4. **Команда → ПК** (`cursor.py`) — `pynput` (SetCursorPos под капотом),
   процесс DPI-aware, чтобы координаты совпадали с физическими пикселями.

## Почему удалено всё остальное

Стек v1 (brain-контейнер, FSM жестов, WebSocket, веб-UI) работал, но курсор
не двигался по причине, которую не удалось локализовать за сутки. v2 — один
локальный процесс без сети, без контейнеров, без промежуточных слоёв: меньше
поверхностей — меньше мест, где сломаться. Proven-куски (drain-reader,
ONNX-декод с фиксами, One Euro) перенесены дословно.

## Models (downloaded into `models/`, gitignored)

| file | source | used by |
|---|---|---|
| `yolox_l.onnx` (216MB) | [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose) | person detector (DWPose backend) |
| `dw-ll_ucoco_384.onnx` (134MB) | [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose) | RTMPose wholebody, 133 keypoints (DWPose backend) |
| `face_detection_yunet_2023mar.onnx` (232KB) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | face gate (legacy backend) |
| `hand_landmarker.task` (7.8MB) | [MediaPipe hand landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) | **primary backend** (official Google pipeline) |
| `yolox_l.onnx` / `dw-ll_ucoco_384.onnx` are optional research artifacts (DWPose experiment, unused at runtime) | | |
| `palm_detection_full_inf_post_192x192.onnx`, `hand_landmark_sparse_Nx3x224x224.onnx` | [PINTO0309 hand-gesture-recognition](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) | fallback backend (no mediapipe package) |

The **primary backend is the official MediaPipe** hand landmarker.
The ONNX palm+landmark pair is the fallback when the mediapipe package
is missing. The DWPose experiment files are unused at runtime.
Historically the DWPose backend activated automatically when `yolox_l.onnx` and
`dw-ll_ucoco_384.onnx` are present; otherwise the legacy palm+landmark
pipeline runs.
