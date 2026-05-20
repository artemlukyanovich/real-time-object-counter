# Конфигурация

## Файлы конфигурации

| Файл | Назначение |
|------|-----------|
| `configs/default.yaml` | Основной конфиг по умолчанию |
| `configs/experiments/*.yaml` | Конфиги экспериментов (та же структура, что у `default.yaml`) |
| `configs/embeddings/default.yaml` | Параметры CLIP-эмбеддингов и памяти ReID (см. [Секция `reid`](#секция-reid)) |
| `configs/embeddings/experiments/*.yaml` | Эксперименты с моделями эмбеддингов |

Загружается классом `Config` (`src/config.py`). Доступ к параметрам — через dot-notation: `config.get("detector.confidence_threshold")`.

**Запуск с кастомным конфигом:**
```bash
python -m src.main --config configs/experiments/my_experiment.yaml
```

---

## Структура файла

```yaml
video:
  source: 0
  fps: null
  fallback_fps: 30
  frame_width: 1280
  frame_height: 720

detector:
  model: "models/yolov8n.pt"
  confidence_threshold: 0.5
  device: "cuda"
  allowed_classes: null

tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.5
  track_low_threshold: 0.1
  matching_cost_threshold: 0.8
  lost_track_buffer: null
  auto_lost_track_buffer_seconds: 3.0
  fuse_score: true          # BoT-SORT only
  gmc_method: "sparseOptFlow"  # BoT-SORT only
  reid_weights: "osnet_x0_25_market.pt"  # botsort_reid only
  proximity_threshold: 0.5    # botsort_reid only
  appearance_threshold: 0.25  # botsort_reid only

counter:
  enable: true
  count_zone: null
  crossing_lines: null

reid:
  enabled: false
  embeddings_config: "configs/embeddings/default.yaml"
  update_interval: 3
  min_track_age: 1

output:
  save_video: false
  output_dir: "outputs"
  video_name: "output.mp4"
  fps: 30

display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
  font_size: 1.0
  show_object_ids: true   # только при reid.enabled: true
  show_reid_stats: true   # только при reid.enabled: true
```

---

## Секция `video`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `source` | int / str | `0` | Источник: `0` — первая камера, `1` — вторая, путь к файлу — видео |
| `fps` | int / null | `null` | `null` — читать FPS из источника; явное значение переопределяет его |
| `fallback_fps` | int | `30` | FPS по умолчанию, если источник сообщает `0` или ничего |
| `frame_width` | int | `1280` | Ширина кадра после ресайза (пиксели) |
| `frame_height` | int | `720` | Высота кадра после ресайза (пиксели) |

**Влияние на производительность:**
- Снижение разрешения до 640×480 может удвоить FPS при CPU-инференсе
- Увеличение разрешения улучшает детекцию мелких объектов, но увеличивает время инференса и ресайза

---

## Секция `detector`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `model` | str | `"yolov8n.pt"` | Имя или путь к файлу весов YOLOv8 |
| `confidence_threshold` | float | `0.5` | Минимальная уверенность детекции [0.0–1.0] |
| `device` | str | `"cuda"` | Устройство инференса: `"cuda"` или `"cpu"` |
| `allowed_classes` | list / null | `null` | Список классов для детекции/трекинга. `null` = все классы. Пример: `["person"]` или `["person", "car", "truck"]` |

**Фильтрация классов (`allowed_classes`):**

Передаётся напрямую в Ultralytics как `classes=[id, ...]` — фильтрация происходит на уровне модели до трекера, поэтому не влияет на скорость трекинга. Имена классов соответствуют меткам датасета COCO (для стандартных моделей YOLOv8/YOLO11).

```yaml
# Считать только людей:
detector:
  allowed_classes: ["person"]

# Считать транспорт:
detector:
  allowed_classes: ["car", "truck", "bus", "motorcycle"]
```

Если указанный класс не найден в модели — выводится предупреждение со списком доступных классов, и этот класс игнорируется.

**Выбор модели:**

Ultralytics автоматически скачивает веса при первом запуске, если файл не найден локально.

| Модель | Скорость (GPU) | Точность | Примечание |
|--------|---------------|---------|-----------|
| `yolov8n.pt` | ~5–10 мс | Базовая | Рекомендуется для реального времени |
| `yolov8s.pt` | ~10–20 мс | Средняя | Компромисс скорость/точность |
| `yolov8m.pt` | ~20–40 мс | Высокая | Требует мощного GPU |
| `yolov8l.pt` | ~30–60 мс | Очень высокая | Для серверного инференса |
| `yolov8x.pt` | ~50–100 мс | Максимальная | Самая точная модель YOLOv8 |
| `yolo11n.pt` | ~4–8 мс | Базовая | Более новая архитектура, быстрее v8n |
| `yolo11s.pt` | ~8–15 мс | Средняя | Новая архитектура, компромисс |
| `yolo11m.pt` | ~15–30 мс | Высокая | Новая архитектура, высокая точность |

> Модели `yolo11*` (YOLO11) — актуальное поколение от Ultralytics; при прочих равных предпочтительнее `yolov8*`.

**Влияние `confidence_threshold`:**
- `0.3–0.4` — больше детекций, больше ложных срабатываний → засоряет трекер
- `0.5` — баланс (дефолт)
- `0.7+` — только уверенные детекции; возможны пропуски при частичном перекрытии объектов

**Влияние `device`:**
- `"cuda"` — инференс на GPU (~5–15 мс для yolov8n)
- `"cpu"` — инференс на CPU (~50–200 мс), FPS падает до 5–10

---

## Секция `tracker`

Все настройки трекера берутся из выбранного конфига (`default.yaml` или `configs/experiments/*.yaml`). При запуске `tracker.py` генерирует YAML для Ultralytics в `.runtime/trackers/` (папка в `.gitignore`) — редактировать эти файлы не нужно.

### Общие параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `algorithm` | str | `"bytetrack"` | Алгоритм трекинга: `"bytetrack"`, `"botsort"` или `"botsort_reid"` |
| `track_activation_threshold` | float | `0.25` | Минимальная уверенность для активации нового трека |
| `track_low_threshold` | float | `0.1` | Минимальная уверенность для второго этапа сопоставления (низкоконфидентные детекции) |
| `matching_cost_threshold` | float | `0.8` | Порог стоимости IoU для сопоставления детекций с треками |
| `lost_track_buffer` | int / null | `null` | Кадров до удаления потерянного трека; `null` = авторасчёт |
| `auto_lost_track_buffer_seconds` | float | `3.0` | Секунды для авторасчёта буфера: `round(fps × seconds)` |

**Маппинг на поля Ultralytics YAML** (генерируется автоматически в `.runtime/trackers/`):

| Параметр конфига | Поле в Ultralytics YAML |
|-----------------|-------------------------|
| `track_activation_threshold` | `track_high_thresh`, `new_track_thresh` |
| `track_low_threshold` | `track_low_thresh` |
| `lost_track_buffer` | `track_buffer` |
| `matching_cost_threshold` | `match_thresh` |

**Влияние `track_activation_threshold`:**
- Малое значение (0.1–0.2): новые треки создаются даже при слабой уверенности → больше ложных треков
- Большое значение (0.5+): только уверенные детекции активируют трек → пропуски при частичном перекрытии

**Влияние `lost_track_buffer`:**
- Малое значение (10–20 кадров): треки удаляются быстро при кратких окклюзиях → частая смена ID → неточный подсчёт
- Большое значение (50–100 кадров): треки сохраняются при длительных окклюзиях → стабильный ID
- При `null` буфер рассчитывается автоматически: `round(fps × auto_lost_track_buffer_seconds)`

**Влияние `matching_cost_threshold`:**
- Высокое значение (0.8+): принимается сопоставление с низким IoU → объект сохраняет свой ID после кратковременного перекрытия (например, прохода мимо фонарного столба)
- Низкое значение (0.4–0.5): для сопоставления требуется высокое перекрытие → кратковременная окклюзия приводит к смене ID и созданию нового трека

### ByteTrack (`algorithm: "bytetrack"`)

ByteTrack использует двухэтапное сопоставление: сначала по высококонфидентным детекциям, затем по низкоконфидентным для «подхвата» уже существующих треков.

Все параметры управляются через общие настройки выше — дополнительных параметров нет.

### BoT-SORT (`algorithm: "botsort"`)

BoT-SORT расширяет ByteTrack двумя компонентами: **Global Motion Compensation (GMC)** — коррекция позиций треков при движении камеры — и опциональным **Re-ID** для повторной идентификации объектов по внешнему виду.

Дополнительные параметры (сверх общих):

| Параметр конфига | По умолчанию | Описание |
|-----------------|-------------|----------|
| `fuse_score` | `true` | Учитывать уверенность детекции в матрице стоимости IoU |
| `gmc_method` | `"sparseOptFlow"` | Метод GMC: `sparseOptFlow` / `orb` / `ecc` / `none` |

**Когда использовать BoT-SORT вместо ByteTrack:**
- Видео снято с движущейся камеры (дрон, PTZ) → GMC стабилизирует треки
- При статичной камере ByteTrack быстрее и достаточно точен

### BoT-SORT с Re-ID (`algorithm: "botsort_reid"`)

Тот же BoT-SORT, но с включённой моделью Re-ID (`with_reid: true`). Позволяет восстанавливать ID объекта по визуальному сходству даже после длительного исчезновения из кадра.

Дополнительные параметры по сравнению с `botsort`:

| Параметр конфига | По умолчанию | Описание |
|-----------------|-------------|----------|
| `reid_weights` | `"osnet_x0_25_market.pt"` | Веса Re-ID модели (скачиваются автоматически) |
| `proximity_threshold` | `0.5` | Порог IoU, ниже которого задействуются признаки Re-ID |
| `appearance_threshold` | `0.25` | Порог косинусного расстояния для сопоставления Re-ID |

**Доступные Re-ID модели:**

| Модель | Размер | Датасет | Примечание |
|--------|--------|---------|-----------|
| `osnet_x0_25_market.pt` | ~1 МБ | Market-1501 | По умолчанию; минимальная нагрузка на FPS |
| `osnet_x0_5_market.pt` | ~3 МБ | Market-1501 | Баланс скорость/точность |
| `osnet_x1_0_market.pt` | ~11 МБ | Market-1501 | Высокая точность Re-ID |
| `osnet_x0_25_msmt17.pt` | ~1 МБ | MSMT17 | Более разнообразный датасет |
| `osnet_x1_0_msmt17.pt` | ~11 МБ | MSMT17 | Максимальная точность Re-ID |

> Все модели скачиваются Ultralytics автоматически при первом запуске. Поменять модель можно через параметр `reid_weights` в конфиге.

**Влияние на производительность:**
- `osnet_x0_25_*` добавляет ~2–5 мс на кадр на GPU; на CPU — ~10–30 мс
- Если FPS критичен — используйте `"botsort"` без Re-ID

**Когда использовать `botsort_reid`:**
- Объекты часто надолго выходят за границы кадра и возвращаются
- Требуется сохранение ID при длительных окклюзиях
- Есть запас по вычислительным ресурсам

---

## Секция `counter`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `enable` | bool | `true` | Включить/выключить подсчёт |
| `count_zone` | list / null | `null` | Полигон зоны подсчёта |
| `crossing_lines` | list / null | `null` | Линии для подсчёта пересечений IN/OUT |

**Формат `count_zone`:**
```yaml
count_zone:
  - [100, 200]   # x1, y1
  - [500, 200]   # x2, y2
  - [500, 400]   # x3, y3
  - [100, 400]   # x4, y4
```
Задаёт прямоугольник или произвольный полигон. Объекты считаются только при нахождении их центроида внутри зоны. `null` — считать все объекты в кадре.

**Формат `crossing_lines`:**
```yaml
crossing_lines:
  - name: "Centre"
    points: [[640, 0], [640, 720]]
```
Направление **in** — объект пересёк линию слева направо (относительно вектора от `points[0]` к `points[1]`). Направление **out** — противоположное.

---

## Секция `reid`

CLIP-based Re-Identification pipeline — назначает каждому объекту постоянный `object_id`, который сохраняется при потере трека, окклюзии и повторном появлении. Работает поверх стандартного трекера.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `enabled` | bool | `false` | Включить CLIP ReID pipeline |
| `embeddings_config` | str | `"configs/embeddings/default.yaml"` | Путь к конфигу эмбеддингов (модель, память, cropper) |
| `update_interval` | int | `3` | Запускать embedding-пайплайн каждые N кадров. `1` = каждый кадр |
| `min_track_age` | int | `1` | Минимум кадров трека до регистрации нового объекта в памяти. `1` = мгновенно |

**`update_interval`** — снижает нагрузку на GPU/CPU. На кадрах между обновлениями переиспользуются последние известные `object_id`. При 30 FPS значение `3` даёт ~10 embedding-проходов в секунду — достаточно для стабильного ReID.

**`min_track_age`** — фильтрует «мелькающие» объекты (ложные детекции на 1–5 кадров). Считается в реальных кадрах по `frame_idx` **независимо от `update_interval`**. Трек, совпавший с уже известным объектом по эмбеддингу, сопоставляется мгновенно независимо от этого порога.

| `min_track_age` | Поведение |
|---|---|
| `1` | Отключено — каждый трек сразу регистрируется |
| `8` | При 30 FPS ≈ 0.25 сек — отсекает мгновенные ложные детекции |
| `30` | 1 секунда — только устойчивые объекты |

**Конфиг эмбеддингов** (`configs/embeddings/default.yaml`) содержит параметры модели и памяти:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `embedder.model_name` | `"ViT-B-32"` | OpenCLIP модель. Варианты: `ViT-B-32`, `ViT-B-16`, `ViT-L-14`, `ViT-H-14` |
| `embedder.pretrained` | `"laion2b_s34b_b79k"` | Тег весов (должен соответствовать модели) |
| `embedder.device` | `"cuda"` | Устройство для эмбеддингов: `"cuda"` или `"cpu"` |
| `cropper.padding` | `8` | Отступ в пикселях вокруг bbox при вырезании crop |
| `cropper.save_crops` | `false` | Сохранять crops в `outputs/crops/` для визуальной проверки |
| `memory.similarity_threshold` | `0.75` | Порог cosine similarity для Re-ID совпадения [0.0–1.0] |
| `memory.max_missing_frames` | `90` | Кадров без детекции до деактивации объекта (~3 сек при 30 FPS) |
| `memory.max_embeddings_per_object` | `5` | Rolling buffer эмбеддингов на объект |

Подробнее — в [docs/embeddings_reid.md](embeddings_reid.md).

---

## Секция `output`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `save_video` | bool | `false` | Сохранить аннотированное видео |
| `output_dir` | str | `"outputs"` | Директория для выходных файлов |
| `video_name` | str | `"output.mp4"` | Имя выходного файла |
| `fps` | int | `30` | FPS выходного видео |

---

## Секция `display`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `show_detections` | bool | `true` | Отображать bbox сырых детекций |
| `show_tracking_ids` | bool | `true` | Отображать bbox треков с ID |
| `show_counts` | bool | `true` | Отображать панель счётчиков |
| `font_size` | float | `1.0` | Масштаб шрифта аннотаций |
| `show_object_ids` | bool | `true` | Показывать постоянный `object_id` на bbox вместо `track_id` (только при `reid.enabled: true`) |
| `show_reid_stats` | bool | `true` | Панель "ReID unique / active" в правом верхнем углу (только при `reid.enabled: true`) |

При `show_object_ids: true` формат метки на bounding box: `#N class [tM]`, где `N` — постоянный `object_id`, `class` — класс, `tM` — текущий `track_id` трекера. Цвет bbox определяется по `object_id` и не меняется при смене `track_id`.

---

## Рекомендуемые конфигурации

### Максимальная производительность (edge/слабое железо)
```yaml
video:
  frame_width: 640
  frame_height: 480
detector:
  model: "models/yolov8n.pt"
  confidence_threshold: 0.5
  device: "cpu"
tracker:
  algorithm: "bytetrack"
display:
  show_detections: false
  show_tracking_ids: true
  show_counts: true
```

### Максимальная точность (GPU-сервер, статичная камера)
```yaml
video:
  frame_width: 1920
  frame_height: 1080
detector:
  model: "models/yolov8s.pt"
  confidence_threshold: 0.4
  device: "cuda"
tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.3
  auto_lost_track_buffer_seconds: 5.0
  matching_cost_threshold: 0.7
```

### Движущаяся камера (дрон / PTZ)
```yaml
tracker:
  algorithm: "botsort"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 4.0
  matching_cost_threshold: 0.7
```
BoT-SORT с GMC (`sparseOptFlow`) компенсирует смещение кадра и снижает количество потерянных треков при движении камеры.

### Устойчивый трекинг с Re-ID (окклюзии, повторные появления)
```yaml
tracker:
  algorithm: "botsort_reid"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 5.0
  matching_cost_threshold: 0.7
```
BoT-SORT с Re-ID восстанавливает ID объекта по внешнему виду даже после длительного отсутствия в кадре. ReID-модель скачается автоматически (`osnet_x0_25_market.pt`).

### Устойчивый подсчёт с CLIP ReID (постоянные ID, фильтр мелькашей)
```yaml
reid:
  enabled: true
  embeddings_config: "configs/embeddings/default.yaml"
  update_interval: 3   # каждые 3 кадра при 30 FPS
  min_track_age: 8     # ≈0.25 сек — не считать объекты, мелькнувшие на 1–7 кадров
display:
  show_object_ids: true
  show_reid_stats: true
```
CLIP ReID назначает объекту постоянный `#N`, не меняющийся при потере трека. `min_track_age: 8` исключает ложные детекции, не успевшие устояться.

### Отладка (видны все детекции и треки)
```yaml
detector:
  confidence_threshold: 0.3
tracker:
  track_activation_threshold: 0.2
display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
```
