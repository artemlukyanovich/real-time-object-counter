# Конфигурация (config.yaml)

Файл конфигурации: `configs/default.yaml`

Загружается классом `Config` (`src/config.py`). Доступ к параметрам — через dot-notation: `config.get("detector.confidence_threshold")`.

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
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  device: "cuda"

tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.25
  lost_track_buffer: null
  auto_lost_track_buffer_seconds: 3.0
  minimum_matching_threshold: 0.8

counter:
  enable: true
  count_zone: null
  crossing_lines: null

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

Секция содержит два уровня настроек:

1. **Общие параметры** в `default.yaml` — задают алгоритм и три ключевых значения, одинаковых для ByteTrack и BoT-SORT.
2. **Шаблоны алгоритмов** в `configs/trackers/bytetrack.yaml` и `configs/trackers/botsort.yaml` — содержат полный набор параметров конкретного алгоритма. Общие параметры из п.1 перезаписывают соответствующие поля шаблона при запуске.

### Общие параметры (`default.yaml`)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `algorithm` | str | `"bytetrack"` | Алгоритм трекинга: `"bytetrack"`, `"botsort"` или `"botsort_reid"` |
| `track_activation_threshold` | float | `0.25` | Минимальная уверенность для активации нового трека |
| `lost_track_buffer` | int / null | `null` | Кадров до удаления потерянного трека; `null` = авторасчёт |
| `auto_lost_track_buffer_seconds` | float | `3.0` | Секунды для авторасчёта буфера: `round(fps × seconds)` |
| `minimum_matching_threshold` | float | `0.8` | Порог IoU для сопоставления детекций с треками |

**Маппинг на поля шаблонов:**

| Параметр `default.yaml` | Поле в шаблоне трекера |
|-------------------------|------------------------|
| `track_activation_threshold` | `track_high_thresh`, `new_track_thresh` |
| `lost_track_buffer` | `track_buffer` |
| `minimum_matching_threshold` | `match_thresh` |

**Влияние `track_activation_threshold`:**
- Малое значение (0.1–0.2): новые треки создаются даже при слабой уверенности → больше ложных треков
- Большое значение (0.5+): только уверенные детекции активируют трек → пропуски при частичном перекрытии

**Влияние `lost_track_buffer`:**
- Малое значение (10–20 кадров): треки удаляются быстро при кратких окклюзиях → частая смена ID → неточный подсчёт
- Большое значение (50–100 кадров): треки сохраняются при длительных окклюзиях → стабильный ID
- При `null` буфер рассчитывается автоматически: `round(fps × auto_lost_track_buffer_seconds)`

**Влияние `minimum_matching_threshold`:**
- Высокое значение (0.8+): требуется сильное пересечение bbox → меньше ложных сопоставлений, больше разрывов треков при быстром движении
- Низкое значение (0.4–0.5): треки сохраняются при большом смещении, но возможны ошибочные сопоставления разных объектов

### ByteTrack (`configs/trackers/bytetrack.yaml`)

ByteTrack использует двухэтапное сопоставление: сначала по высококонфидентным детекциям, затем по низкоконфидентным для «подхвата» уже существующих треков.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `track_high_thresh` | `0.25` | Порог для высококонфидентных детекций (1-й этап) |
| `track_low_thresh` | `0.1` | Порог для низкоконфидентных детекций (2-й этап) |
| `new_track_thresh` | `0.25` | Минимальная уверенность для инициализации нового трека |
| `track_buffer` | `30` | Кадров до удаления потерянного трека |
| `match_thresh` | `0.8` | Порог IoU для сопоставления |

> `track_high_thresh` и `new_track_thresh` перезаписываются из `track_activation_threshold`.
> `track_buffer` перезаписывается из `lost_track_buffer`.
> `match_thresh` перезаписывается из `minimum_matching_threshold`.

### BoT-SORT (`configs/trackers/botsort.yaml`)

BoT-SORT расширяет ByteTrack двумя компонентами: **Global Motion Compensation (GMC)** — коррекция позиций треков при движении камеры — и опциональным **Re-ID** для повторной идентификации объектов по внешнему виду.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `track_high_thresh` | `0.5` | Порог для высококонфидентных детекций (1-й этап) |
| `track_low_thresh` | `0.1` | Порог для низкоконфидентных детекций (2-й этап) |
| `new_track_thresh` | `0.5` | Минимальная уверенность для инициализации нового трека |
| `track_buffer` | `30` | Кадров до удаления потерянного трека |
| `match_thresh` | `0.8` | Порог IoU для сопоставления |
| `fuse_score` | `true` | Учитывать уверенность детекции в матрице стоимости IoU |
| `gmc_method` | `sparseOptFlow` | Метод GMC: `sparseOptFlow` / `orb` / `ecc` / `none` |
| `proximity_thresh` | `0.5` | Порог IoU, ниже которого задействуются признаки Re-ID |
| `appearance_thresh` | `0.25` | Порог косинусного расстояния для сопоставления Re-ID |
| `with_reid` | `false` | Включить Re-ID модель (требует отдельную модель весов) |

> Параметры `track_high_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh` перезаписываются так же, как в ByteTrack.

**Когда использовать BoT-SORT вместо ByteTrack:**
- Видео снято с движущейся камеры (дрон, PTZ) → GMC стабилизирует треки
- При статичной камере ByteTrack быстрее и достаточно точен

### BoT-SORT с Re-ID (`configs/trackers/botsort_reid.yaml`)

Тот же BoT-SORT, но с включённой моделью Re-ID (`with_reid: true`). Позволяет восстанавливать ID объекта по визуальному сходству даже после длительного исчезновения из кадра.

Дополнительные параметры по сравнению с `botsort`:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `with_reid` | `true` | Включить Re-ID |
| `reid_weights` | `osnet_x0_25_market.pt` | Веса Re-ID модели (скачиваются автоматически) |

**Доступные Re-ID модели:**

| Модель | Размер | Датасет | Примечание |
|--------|--------|---------|-----------|
| `osnet_x0_25_market.pt` | ~1 МБ | Market-1501 | По умолчанию; минимальная нагрузка на FPS |
| `osnet_x0_5_market.pt` | ~3 МБ | Market-1501 | Баланс скорость/точность |
| `osnet_x1_0_market.pt` | ~11 МБ | Market-1501 | Высокая точность Re-ID |
| `osnet_x0_25_msmt17.pt` | ~1 МБ | MSMT17 | Более разнообразный датасет |
| `osnet_x1_0_msmt17.pt` | ~11 МБ | MSMT17 | Максимальная точность Re-ID |

> Все модели скачиваются Ultralytics автоматически при первом запуске. Указать другую модель можно прямо в `botsort_reid.yaml` → поле `reid_weights`.

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

---

## Рекомендуемые конфигурации

### Максимальная производительность (edge/слабое железо)
```yaml
video:
  frame_width: 640
  frame_height: 480
detector:
  model: "yolov8n.pt"
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
  model: "yolov8s.pt"
  confidence_threshold: 0.4
  device: "cuda"
tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.3
  auto_lost_track_buffer_seconds: 5.0
  minimum_matching_threshold: 0.7
```

### Движущаяся камера (дрон / PTZ)
```yaml
tracker:
  algorithm: "botsort"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 4.0
  minimum_matching_threshold: 0.7
```
BoT-SORT с GMC (`sparseOptFlow`) компенсирует смещение кадра и снижает количество потерянных треков при движении камеры.

### Устойчивый трекинг с Re-ID (окклюзии, повторные появления)
```yaml
tracker:
  algorithm: "botsort_reid"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 5.0
  minimum_matching_threshold: 0.7
```
BoT-SORT с Re-ID восстанавливает ID объекта по внешнему виду даже после длительного отсутствия в кадре. ReID-модель скачается автоматически (`osnet_x0_25_market.pt`).

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
