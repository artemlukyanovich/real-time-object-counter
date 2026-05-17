# Обзор проекта: Real-time Object Counter

## Назначение

Система компьютерного зрения для детекции, трекинга и подсчёта объектов в реальном времени. Принимает видеопоток (веб-камера или видеофайл), распознаёт объекты с помощью YOLOv8/YOLO11, присваивает каждому уникальный ID и подсчитывает их — по зоне подсчёта и/или по пересечению линий.

Целевые сценарии: мониторинг трафика, подсчёт людей, аналитика на борту БПЛА с edge-устройствами.

**Ограничения среды:** ноутбук с RTX 3050 Ti (~4 ГБ VRAM), цель — 20–30 FPS в реальном времени.

---

## Архитектура системы

Линейный pipeline покадровой обработки:

```
VideoSource → UltralyticsTracker → Counter → Renderer → Display
```

Детекция и трекинг объединены в один inference pass через `model.track()` (Ultralytics API). Каждый компонент — независимый класс с чётко определённым интерфейсом. Оркестрацию выполняет `ObjectCounterApp` в `src/main.py`. Производительность отслеживается через отдельный модуль `metrics`.

---

## Основные компоненты

### video_source — `src/video_source.py`

Абстракция над источником видео: унифицирует работу с веб-камерой (`cv2.VideoCapture(0)`) и видеофайлом. Выполняет захват кадра, ресайз до заданного разрешения и подсчёт кадров. Поддерживает контекстный менеджер для корректного освобождения ресурсов.

### detector — `src/detector.py`

Загружает модель YOLOv8/YOLO11 (Ultralytics) и хранит её инстанс. Модель используется совместно с трекером — детекция происходит внутри `UltralyticsTracker.update()`, поэтому отдельного вызова инференса в pipeline нет.

### tracker — `src/tracker.py`

`UltralyticsTracker` — обёртка над Ultralytics tracking API (`model.track()`). Объединяет детекцию и трекинг в одном inference pass.

**Поддерживаемые алгоритмы:**
- **ByteTrack** — двухэтапное IoU-сопоставление (высоко- и низкоконфидентные детекции). Рекомендован для статичной камеры.
- **BoT-SORT** — расширяет ByteTrack Global Motion Compensation (GMC) для стабилизации треков при движении камеры.
- **BoT-SORT + Re-ID** (`botsort_reid`) — добавляет восстановление ID по визуальному сходству (OSNet) при длительных окклюзиях.

При инициализации `UltralyticsTracker` генерирует YAML-конфиг трекера в `.runtime/trackers/` и передаёт его путь в `model.track()`. Поддерживает фильтрацию по классам (`allowed_classes`): имена преобразуются в ID классов модели и передаются напрямую в inference.

**Формат выхода `update()`:**
- `detections`: `List[((x1,y1,x2,y2), class_name, confidence)]`
- `tracked_objects`: `Dict[track_id, ((x1,y1,x2,y2), class_name)]`

### counter — `src/counter.py`

Подсчитывает объекты двумя независимыми методами (можно комбинировать):

- **Зональный подсчёт** (`count_zone`): объект учитывается ровно один раз, когда его центроид впервые попадает внутрь полигона. `null` — считать все объекты в кадре. Хранит счётчики по классам и суммарный счётчик.
- **Подсчёт пересечений линий** (`crossing_lines`): при каждом пересечении линии центроидом трека увеличивает счётчик направления `in` или `out` (определяется по смене стороны относительно вектора линии). Поддерживает несколько именованных линий.

Состояние сторон треков (`_prev_sides`) автоматически очищается при исчезновении трека.

### renderer — `src/renderer.py`

Отрисовывает аннотации поверх кадра средствами OpenCV: рамки детекций с уверенностью, рамки треков с ID, панель счётчиков зоны, панель счётчиков линий, визуализацию линий пересечения, индикатор FPS. Цвета треков детерминированы по `track_id % 256` — один объект всегда одного цвета в течение всей сессии.

### metrics — `src/metrics.py`

Измеряет производительность pipeline: FPS (скользящее окно 30 кадров), среднее время инференса детектора и трекера в миллисекундах. Не влияет на логику обработки — только мониторинг.

---

## Вспомогательные компоненты

- **`src/config.py`** — загрузка YAML-конфигурации с поддержкой dot-notation (`video.frame_width`). Метод `get_raw()` возвращает значение без fallback (нужен для различения `null` и отсутствующего ключа).
- **`src/utils.py`** — геометрические утилиты: центроид bbox, евклидово расстояние, IoU, проверка точки в полигоне (`ray casting`), определение стороны точки относительно линии (`point_side_of_line`).

---

## Утилиты и скрипты

### extract_frames — `scripts/extract_frames.py`

Извлекает каждый N-й кадр из видеофайла и сохраняет как JPG. Поддерживает batch-обработку нескольких видео в одну директорию с продолжением нумерации. Используется для подготовки датасетов.

```bash
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 30
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 15 --dry-run
```

Подробнее — в `docs/extract_frames.md`.

### cors_http_server — `scripts/cors_http_server.py`

Поднимает локальный HTTP-сервер (порт 9000) с CORS-заголовками для доступа Label Studio к изображениям из файловой системы.

```bash
python -m scripts.cors_http_server
```

### convert_yolo_predictions_to_label_studio — `scripts/convert_yolo_predictions_to_label_studio.py`

Конвертирует предсказания YOLO (`.txt` файлы в формате `class cx cy w h`) в JSON-формат Label Studio для импорта и ревью разметки.

```bash
python -m scripts.convert_yolo_predictions_to_label_studio \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  outputs/predictions_all.json
```

### review_yolo_labels — `scripts/review_yolo_labels.py`

Интерактивный просмотрщик разметки на базе OpenCV. Позволяет быстро просмотреть изображения с YOLO bbox и принять решение по каждому: одобрить, удалить разметку или отметить для ручной проверки. Логирует результаты в CSV.

```bash
python -m scripts.review_yolo_labels \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  --output-review-dir outputs/label_review \
  --dry-run
```

Клавиши: `n`/`p` — следующий/предыдущий, `k` — OK, `d` — удалить разметку, `m` — отметить для ревью, `q` — выход.

### split_yolo_dataset — `scripts/split_yolo_dataset.py`

Разделяет экспорт из Label Studio на тренировочный и валидационный наборы в формате YOLO. Генерирует `data.yaml` для запуска `yolo detect train`.

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-export-dir \
  data/yolo_output/project-output-dir \
  --val-ratio 0.2
```

### Инструменты аннотирования (отдельная среда `annotations`)

**Label Studio** — веб-интерфейс для централизованного управления проектом аннотирования:
```bash
conda activate annotations
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
```

**labelImg** — GUI для быстрого аннотирования изображений в формате YOLO/Pascal VOC:
```bash
conda activate annotations
labelImg
```

Требуют отдельного окружения из-за конфликтов зависимостей. Инструкции по установке — в `docs/how_to_run.md`.

Полный workflow подготовки датасета и обучения модели — в `docs/dataset_preparation.md`.

---

## Конфигурация

| Файл | Назначение |
|------|-----------|
| `configs/default.yaml` | Основной конфиг по умолчанию |
| `configs/experiments/*.yaml` | Пресеты для экспериментов (та же структура) |
| `.runtime/trackers/*.yaml` | Генерируются автоматически при запуске, в `.gitignore` |

Запуск с кастомным конфигом:
```bash
python -m src.main --config configs/experiments/01_bytetrack_fast.yaml
python -m src.main --source /path/to/video.mp4
```

Доступные пресеты:
- `01_bytetrack_fast.yaml` — максимальная скорость
- `02_bytetrack_stable.yaml` — стабильный трекинг
- `03_botsort_balanced.yaml` — движущаяся камера
- `04_botsort_reid_light.yaml` — Re-ID с минимальной нагрузкой
- `05_bytetrack_accuracy.yaml` — максимальная точность

---

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Детекция | YOLOv8 / YOLO11 (Ultralytics) |
| Трекинг | ByteTrack / BoT-SORT (Ultralytics built-in) |
| Re-ID | OSNet (torchreid, через Ultralytics) |
| Инференс | PyTorch + CUDA |
| Захват видео | OpenCV |
| Конфигурация | YAML |
| Язык | Python 3.10+ |

---

## Архитектурные замечания

**Текущие ограничения:**
- Синхронный pipeline: inference блокирует основной поток.
- `output.save_video` предусмотрен в конфиге, но запись видео не подключена к pipeline.
- `calculate_iou` в `utils.py` реализован, но нигде не используется.

**Вектор развития:**
- Асинхронный захват кадров (отдельный поток для `VideoSource`)
- Запись аннотированного видео (`output.save_video`)
- Экспорт модели в ONNX / TensorRT для оптимизации инференса
- Интеграция с симулятором дрона (PX4 / AirSim)
