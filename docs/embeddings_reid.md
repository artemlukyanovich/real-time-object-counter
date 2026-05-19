# Embeddings и Re-Identification (Этапы 4–5)

---

## Обзор

Новый pipeline поверх существующего:

```
YOLO → Tracker → tracked_objects
                      │
                  ObjectCropper   ← crop из bbox
                      │
                  ObjectEmbedder  ← feature vector (OpenCLIP)
                      │
                  ObjectMemory    ← поиск по similarity
                      │
                  ReIDManager     → Dict[track_id, object_id]
```

**Ключевое отличие от обычного трекинга:**
- `track_id` — временный ID, выдаваемый ByteTrack/BoT-SORT. Сбрасывается при потере объекта.
- `object_id` — постоянный ID, выживающий пересечение кадра, окклюзию и смену `track_id`.

---

## Установка зависимостей

`open-clip-torch` уже включён в `requirements.txt`. При первом обращении к `ObjectEmbedder`
веса модели (`ViT-B-32`, ~350 МБ) скачиваются автоматически в кэш Torch Hub.

```bash
pip install -r requirements.txt
```

Для ускорения на GPU убедитесь, что установлена CUDA-версия torch и задайте `device: "cuda"`
в `configs/embeddings/default.yaml`.

---

## Быстрый тест: crop → embedding → similarity

Следующий скрипт проверяет работу всех новых модулей без видео-источника.

```python
import cv2
import numpy as np
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.similarity import cosine_similarity, find_best_match

# Синтетический кадр и два bbox
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
bbox_a = (100, 100, 300, 400)
bbox_b = (600, 200, 900, 500)

cropper = ObjectCropper(padding=8)
embedder = ObjectEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")

crop_a = cropper.crop(frame, bbox_a)
crop_b = cropper.crop(frame, bbox_b)

emb_a = embedder.embed(crop_a)
emb_b = embedder.embed(crop_b)

score = cosine_similarity(emb_a, emb_b)
print(f"Similarity between two random crops: {score:.4f}")  # ~0.8–0.95 для случайных патчей

# Тест find_best_match
gallery = np.stack([emb_a, emb_b])
matched_id, best_score = find_best_match(emb_a, gallery, object_ids=[1, 2], threshold=0.8)
print(f"Best match: object_id={matched_id}, score={best_score:.4f}")  # ожидается id=1, score≈1.0
```

---

## Быстрый тест: ReIDManager на видеофайле

```python
import cv2
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.object_memory import ObjectMemory
from src.reid import ReIDManager

cropper = ObjectCropper(padding=8, save_crops=True)          # сохраняет crops в outputs/crops/
embedder = ObjectEmbedder(device="cpu")
memory = ObjectMemory(similarity_threshold=0.75, max_missing_frames=90)
reid = ReIDManager(cropper, embedder, memory)

cap = cv2.VideoCapture("data/input/your_video.mp4")
frame_idx = 0

# Предположим, tracked_objects уже получен от UltralyticsTracker
# Здесь имитируем одним объектом
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # В реальном использовании: detections, tracked_objects = tracker.update(frame)
    fake_tracked = {42: ((200, 100, 400, 350), "person")}

    track_to_obj = reid.update(frame, fake_tracked, frame_idx)
    print(f"Frame {frame_idx}: track 42 → object_id {track_to_obj.get(42)}")

    frame_idx += 1

cap.release()
print(f"Total unique objects seen: {reid.total_object_count()}")
```

---

## Интеграционный тест на реальном видео

Smoke-тест с синтетическими данными проверяет, что код работает без ошибок, но ничего не говорит о качестве ReID на реальных объектах. Следующий шаг — запустить полный pipeline `YOLO → Tracker → Crop → Embedding → ReID` на настоящем видео и проанализировать распределение similarity-score.

### Запуск

```bash
python -m scripts.test_reid_integration \
    --source data/input/your_video.mp4 \
    --config configs/default.yaml \
    --max-frames 300 \
    --threshold 0.75
```

Аргументы:

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--source` | обязательный | Путь к видео или индекс веб-камеры (`0`) |
| `--config` | `configs/default.yaml` | Конфиг пайплайна (модель, трекер, разрешение) |
| `--max-frames` | `300` | Число кадров для обработки |
| `--threshold` | `0.75` | Порог cosine similarity для матчинга |

### Что скрипт делает

1. Открывает видео, прогоняет детектор + трекер на каждом кадре.
2. Для каждого трека вырезает crop, считает embedding, сравнивает с памятью.
3. Логирует каждое событие в консоль:
   - **`continuous`** — трек уже привязан к object_id, обновляем его.
   - **`RE-ID ✓`** — новый track_id совпал с известным объектом (настоящий re-id).
   - **`new`** — нет совпадения выше threshold, создаём новый object_id.
4. Сохраняет все crops в `outputs/crops/` для визуальной проверки.
5. Выводит итоговую таблицу распределения scores и подсказку по выбору threshold.

### Пример вывода

```
 frame  track_id  object_id  event         score
────────────────────────────────────────────────────────────────────────
     0         1          1  new           0.0000
     0         2          2  new           0.0000
     1         1          1  continuous    0.9821
     1         2          2  continuous    0.9743
    47         3          1  RE-ID ✓       0.8612   ← трек потерян, но объект узнан
    47         4          3  new           0.4231
...

Similarity score distributions
────────────────────────────────────────────────────────────────────────
  same object (continuous)        n= 284  min=0.8901  mean=0.9612  max=0.9981
  re-id match (new track_id)      n=   3  min=0.8401  score=0.8612  max=0.8901
  no match (new object)           n=   4  min=0.3120  mean=0.4105  max=0.5230

Suggested threshold range: 0.76 – 0.86  (current: 0.75)
```

### Что проверять в консоли

| Что смотреть | Хороший признак | Тревожный признак |
|---|---|---|
| `continuous` scores | `mean > 0.90` — объект стабильно узнаётся | `mean < 0.80` — модель нестабильна на этом типе объектов |
| Разрыв между `continuous` и `no match` mean | `> 0.20` — threshold легко выбрать | `< 0.10` — классы объектов плохо разделяются CLIP |
| Число `RE-ID ✓` событий | Появляются при потере/смене трека | `0` при длинном видео — объекты не теряются (OK) или threshold слишком высок |
| Число уникальных объектов | Соответствует реальному числу людей/машин в кадре | Сильно больше — false negatives (threshold высок); сильно меньше — false positives (threshold низок) |

### Что проверять в `outputs/crops/`

Откройте папку и убедитесь, что crops:
- содержат объект целиком, а не фон или край кадра;
- не пустые и не слишком маленькие (< 20×20 px);
- для одного `track_id` — одинаковый объект на всех кадрах;
- crops с одним `object_id` (суффикс `_objN`) действительно изображают один физический объект, даже если `track_id` разный — это подтверждает корректность re-id.

Если crops плохие — увеличьте `padding` в `configs/embeddings/default.yaml` или уменьшите `detector.confidence_threshold`.

### Подбор threshold

После запуска скрипт выводит подсказку:
```
Suggested threshold range: 0.76 – 0.86  (current: 0.75)
```

Формула: `mean(same_scores) - 0.4 * gap`, где `gap = mean(same) - mean(no_match)`.
Это точка примерно на 40% пути от "нет совпадения" до "совпадение". Проверьте значения на обоих концах диапазона:

```bash
# Строгий порог — меньше ложных re-id
python -m scripts.test_reid_integration --source ... --threshold 0.85

# Мягкий порог — больше re-id событий
python -m scripts.test_reid_integration --source ... --threshold 0.70
```

Занесите найденный threshold в `configs/embeddings/default.yaml`:
```yaml
memory:
  similarity_threshold: 0.82  # подобрано экспериментально
```

---

## Интеграция в основной pipeline (`src/main.py`)

ReID-pipeline **полностью интегрирован** в `ObjectCounterApp` и включается одной строкой в конфиге.

### Включение

В `configs/default.yaml` установите:

```yaml
reid:
  enabled: true
  embeddings_config: "configs/embeddings/default.yaml"  # параметры модели и памяти
  update_interval: 3
```

Параметры модели, cropper'а и памяти берутся из `configs/embeddings/default.yaml` — подбор порогов и устройства производится там.

### Что происходит при запуске

При `reid.enabled: true` приложение:

1. Загружает `configs/embeddings/default.yaml` и инициализирует `ObjectCropper`, `ObjectEmbedder`, `ObjectMemory` и `ReIDManager` (`_initialize_reid()`).
2. В каждом кадре после `tracker.update()` вызывает `reid_manager.update(frame, tracked_objects, frame_idx)`, получая `Dict[track_id, object_id]`.
3. Передаёт маппинг в `FrameRenderer` для отображения.

### Визуальная оценка

Включите соответствующие опции в `configs/default.yaml`:

```yaml
display:
  show_object_ids: true   # показывать OBJ ID вместо track_id на bounding box
  show_reid_stats: true   # панель "ReID unique / active" в правом верхнем углу
```

**Формат метки на bounding box:**

```
#N class [tM]
```

- `#N` — постоянный `object_id` (не меняется при потере и повторном обнаружении объекта)
- `class` — класс объекта
- `[tM]` — текущий временный `track_id` трекера

**Цвет bounding box** определяется по `object_id`, а не по `track_id`: один и тот же физический объект всегда выделен одним цветом, даже если трекер сменил ему `track_id`.

**Панель ReID stats** (правый верхний угол, под FPS):

```
ReID unique: 5
ReID active: 3
```

- `unique` — суммарное количество уникальных объектов за сессию
- `active` — объекты, активные в данный момент (не истёкшие по `max_missing_frames`)

### Как читать визуализацию

| Что видно на экране | Интерпретация |
|---|---|
| Цвет bounding box не меняется при кратковременном исчезновении | ReID корректно переопределил объект |
| `#N` остаётся прежним при смене `tM` | Успешный re-id: новый track_id сопоставлен со старым object_id |
| `unique` растёт быстрее реального числа объектов | threshold слишком высокий — объекты не узнаются |
| `unique` меньше реального числа объектов | threshold слишком низкий — разные объекты сливаются в один |

### Пример вывода при запуске с ReID

```
Tracker: bytetrack (Ultralytics) | fps=30, activation_threshold=0.5, ...
ReID: enabled | model=ViT-B-32 device=cpu threshold=0.75
Starting object counter. Press 'q' to exit.
```

---

## Конфигурация

Параметры находятся в `configs/embeddings/default.yaml`:

```yaml
embedder:
  model_name: "ViT-B-32"          # модель OpenCLIP
  pretrained: "laion2b_s34b_b79k" # веса
  device: "cpu"                   # "cuda" для GPU
  normalize: true                 # L2-нормализация (рекомендуется)

cropper:
  padding: 8          # пиксели отступа вокруг bbox
  save_crops: false   # сохранять crops в outputs/crops/

memory:
  similarity_threshold: 0.75   # порог для re-id совпадения
  max_missing_frames: 90       # ~3 сек при 30 FPS
  max_embeddings_per_object: 5 # rolling buffer

output:
  save_embeddings: false  # сохранять .npy в outputs/embeddings/
  save_reid_log: false    # сохранять лог событий в outputs/reid/
```

**Настройка порога:**

| `similarity_threshold` | Поведение |
|---|---|
| `0.90+` | Строгий — почти нет ложных совпадений, но объекты чаще считаются новыми |
| `0.75` | Баланс (рекомендуется для старта) |
| `0.60–` | Мягкий — больше re-id совпадений, выше риск склейки разных объектов |

---

## Справочник API

### `ObjectCropper` (`src/cropper.py`)

| Метод | Описание |
|---|---|
| `__init__(padding, save_crops, output_dir)` | Создаёт cropper; `padding` — отступ в пикселях |
| `crop(frame, bbox, track_id, frame_idx, object_id)` | Вырезает один crop; возвращает `np.ndarray` BGR |
| `crop_all(frame, tracked_objects, frame_idx)` | Batch: `Dict[track_id, crop]` |

Возвращает пустой массив `shape=(0,0,3)` если bbox выходит за границу кадра.

**Именование сохраняемых файлов:**

| Переданные параметры | Имя файла |
|---|---|
| `frame_idx=31, track_id=2` | `frame000031_id2.jpg` |
| `frame_idx=31, track_id=2, object_id=1` | `frame000031_id2_obj1.jpg` |
| только `track_id=2` | `id2.jpg` |

Суффикс `_obj{N}` добавляется только при явной передаче `object_id`. Это позволяет сопоставлять crops с разными `track_id` (временными) по общему `object_id` (постоянному): например, `frame000031_id2_obj1.jpg` и `frame000035_id5_obj1.jpg` — один и тот же физический объект.

**Если суффикс `_obj` отсутствует** (например `frame000031_id2.jpg`), это означает, что в момент сохранения crop `object_id` ещё не был известен. Такое происходит только при первом появлении трека: объект только что обнаружен, и ReIDManager ещё не успел сопоставить его с памятью. Начиная со следующего кадра тот же трек уже имеет `object_id`, и все последующие crops сохраняются с суффиксом.

---

### `ObjectEmbedder` (`src/embedder.py`)

| Метод | Описание |
|---|---|
| `__init__(model_name, pretrained, device, normalize)` | Загружает OpenCLIP модель |
| `embed(crop)` | Один crop → `np.ndarray (512,)` float32 |
| `embed_batch(crops)` | Список crops → `np.ndarray (N, 512)` float32 |

Размерность зависит от модели: `ViT-B-32` → 512, `ViT-L-14` → 768.

---

### `similarity.py`

| Функция | Описание |
|---|---|
| `cosine_similarity(a, b)` | Скалярное сходство двух векторов `[-1, 1]` |
| `cosine_similarity_batch(query, gallery)` | `query` против матрицы → `(N,)` float32 |
| `find_best_match(query, gallery, object_ids, threshold)` | → `(object_id \| None, score)` |

---

### `ObjectMemory` (`src/object_memory.py`)

| Метод | Описание |
|---|---|
| `add(class_name, bbox, embedding, track_id, frame_idx)` | Новый объект → возвращает `object_id` |
| `update(object_id, bbox, embedding, track_id, frame_idx)` | Обновить известный объект |
| `find_match(embedding, current_frame)` | → `(object_id \| None, score)` |
| `expire_old(current_frame)` | Деактивировать старые объекты → список деактивированных ID |
| `get(object_id)` | `ObjectRecord` по ID |
| `active_count()` | Количество активных объектов |
| `total_count()` | Всего объектов за сессию |

---

### `ReIDManager` (`src/reid.py`)

| Метод | Описание |
|---|---|
| `__init__(cropper, embedder, memory)` | Принимает готовые экземпляры модулей |
| `update(frame, tracked_objects, frame_idx)` | → `Dict[track_id, object_id]` |
| `get_object_id(track_id)` | Быстрый lookup без обработки кадра |
| `active_object_count()` | Активные объекты в памяти |
| `total_object_count()` | Всего уникальных объектов за сессию |

---

### `PipelineBenchmark` (`src/benchmark.py`)

```python
from src.benchmark import PipelineBenchmark

bench = PipelineBenchmark()

bench.start_frame()

bench.start("detection")
detections, tracks = tracker.update(frame)
bench.stop("detection")

bench.start("reid")
reid_result = reid.update(frame, tracks, frame_idx)
bench.stop("reid")

bench.end_frame()

# По окончании:
print(bench.summary())
# {'frames': 300, 'fps': 18.4, 'detection_mean_ms': 32.1, 'reid_mean_ms': 45.7, ...}

bench.save("run_001.json")  # → outputs/benchmarks/run_001.json
```

---

## Выходные папки

| Папка | Содержимое | Когда используется |
|---|---|---|
| `outputs/crops/` | JPEG-изображения вырезанных объектов | `ObjectCropper(save_crops=True)` |
| `outputs/embeddings/` | `.npy` файлы эмбеддингов | будущая опция `save_embeddings: true` |
| `outputs/reid/` | Логи re-id событий, видео с постоянными ID | Этап 5 |
| `outputs/benchmarks/` | JSON-репорты производительности | `PipelineBenchmark.save()` |

---

## Ограничения текущей реализации

- **In-memory only** — память объектов не сохраняется между запусками.
- **Один embedding на вызов** — `embed()` не батчится внутри `ReIDManager`. При большом числе объектов в кадре это узкое место. Оптимизация: использовать `embed_batch()` для всех crops за один GPU-проход.
- **Нет разрешения конфликтов** — если два трека претендуют на один object_id, побеждает первый. Более сложная логика — задача Этапа 5.
- **CLIP не обучен на конкретных объектах** — при работе с нестандартными классами (дроны, специфический транспорт) точность re-id будет ниже, чем у специализированной ReID-модели.
