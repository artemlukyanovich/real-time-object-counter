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

## Интеграция в основной pipeline (`src/main.py`)

Новые модули **не встроены** в основной цикл по умолчанию — они готовы к подключению.
Минимальный пример интеграции в `ObjectCounterApp._initialize_components()`:

```python
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.object_memory import ObjectMemory
from src.reid import ReIDManager

# В _initialize_components():
self.cropper = ObjectCropper(padding=8)
self.embedder = ObjectEmbedder(device=self.config.get("detector.device", "cpu"))
self.memory = ObjectMemory(
    similarity_threshold=0.75,
    max_missing_frames=90,
)
self.reid_manager = ReIDManager(self.cropper, self.embedder, self.memory)
```

И в `run()`, после `tracker.update()`:

```python
detections, tracked_objects = self.tracker.update(frame)

# Новая строка:
track_to_object_id = self.reid_manager.update(frame, tracked_objects, frame_idx)
```

`track_to_object_id` содержит `Dict[track_id, object_id]` — его можно передавать в renderer
для отображения постоянных ID вместо временных трекерных.

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
| `crop(frame, bbox, track_id, frame_idx)` | Вырезает один crop; возвращает `np.ndarray` BGR |
| `crop_all(frame, tracked_objects, frame_idx)` | Batch: `Dict[track_id, crop]` |

Возвращает пустой массив `shape=(0,0,3)` если bbox выходит за границу кадра.

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
