# Запуск проекта

---

## Установка

### Основная среда (object-counter)

#### Conda (рекомендуется)

```bash
conda create -n object-counter python=3.10
conda activate object-counter
pip install -r requirements.txt
```

#### pip (без conda)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Отдельная среда для аннотирования (annotations)

Label Studio и labelImg требуют отдельного окружения из-за конфликтов зависимостей.

#### Conda

```bash
conda create -n annotations python=3.10
conda activate annotations
pip install -r requirements-annotations.txt
```

#### pip

```bash
python -m venv .venv-annotations
source .venv-annotations/bin/activate   # Linux/macOS
# .venv-annotations\Scripts\activate    # Windows

pip install -r requirements-annotations.txt
```

**Основные зависимости:**
- `ultralytics` — YOLOv8 + автоматическая загрузка весов
- `opencv-python` — захват видео и отрисовка
- `torch` — инференс (CPU или CUDA)
- `pyyaml` — парсинг конфигурации

При первом запуске `yolov8n.pt` скачивается автоматически (~6 МБ). Либо положить файл в корень проекта вручную.

---

## Аннотирование (Label Studio + labelImg)

После установки в окружении `annotations`:

### Label Studio

```bash
conda activate annotations  # или source .venv-annotations/bin/activate
label-studio
```

Откроется веб-интерфейс на `http://localhost:8080`. Используется для централизованного управления проектом аннотирования.

### labelImg

```bash
conda activate annotations
labelImg
```

Открывается GUI для быстрого аннотирования отдельных изображений. Используется для создания YOLO-формата датасетов.

---

## Запуск

### Веб-камера (индекс 0)

```bash
python -m src.main --source 0
```

### Видеофайл

```bash
python -m src.main --source path/to/video.mp4
```

### С указанием конфига

```bash
python -m src.main --config configs/default.yaml --source 0
```

### Видеофайл с кастомным конфигом

```bash
python -m src.main --config configs/default.yaml --source path/to/video.mp4
```

---

## CLI аргументы

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `--source` | int или str | из конфига (`video.source`) | Источник видео: `0` — камера, путь — файл |
| `--config` | str | `configs/default.yaml` | Путь к YAML-конфигурации |

`--source` перекрывает значение `video.source` из конфига.

---

## Управление во время работы

| Клавиша | Действие |
|---------|----------|
| `q` | Выход |
| `ESC` | Выход |

---

## Конфигурация через config.yaml

Все основные параметры задаются в `configs/default.yaml`:

```yaml
detector:
  model: "yolov8n.pt"          # Сменить на yolov8s.pt для большей точности
  confidence_threshold: 0.5
  device: "cuda"               # "cpu" если нет GPU

display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
```

Подробно — в `docs/config.md`.

---

## Выходные данные

По завершении работы в stdout выводится сводка:

```
=== Final Metrics ===
FPS: 28.4
Total frames: 1250
Avg detection time: 12.3 ms
Avg tracking time: 1.1 ms
```

Также итоговые счётчики по классам (если включён Counter).

---

## Типичные проблемы

**Камера не открывается:**
```
VideoSource: failed to open source 0
```
→ Проверить индекс камеры. Попробовать `--source 1` или `--source 2`.

**CUDA недоступна, падает с ошибкой:**
→ В конфиге установить `detector.device: "cpu"`. Детектор автоматически fallback-ается, но явное указание надёжнее.

**Слабый FPS:**
→ Снизить разрешение (`video.frame_width: 640`, `video.frame_height: 480`), сменить модель на `yolov8n.pt`, отключить лишние слои отрисовки (`show_detections: false`).

**Модель не найдена:**
```
FileNotFoundError: yolov8n.pt
```
→ Ultralytics скачивает модель автоматически при первом запуске. При отсутствии интернета — скопировать `.pt` файл в корень проекта вручную.
