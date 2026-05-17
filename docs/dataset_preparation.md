# Подготовка датасета и обучение модели

Описание полного цикла: от сырых видео до обученной YOLO-модели для детекции кастомных объектов.

---

## Обзор pipeline

```
Видео → Извлечение кадров → Ручная разметка (Label Studio)
     → Bootstrap-модель → Auto-labeling → Ревью разметки
     → Финальный датасет → Обучение финальной модели
```

---

## 1. Извлечение кадров из видео

`scripts/extract_frames.py` — сохраняет каждый N-й кадр как JPG.

```bash
python -m scripts.extract_frames data/raw_videos/taar_1.mp4 data/frames/taar --step 30
python -m scripts.extract_frames data/raw_videos/the_institute_3.mp4 data/frames/the_institute --step 30 --prefix the_institute

# Предпросмотр без сохранения
python -m scripts.extract_frames data/raw_videos/taar_1.mp4 data/frames/taar --step 30 --dry-run
```

Нумерация продолжается с последнего существующего файла — несколько видео можно обрабатывать в одну директорию без перезаписи.

Подробнее — в `docs/extract_frames.md`.

---

## 2. Ручная разметка в Label Studio

Label Studio требует отдельного окружения `annotations`.

### Запуск Label Studio

```bash
conda activate annotations
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
label-studio
```

Веб-интерфейс доступен на `http://localhost:8080`.

### CORS-сервер для локальных изображений

Label Studio не может напрямую читать файлы с диска из браузера. Запустить в отдельном терминале:

```bash
python -m scripts.cors_http_server
```

Сервер раздаёт содержимое корня проекта на `http://localhost:9000` с CORS-заголовками. Изображения указываются относительными URL вида `http://localhost:9000/data/frames/...`.

---

## 3. Bootstrap-модель для semi-automatic annotation

### Разделение датасета (bootstrap)

Экспортировать разметку из Label Studio в формате YOLO, затем разделить на train/val:

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-6-at-2026-05-15-05-52-0a94b509 \
  data/yolo_bootstrap/project-6-at-2026-05-15-05-52-0a94b509 \
  --val-ratio 0.2
```

Результат: директория с подпапками `train/` и `val/` и файлом `data.yaml`.

### Обучение bootstrap-модели

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/yolo_bootstrap/project-6-at-2026-05-15-05-52-0a94b509/data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=8 \
  project=custom_models/bootstrap \
  name=custom_bootstrap_1
```

Веса сохраняются в `custom_models/bootstrap/custom_bootstrap_1/weights/best.pt`.

---

## 4. Auto-labeling: генерация bbox для всего датасета

```bash
yolo detect predict \
  model=custom_models/bootstrap/custom_bootstrap_1/weights/best.pt \
  source=data/frames/all \
  conf=0.25 \
  save=True \
  save_txt=True \
  project=custom_models/bootstrap_predictions \
  name=custom_bootstrap_1_frames_all
```

Предсказания (`.txt`) сохраняются в `custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels/`.

---

## 5. Ревью auto-generated разметки

### Локальный просмотр

`scripts/review_yolo_labels.py` — интерактивный просмотрщик на OpenCV:

```bash
python -m scripts.review_yolo_labels \
  data/frames/all \
  custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels \
  --output-review-dir outputs/label_review \
  --dry-run
```

Убрать `--dry-run` для активного режима (с сохранением решений).

**Клавиши управления:**

| Клавиша | Действие |
|---------|----------|
| `n` | Следующее изображение |
| `p` | Предыдущее изображение |
| `k` | OK — разметка верна |
| `d` | Удалить разметку для этого изображения |
| `m` | Отметить для ручной проверки |
| `q` | Выход |

Результаты логируются в CSV, формируются файлы `ok.txt` и `needs_manual_review.txt`.

### Импорт в Label Studio для корректировки

Конвертировать предсказания YOLO в формат Label Studio:

```bash
python -m scripts.convert_yolo_predictions_to_label_studio \
  data/frames/all \
  custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels \
  outputs/predictions_all.json
```

Импортировать `outputs/predictions_all.json` в Label Studio как pre-annotations, проверить и откорректировать ошибки, затем экспортировать финальный датасет.

---

## 6. Финальная модель

### Разделение финального датасета

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-7-at-2026-05-15-14-03-f712bcc8 \
  data/yolo_final/project-7-at-2026-05-15-14-03-f712bcc8 \
  --val-ratio 0.2
```

### Обучение финальной модели

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/yolo_final/project-7-at-2026-05-15-14-03-f712bcc8/data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  project=custom_models/final \
  name=custom_final_1
```

Веса сохраняются в `custom_models/final/custom_final_1/weights/best.pt`.

---

## 7. Запуск с кастомной моделью

В конфиге (`configs/experiments/custom_objects.yaml`) указать путь к весам:

```yaml
detector:
  model: "custom_models/final/custom_final_1/weights/best.pt"
  allowed_classes: ["arx", "taar", "the_institute"]
```

Запуск:

```bash
python -m src.main --config configs/experiments/custom_objects.yaml
python -m src.main --source data/input/video.mp4 --config configs/experiments/custom_objects.yaml
```

---

## Классы

Модель обучена на 3 кастомных классах:

| ID | Имя класса |
|----|-----------|
| 0 | arx |
| 1 | taar |
| 2 | the_institute |
