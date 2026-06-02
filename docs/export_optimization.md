# Экспорт и оптимизация модели (ONNX / TensorRT)

Ускорение инференса и переносимость на другое железо за счёт экспорта обученной
YOLO-модели в оптимизированные форматы.

---

## Зачем это нужно

| Формат | Где работает | Скорость | Назначение |
|--------|--------------|----------|------------|
| `.pt` (PyTorch) | CPU / любая CUDA-GPU | базовая | разработка, обучение |
| `.onnx` | **любое железо** через ONNX Runtime (CPU, NVIDIA, AMD, Intel, Apple) | средняя/высокая | **переносимость** + промежуточный шаг к TensorRT |
| `.engine` (TensorRT) | только NVIDIA GPU | максимальная | продакшен на конкретной NVIDIA-карте |

> **Как роли разделены в этом проекте (ultralytics 8.0.200):**
> - **GPU на NVIDIA → `.engine` (TensorRT)** — самый быстрый путь, валидирован.
> - **Переносимость / CPU / не-NVIDIA → `.onnx`** — работает на CPU.
>
> Важно: **в пайплайне `.onnx` исполняется на CPU**, даже при `device: "cuda"`.
> Это ограничение Ultralytics 8.0.200: его `AutoBackend` принудительно переводит
> ONNX (и другие не-`pt`/`engine` форматы) на CPU
> (`ultralytics/nn/autobackend.py:103-106`). Поэтому GPU-ускорение здесь даёт
> только TensorRT, а ONNX используется как переносимый формат. ONNX-инференс на
> GPU потребовал бы апгрейда Ultralytics — отдельная задача.

### ONNX vs TensorRT — в чём разница

- **ONNX** — открытый, вендор-нейтральный формат обмена. Один и тот же `.onnx`
  гоняется через разные *execution providers* ONNX Runtime: `CPUExecutionProvider`,
  `CUDAExecutionProvider` (NVIDIA), `DmlExecutionProvider` (DirectML — AMD/Intel на
  Windows), `CoreMLExecutionProvider` (Apple), `OpenVINOExecutionProvider` (Intel).
  Это и есть слой переносимости — если целевая машина без NVIDIA, выбираем ONNX.
- **TensorRT** — проприетарная оптимизация NVIDIA. Работает только на их GPU и даёт
  максимальный FPS. Ultralytics строит `.engine` **через промежуточный ONNX**, так
  что понимание ONNX — фундамент и для TensorRT.

> **Важно:** `.engine` привязан к конкретной модели GPU и версии TensorRT, на
> которых был собран. Его нельзя переносить между машинами и нельзя коммитить —
> только пересобирать. `.onnx` и `.engine` добавлены в `.gitignore`.

---

## Установка зависимостей

Зависимости разнесены на два файла: переносимый ONNX-путь — в основном
`requirements.txt`, опциональный NVIDIA-only TensorRT — в `requirements-tensorrt.txt`.

### ONNX-путь — в основном `requirements.txt` (ставится на любом железе)

```
onnx==1.15.0
onnxruntime==1.16.3      # CPU-рантайм; именно его пайплайн использует для .onnx
```

Ставится `onnxruntime` (CPU), а **не** `onnxruntime-gpu`, потому что Ultralytics
8.0.200 всё равно гоняет ONNX на CPU (см. заметку выше). GPU-пакет здесь не даёт
выигрыша, а его установка к тому же затирает общий пакет `onnxruntime` (они делят
один namespace) — на NVIDIA для GPU используем `.engine` (TensorRT).

- **На не-NVIDIA машине** при желании можно поставить EP под своё железо
  (`onnxruntime-directml` для AMD/Intel на Windows, `onnxruntime-openvino` для
  Intel и т.п.) — формат `.onnx` от этого не меняется.

### TensorRT-путь — `requirements-tensorrt.txt` (только NVIDIA GPU, opt-in)

```bash
pip install --no-build-isolation -r requirements-tensorrt.txt
# Проверка:
python -c "import tensorrt as trt; print(trt.__version__)"
```

Флаг `--no-build-isolation` **обязателен**. PyPI-пакет `tensorrt` — это мета-пакет,
чей `setup.py` (кастомный `InstallCommand`) запускает подпроцесс
`python -m pip install tensorrt_libs ...`. При изоляции сборки pip прячет от этого
подпроцесса `site-packages` основного окружения (вместе с `pip`) → `No module named
pip` → сборка падает. `--no-build-isolation` собирает с видимым окружением, и
подпроцесс находит `pip`. Сам мета-пакет нужен по-настоящему: импортируемый модуль
`tensorrt` — это его 2-строчная обёртка (`from tensorrt_bindings import *`); одни
`tensorrt-libs`/`tensorrt-bindings` модуль `tensorrt` **не** дают.

> **Несовпадение версий CUDA — проверено, работает.** `tensorrt-libs 8.6.1`
> тянет CUDA-12 рантайм (`nvidia-*-cu12`, ~2.5 ГБ) рядом с CUDA-11.8 torch (cu118).
> Реальный экспорт `.pt → .engine` (FP16) и инференс `.engine` отрабатывают —
> два рантайма сосуществуют за счёт разных soname. Биндингам TRT нужна
> `libcudnn.so.8`, и её **несёт сам torch** (`torch/lib/libcudnn.so.8`, cuDNN 8.7):
> она находится, пока torch импортируется раньше tensorrt — а пайплайн всегда так
> и делает (ultralytics/torch грузятся первыми). Проверка: `python -c "import
> torch, tensorrt"`.
>
> Не ставить на машинах без NVIDIA GPU.

---

## Экспорт: `scripts/export_model.py`

Обёртка над `YOLO(...).export(...)`. Экспортированный файл кладётся рядом с
исходными весами (поведение Ultralytics); `--output-dir` переносит его в нужную
папку (например `models/`).

### ONNX (переносимый формат)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format onnx
python -m scripts.export_model --weights models/yolov8n.pt --format onnx --half
```

### TensorRT FP16 (NVIDIA GPU)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format engine --half
```

### TensorRT INT8 (максимальная скорость)

INT8 требует **калибровочный датасет** — используем тот же `data.yaml`, что и при
обучении:

```bash
python -m scripts.export_model \
  --weights custom_models/final/custom_final_1/weights/best.pt \
  --format engine --int8 \
  --data data/yolo_final/<project>/data.yaml
```

### Аргументы

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--weights` | `models/yolov8n.pt` | Исходные `.pt`-веса |
| `--format` | — (обязательный) | `onnx` или `engine` |
| `--imgsz` | `640` | Размер входа, зашиваемый в модель |
| `--half` | off | FP16 (быстрее, ~без потери точности) |
| `--int8` | off | INT8-квантизация (только `engine`, требует `--data`) |
| `--data` | — | `data.yaml` для калибровки INT8 |
| `--dynamic` | off | Динамические размеры входа (ONNX; несовместимо с `--int8`) |
| `--batch` | `1` | Максимальный batch |
| `--device` | `0` | Устройство сборки (`0` = первая GPU, `cpu`) |
| `--output-dir` | — | Куда перенести результат (напр. `models/`) |

Ограничения валидируются: `--int8` требует `--data` и `--format engine`;
`--half` и `--int8` взаимоисключающие; `--dynamic` несовместим с `--int8`.

---

## Подключение в пайплайн

Достаточно указать путь к экспортированной модели в конфиге — формат определяется
по расширению автоматически:

```yaml
detector:
  model: "models/yolov8n.onnx"     # или models/yolov8n.engine
  device: "cuda"                    # для .engine всегда GPU
```

```bash
python -m src.main --config configs/default.yaml --source data/input/video.mp4
```

### Как это работает внутри

`ObjectDetector` определяет бэкенд по суффиксу файла
(`_detect_backend`: `.pt` → `pytorch`, `.onnx` → `onnx`, `.engine` → `tensorrt`):

- `.pt` — модель грузится и переносится на устройство через `.to(device)`.
- `.onnx` / `.engine` — `.to()` **не вызывается** (на engine это падает; формат уже
  привязан к своему рантайму). Устройство пробрасывается в `model.track(device=...)`
  через трекер.
- Для `.engine` устройство принудительно `cuda` (CPU-fallback'а нет, GPU обязателен).
- Для `.onnx` Ultralytics 8.0.200 принудительно выбирает CPU
  (`autobackend.py:103-106`) — `.onnx` всегда исполняется на CPU независимо от
  `device`. Это переносимый путь; для GPU на NVIDIA используется `.engine`.

Модель загружается один раз в `ObjectDetector` и переиспользуется трекером
(`model.track()`), поэтому экспортированный формат работает и для детекции, и для
трекинга без отдельной конвертации.

---

## Бенчмарк: `scripts/benchmark_backends.py`

Сравнение скорости инференса между бэкендами на одном видео (замеряется forward-pass
детекции — именно его ускоряет экспорт; оверхед трекинга одинаков для всех форматов):

```bash
python -m scripts.benchmark_backends \
  --source data/input/video.mp4 \
  --models models/yolov8n.pt models/yolov8n.onnx models/yolov8n.engine \
  --max-frames 300 --warmup 20
```

Выводит таблицу: avg/p50/p95 латентности (мс), FPS и ускорение относительно первой
модели в списке (обычно `.pt` как базлайн). Первые `--warmup` кадров не учитываются
(ленивая инициализация, загрузка engine, прогрев CUDA). Отсутствующие модели
пропускаются — можно сравнивать только то, что собрано.

Измеренный результат на этом проекте (yolov8n, 640, RTX 3050 Ti Laptop):

| Backend | FPS | Avg ms | Speedup |
|---------|-----|--------|---------|
| `.pt` (pytorch, GPU) | ~147 | 6.8 | 1.00× |
| `.onnx` (onnxruntime, **CPU**) | ~28 | 36 | 0.19× |
| `.engine` (TensorRT FP16, GPU) | ~189 | 5.3 | **1.28×** |

`.onnx` медленнее именно потому, что Ultralytics исполняет его на CPU (см. выше) —
это ожидаемо, а не ошибка. Быстрый путь на NVIDIA — `.engine`.

---

## Ручная проверка функционала

Пошагово, что и как проверить вручную. Команды запускать из корня проекта в
основном окружении. Перед прогонами, использующими ONNX через Ultralytics,
выставляй `YOLO_AUTOINSTALL=false` — иначе Ultralytics может доустановить CPU-пакет
`onnxruntime` и затереть установленный (см. «Типичные проблемы»).

### 1. Экспорт ONNX

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format onnx
ls -lh models/yolov8n.onnx
```
**Ожидается:** `Export complete: models/yolov8n.onnx`, файл создан (~12 МБ).

### 2. Экспорт TensorRT (FP16)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format engine --half
ls -lh models/yolov8n.engine
```
**Ожидается:** `building FP16 engine ...`, в конце `export success ✅`,
`models/yolov8n.engine` (~9 МБ). Сборка занимает несколько минут — это нормально.
Падение с `libcudnn.so.8` означает, что tensorrt импортируется раньше torch — в
наших скриптах такого нет, но при ручных проверках импортируй `torch` первым.

### 3. TensorRT работает на GPU (импорт)

```bash
python -c "import torch, tensorrt as trt; print(trt.__version__)"
```
**Ожидается:** печатает `8.6.1` без ошибок (torch первым — он даёт `libcudnn.so.8`).

### 4. Прогон пайплайна на каждом бэкенде

Указать модель в конфиге (`detector.model: "models/yolov8n.engine"` либо `.onnx`,
либо `.pt`) или быстро проверить на видео. В выводе при старте печатается строка
`Detector backend: ...`:

```bash
# .pt  -> backend pytorch (GPU)
# .onnx -> backend onnx (CPU, ожидаемо)
# .engine -> backend tensorrt (GPU)
python -m src.main --source data/input/street_camera_2x_speed.mp4 \
  --config configs/default.yaml
```
**Ожидается:** окно с детекциями/трекингом; для `.engine` и `.pt` — высокий FPS,
для `.onnx` — заметно ниже (CPU). `track_id` стабильны на всех бэкендах (проверка
совместимости `model.track()` с экспортированными форматами).

### 5. Сравнение скорости (бенчмарк)

```bash
YOLO_AUTOINSTALL=false python -m scripts.benchmark_backends \
  --source data/input/street_camera_2x_speed.mp4 \
  --models models/yolov8n.pt models/yolov8n.onnx models/yolov8n.engine \
  --max-frames 120 --warmup 15
```
**Ожидается:** таблица; `.engine` быстрее `.pt` (~1.3×), `.onnx` медленнее (CPU).

### 6. Переносимость ONNX (CPU)

То, что `.onnx` идёт на CPU, и есть демонстрация переносимости — этот же файл
запустится на машине без NVIDIA. Проверка, что `onnxruntime` реально берёт CPU:
```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```
**Ожидается:** в списке есть `CPUExecutionProvider` (на чистом `onnxruntime` —
только он; это и нужно).

### 7. INT8-engine (опционально, нужен датасет)

```bash
python -m scripts.export_model \
  --weights custom_models/final/custom_final_1/weights/best.pt \
  --format engine --int8 --data data/yolo_final/<project>/data.yaml
```
**Ожидается:** успешная сборка INT8-движка. Сравнить детекции с FP16/`.pt` на
тестовом наборе — убедиться, что просадка точности приемлема.

---

## Типичные проблемы

**`.engine` не запускается / ошибка версии TensorRT** → engine собран под другую
версию TensorRT или другую GPU. Пересобрать `scripts/export_model.py` на текущей машине.

**ONNX работает на CPU при `device: "cuda"`** → это ожидаемо: Ultralytics 8.0.200
принудительно исполняет ONNX на CPU (`autobackend.py:103-106`). Для GPU на NVIDIA
используй `.engine` (TensorRT). ONNX-на-GPU потребовал бы апгрейда Ultralytics.

**После прогона ONNX в окружении появился пакет `onnxruntime` (а был только нужный)**
→ Ultralytics при onnx-инференсе вызывает `check_requirements('onnxruntime')` и
доустанавливает CPU-пакет. Это норма для Пути 1 (мы используем именно CPU-`onnxruntime`).
Чтобы Ultralytics ничего не доустанавливал в проде, запускай с переменной окружения
`YOLO_AUTOINSTALL=false`.

**`Failed building wheel for tensorrt` (`No module named pip`)** → забыт флаг
`--no-build-isolation`. Ставить строго так:
`pip install --no-build-isolation -r requirements-tensorrt.txt`. Если и с флагом
падает — проверить, что в окружении есть `setuptools` и `wheel`
(`python -m pip install -U pip setuptools wheel`), и повторить.

**INT8: падает с требованием данных** → `--int8` обязательно нужен `--data <data.yaml>`
с калибровочными изображениями.
