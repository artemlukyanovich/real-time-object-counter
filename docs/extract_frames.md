# Извлечение кадров из видео

`scripts/extract_frames.py` — утилита для сохранения каждого N-го кадра из видеофайла в виде JPG-изображений.

## Использование

```bash
python -m scripts.extract_frames <video> <output> [--step N]
```

### Аргументы

| Аргумент | Тип | Описание |
|----------|-----|----------|
| `video` | path | Путь к видеофайлу |
| `output` | path | Папка для сохранения кадров (создаётся автоматически) |
| `--step` | int | Каждый N-й кадр (по умолчанию: `15`) |
| `--dry-run` | flag | Предпросмотр: показать количество кадров без сохранения |

## Примеры

```bash
# Каждый 15-й кадр (по умолчанию)
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip

# Каждый 30-й кадр
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 30

# Каждый кадр
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 1

# Предпросмотр (без сохранения файлов)
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 15 --dry-run
```

## Вывод

```
Processed : 1500 frames
Saved     : 100 frames → data/frames/clip
```

Предпросмотр (`--dry-run`):

```
Dry run   : 1500 frames in video, step=15
Would save: 100 frames → data/frames/clip
```

## Именование файлов

Кадры сохраняются в формате `frame_0001.jpg`, `frame_0002.jpg`, ... — порядковый номер сохранённого кадра (не номер кадра в исходном видео).

## Структура данных

```
data/
  raw_videos/     ← исходные видеофайлы
  frames/
    clip_name/    ← кадры конкретного видео
      frame_0001.jpg
      frame_0002.jpg
      ...
```
