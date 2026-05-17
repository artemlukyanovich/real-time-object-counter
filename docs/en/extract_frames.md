# Frame Extraction from Video

`scripts/extract_frames.py` — utility for saving every N-th frame from a video file as JPG images.

## Usage

```bash
python -m scripts.extract_frames <video> <output> [--step N] [--prefix STR]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `video` | path | Path to the video file |
| `output` | path | Output directory (created automatically) |
| `--step` | int | Save every N-th frame (default: `15`) |
| `--prefix` | str | File name prefix (default: `frame`) |
| `--dry-run` | flag | Preview: show frame count without saving |

## Examples

```bash
# Every 15th frame (default)
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip

# Every 30th frame
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 30

# With a custom prefix
python -m scripts.extract_frames data/raw_videos/arx_1.mp4 data/frames/arx --step 30 --prefix arx
python -m scripts.extract_frames data/raw_videos/arx_2.mp4 data/frames/arx --step 30 --prefix arx

# Preview (without saving files)
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 15 --dry-run
```

## Output

```
Processed : 1500 frames
Saved     : 100 frames → data/frames/clip
```

Preview (`--dry-run`):

```
Dry run   : 1500 frames in video, step=15
Would save: 100 frames → data/frames/clip
```

## File Naming

Frames are saved as `{prefix}_{number}.jpg`, for example:
- `frame_0001.jpg`, `frame_0002.jpg`, ... (default)
- `arx1_0001.jpg`, `arx2_0001.jpg`, ... (custom prefix)

Numbering continues from the last existing file with the same prefix, allowing multiple videos to be processed into the same directory without overwriting.

## Directory Structure

```
data/
  raw_videos/     ← source video files
  frames/
    clip_name/    ← frames from a specific video
      frame_0001.jpg
      frame_0002.jpg
      ...
    mixed/        ← frames from multiple videos with different prefixes
      arx1_0001.jpg
      arx2_0001.jpg
      ...
```
