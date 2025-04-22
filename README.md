# SuperStack

> **Multi‑frame super‑resolution & denoising via stacking**  
> Pure signal processing—no AI. Supports image folders, video files and `.ser` clips.

---

## 🚀 Features

- **Quality filtering**: reject blurry frames by  
  - **Laplacian variance** (`--lap-thresh`)  
  - **Top‑N% sharpest** (`--top-percent`)
- **Alignment**  
  - **ECC** (sub‑pixel translation / Euclidean)  
  - **ORB+RANSAC** (rotation, scale, homography)
- **Stacking**  
  - **Average** (classic SNR ↑√N)  
  - **Median** (robust to outliers like passing birds)
- **Sharpening**: unsharp mask boost (`--unsharp-amount`)
- **Input types**  
  - Folder of images (`--input-folder`)  
  - Video file (`--input-video`)  
  - `.ser` astronomical clips (`--input-ser`) via [pySER‑Reader](https://github.com/Copper280z/pySER-Reader)
- **Defaults & conventions**  
  - `Input/` folder & `Output/` directory by default  
  - Output is named after your input (e.g. `2022-12-10-0254.png`)

---

## 💾 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourname/SuperStack.git
   cd SuperStack
   ```
2. **Install dependencies**  
   ```bash
   pip install numpy opencv-python tqdm
   ```
3. **(SER support)**  
   - Copy `ser_reader.py` from [pySER‑Reader](https://github.com/Copper280z/pySER-Reader) into this folder.

---

## ⚙️ Usage

Make sure you have some frames in `Input/` (or a video / `.ser` file). Then:

```bash
# default Input/ → Output/<basename>.png
python SuperStack.py

# from a .ser clip
python SuperStack.py --input-ser Input/2022-12-10-0254_3.ser

# custom folder + top‑10% sharpest + ORB alignment
python SuperStack.py   --input-folder MyFrames   --top-percent 10   --align-method ORB   --output Output/result.png

# extract from video
python SuperStack.py --input-video clip.mp4 --lap-thresh 50
```

### CLI Flags

| Flag                   | What it does                                        | Default         |
|------------------------|------------------------------------------------------|-----------------|
| `--input-folder PATH`  | Folder of images (glob .jpg/.png/.tif)               | `Input/`        |
| `--input-video FILE`   | Video file to extract frames from                    | —               |
| `--input-ser FILE`     | `.ser` file to load via pySER‑Reader                 | —               |
| `--lap-thresh FLOAT`   | Reject frames below this Laplacian variance          | *off*           |
| `--top-percent FLOAT`  | Keep top P% of sharpest frames                       | *off*           |
| `--align-method STR`   | `ECC` or `ORB`                                       | `ECC`           |
| `--stack-method STR`   | `average` or `median`                                | `average`       |
| `--unsharp-amount FLT` | Amount for unsharp mask                              | `1.5`           |
| `--output FILE`        | Output filename (`.png`/`.jpg`)                      | `Output/<base>.png` |

---

## 🛠 Under the Hood

1. **Load** frames (images / video / SER)  
2. **Score & filter** by blur metric  
3. **Register** all kept frames to a reference  
4. **Stack** (mean or median) → high SNR result  
5. **Sharpen** via unsharp mask → crisp details  
6. **Save** with auto‑naming

---

## ⚡ Performance Tips

- **Coarse‑to‑fine ECC** (pyramid) → 3–5× speedup  
- **Translation‑only** if no rotation → 2× faster  
- **ROI crop** (center region) → focus warp on your target  
- **Parallel align** with `concurrent.futures` → ~linear core scaling  

---

## 🤝 Contributing

1. Fork & branch  
2. Add tests under `tests/`  
3. Send a PR with a clear description

---

## 📜 License

MIT © Sepp Beld
