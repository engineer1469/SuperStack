# SuperStack

> **Multi‑frame super‑resolution & denoising via stacking**  
> Pure signal processing—no AI. Supports image folders, video files, and `.ser` clips,  
> now with classical super‑resolution (RL) & wavelet sharpening, plus progress bars.

---

## 🚀 Features

- **Quality filtering**  
  - **Laplacian variance** (`--lap-thresh`) on luma  
  - **Top‑N% sharpest** (`--top-percent`)
- **Alignment**  
  - **ECC** (sub‑pixel translation/Euclidean)  
  - **ORB+RANSAC** (rotation, scale, homography)
- **Stacking**  
  - **Average** (SNR ↑√N)  
  - **Median** (robust to outliers)
- **Sharpening**  
  - **Unsharp mask** (`--unsharp-amount`)
  - **Wavelet pyramid boost** (`--wavelet-levels`, `--wavelet-boost`)
- **Super‑Resolution**  
  - **Drizzle‑style upsample** (`--sr-upscale`)  
  - **Richardson–Lucy deconvolution** (`--sr-iterations`, `--psf-size`, `--psf-sigma`)
- **Progress bars** via `tqdm` on all heavy loops
- **Input types**  
  - Folder of images (`--input-folder`)  
  - Video file (`--input-video`)  
  - `.ser` astronomical clips (`--input-ser`)
- **Defaults**  
  - **Input/** folder & **Output/** directory  
  - Auto‑naming: output matches input base name

---

## 💾 Installation

```bash
git clone https://github.com/yourname/SuperStack.git
cd SuperStack
pip install numpy opencv-python tqdm
# (for .ser support)
# copy ser_reader.py from https://github.com/Copper280z/pySER-Reader into this folder
```

---

## ⚙️ Usage

Drop frames into `Input/` (or specify a video/.ser). Then run:

```bash
python SuperStack.py
```

By default, frames in `Input/` are scored → aligned → stacked → sharpened → saved as `Output/<base>.png`.

### Examples

- **Stack a .ser clip**  
  ```bash
  python SuperStack.py --input-ser Input/2022-12-10-0254_3.ser
  ```
- **Use top 0.5% sharpest frames + ORB**  
  ```bash
  python SuperStack.py     --input-folder MyFrames     --top-percent 0.5     --align-method ORB     --output Output/result.png
  ```
- **Apply 2× super‑res + 10 RL iters + wavelet sharpen**  
  ```bash
  python SuperStack.py     --input-ser Input/2022-12-10-0254_3.ser     --top-percent   0.5     --align-method  ECC     --stack-method  average     --sr-upscale    2     --sr-iterations 10     --psf-size      5     --psf-sigma     1.2     --wavelet-levels 2     --wavelet-boost 1.1     --unsharp-amount 1.3     --output        Output/mars_tuned.png
  ```

---

### CLI Flags

| Flag                        | Description                                        | Default          |
|-----------------------------|----------------------------------------------------|------------------|
| `--input-folder PATH`       | Folder of images (`*.jpg`,`*.png`,`*.tif`)         | `Input/`         |
| `--input-video FILE`        | Video file to extract frames                       | —                |
| `--input-ser FILE`          | `.ser` file to load via pySER‑Reader               | —                |
| `--lap-thresh FLOAT`        | Luma-Laplacian variance threshold to reject blur   | *off*            |
| `--top-percent FLOAT`       | Keep top P% of sharpest frames                     | *off*            |
| `--align-method {ECC,ORB}`  | Alignment algorithm                                | `ECC`            |
| `--stack-method {average,median}` | Stacking method                               | `average`        |
| `--unsharp-amount FLOAT`    | Unsharp mask amount                                | `1.5`            |
| `--sr-upscale INT`          | Super‑res upsample factor (1=off)                  | `1`              |
| `--sr-iterations INT`       | Richardson–Lucy deconv iterations (0=off)          | `0`              |
| `--psf-size INT`            | Gaussian PSF kernel size for RL                    | `5`              |
| `--psf-sigma FLOAT`         | Gaussian PSF σ                                       | `1.0`            |
| `--wavelet-levels INT`      | Wavelet pyramid levels for sharpening (0=off)      | `0`              |
| `--wavelet-boost FLOAT`     | Boost factor for wavelet bands                     | `1.2`            |
| `--output FILE`             | Output filename (`.png`/`.jpg`)                    | `Output/<base>.png` |

---

## 🛠 Under the Hood

1. **Load** images/video/SER frames  
2. **Score** luma sharpness & filter  
3. **Align** frames (ECC or ORB)  
4. **Stack** (mean/median) → high SNR  
5. **Super‑res** (optional RL deconv)  
6. **Wavelet** sharpen (optional multi-scale boost)  
7. **Unsharp mask** → final crisp image  
8. **Save** & auto‑name

---

## ⚡ Performance Tips

- **Pyramid ECC** for speed  
- **Translation‑only** if no rotation  
- **ROI crop** around target  
- **Parallelize** align/stack with `concurrent.futures`

---

## 🧪 Mars Reference

The **Mars‑tuned** command above picks 0.5% sharpest of ~32 k frames, 2× super‑res, 10 RL iters, 2‑level wavelet, then unsharp mask—ideal for your `Input/2022-12-10-0254_3.ser` data.

---

## 🤝 Contributing

1. Fork & branch  
2. Add tests in `tests/`  
3. Send a PR

---

## 📜 License

MIT © Sepp Beld
