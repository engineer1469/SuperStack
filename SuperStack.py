#!/usr/bin/env python3
"""
SuperStack.py – multi-frame super-resolution & sharpening via stacking

Supports: image folders, video files, .ser clips,  
+ classical RL super-res + wavelet sharpening  
(No AI, pure signal processing)
"""

import os, glob, argparse
import cv2, numpy as np
from tqdm import tqdm
from ser_reader import reader  # from pySER-Reader

# ─── I/O ────────────────────────────────────────────────────────

def load_images_from_folder(folder, exts=('*.jpg','*.png','*.tif')):
    files = []
    for e in exts:
        files += sorted(glob.glob(os.path.join(folder, e)))
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            yield img

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def load_frames_from_ser(path):
    ser = reader(path)
    frames = []
    for i in range(ser.header.frameCount):
        img = ser.getImg(i)
        # demosaic / mono→BGR
        if ser.header.numPlanes == 1:
            cid = ser.header.colorID
            if cid == 0:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif cid in (8,9,10,11):
                code = {8:cv2.COLOR_BAYER_RG2BGR,
                        9:cv2.COLOR_BAYER_GR2BGR,
                        10:cv2.COLOR_BAYER_GB2BGR,
                        11:cv2.COLOR_BAYER_BG2BGR}[cid]
                img = cv2.cvtColor(img, code)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            cid = ser.header.colorID
            if cid == 100:          # RGB→BGR
                img = img[..., ::-1]
        frames.append(img)
    return frames

# ─── QUALITY FILTER ────────────────────────────────────────────

def laplacian_score(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y   = yuv[...,0]
    return cv2.Laplacian(y, cv2.CV_64F).var()

def filter_by_quality(imgs, lap_thresh=None, top_percent=None):
    scores = [laplacian_score(im) for im in tqdm(imgs, desc="Scoring  ")]
    if lap_thresh is not None:
        return [im for im,s in zip(imgs,scores) if s>=lap_thresh]
    if top_percent is not None:
        n = int(len(imgs)*(top_percent/100))
        idx = np.argsort(scores)[-n:]
        return [imgs[i] for i in sorted(idx)]
    return imgs

# ─── ALIGNMENT ─────────────────────────────────────────────────

def align_ecc(imgs, iterations=5000, eps=1e-10):
    ref = imgs[0]
    h,w = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    aligned = [ref]
    for im in tqdm(imgs[1:], desc="Aligning ECC"):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        M = np.eye(2,3, dtype=np.float32)
        _, M = cv2.findTransformECC(
            ref_gray, gray, M, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, iterations, eps)
        )
        warped = cv2.warpAffine(
            im, M, (w,h),
            flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP
        )
        aligned.append(warped)
    return aligned

def align_orb(imgs, keep_kp=5000, match_ratio=0.9):
    ref = imgs[0]
    h,w = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(keep_kp)
    kp1,des1 = orb.detectAndCompute(ref_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    aligned = [ref]
    for im in tqdm(imgs[1:], desc="Aligning ORB"):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp2,des2 = orb.detectAndCompute(gray, None)
        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        keep    = matches[:int(len(matches)*match_ratio)]
        src = np.float32([kp1[m.queryIdx].pt for m in keep]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in keep]).reshape(-1,1,2)
        M,_ = cv2.findHomography(dst, src, cv2.RANSAC)
        warped = cv2.warpPerspective(im, M, (w,h))
        aligned.append(warped)
    return aligned

# ─── STACK & SHARPEN ────────────────────────────────────────────

def stack_images(imgs, method='average'):
    arr = np.stack([im.astype(np.float32) for im in imgs], axis=0)
    res = np.median(arr,axis=0) if method=='median' else np.mean(arr,axis=0)
    return np.clip(res,0,255).astype(np.uint8)

def unsharp_mask(img, ksize=(5,5), sigma=1.0, amount=1.5, thresh=0):
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    sharp = float(amount+1)*img - float(amount)*blurred
    sharp = np.clip(sharp,0,255).astype(np.uint8)
    if thresh>0:
        mask = np.abs(img-blurred)<thresh
        sharp[mask] = img[mask]
    return sharp

# ─── SUPER-RESOLUTION (RL) ───────────────────────────────────────

def generate_psf(size=5, sigma=1.0):
    g = cv2.getGaussianKernel(size, sigma)
    return g @ g.T

def richardson_lucy(img, psf, iterations=10):
    img_f = img.astype(np.float32) + 1e-6
    estimate = img_f.copy()
    psf_m  = psf[::-1, ::-1]
    for _ in tqdm(range(iterations), desc="RL Deconv"):
        conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        rel  = img_f/(conv+1e-6)
        estimate *= cv2.filter2D(rel, -1, psf_m, borderType=cv2.BORDER_REFLECT)
    return estimate

# ─── WAVELET SHARPENING ─────────────────────────────────────────

def wavelet_sharpen(img, levels=3, boost=1.2):
    # build Gaussian pyramid
    G = [img.copy()]
    for i in tqdm(range(levels), desc="Wavelet↓"):
        G.append(cv2.pyrDown(G[-1]))
    # build Laplacian pyramid
    L = []
    for i in range(levels):
        up = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1],G[i].shape[0]))
        L.append(cv2.subtract(G[i], up))
    L.append(G[-1])
    # boost all but the smallest residual
    for i in tqdm(range(levels), desc="Boosting"):
        L[i] = cv2.multiply(L[i], boost)
    # reconstruct
    rec = L[-1]
    for i in reversed(range(levels)):
        rec = cv2.pyrUp(rec, dstsize=(L[i].shape[1],L[i].shape[0]))
        rec = cv2.add(rec, L[i])
    return np.clip(rec,0,255).astype(np.uint8)

# ─── MAIN ────────────────────────────────────────────────────────

def main():
    default_input  = 'Input'
    default_output = os.path.join('Output','stacked.png')

    p = argparse.ArgumentParser(prog="SuperStack",
        description="StackSR + RL-superres + wavelet sharpening")
    g = p.add_mutually_exclusive_group()
    g.add_argument('--input-folder', help="Folder of images", default=default_input)
    g.add_argument('--input-video',  help="Video file")
    g.add_argument('--input-ser',    help=".ser file")
    p.add_argument('--lap-thresh',  type=float, help="Laplacian blur threshold")
    p.add_argument('--top-percent', type=float, help="Top P% by sharpness")
    p.add_argument('--align-method', choices=['ECC','ORB'], default='ECC')
    p.add_argument('--stack-method', choices=['average','median'], default='average')
    p.add_argument('--unsharp-amount', type=float, default=1.5)
    p.add_argument('--sr-upscale',    type=int,   default=1, help="Upscale factor")
    p.add_argument('--sr-iterations', type=int,   default=0, help="RL deconv iters")
    p.add_argument('--psf-size',      type=int,   default=5)
    p.add_argument('--psf-sigma',     type=float, default=1.0)
    p.add_argument('--wavelet-levels',type=int,   default=0, help="Enable wavelet sharpen")
    p.add_argument('--wavelet-boost', type=float, default=1.2, help="Wavelet gain")
    p.add_argument('--output',        help="Output file", default=default_output)
    args = p.parse_args()

    # derive output name from input if left default
    if args.output==default_output:
        base = args.input_ser or args.input_video or args.input_folder
        base = os.path.splitext(os.path.basename(base.rstrip("/\\")))[0]
        args.output = os.path.join(os.path.dirname(args.output), f"{base}.png")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("→ Loading frames")
    if args.input_ser:
        frames = load_frames_from_ser(args.input_ser)
    elif args.input_video:
        frames = list(load_frames_from_video(args.input_video))
    else:
        frames = list(load_images_from_folder(args.input_folder))

    if not frames:
        raise RuntimeError("No frames found.")

    print("→ Filtering by sharpness")
    good = filter_by_quality(frames,
                             lap_thresh=args.lap_thresh,
                             top_percent=args.top_percent)
    if not good:
        raise RuntimeError("All frames filtered out.")

    print("→ Aligning frames")
    aligned = (align_ecc(good) if args.align_method=='ECC'
               else align_orb(good))

    print("→ Stacking")
    stacked = stack_images(aligned, method=args.stack_method)

    # super-res step
    if args.sr_upscale>1:
        print(f"→ Upscaling ×{args.sr_upscale}")
        h,w = stacked.shape[:2]
        stacked = cv2.resize(
            stacked, (w*args.sr_upscale,h*args.sr_upscale),
            interpolation=cv2.INTER_CUBIC
        )
        if args.sr_iterations>0:
            print(f"→ RL deconvolution ({args.sr_iterations} iters)")
            psf = generate_psf(size=args.psf_size, sigma=args.psf_sigma)
            stacked = richardson_lucy(stacked, psf, iterations=args.sr_iterations)
            stacked = np.clip(stacked,0,255).astype(np.uint8)

    # wavelet sharpen
    if args.wavelet_levels>0:
        print(f"→ Wavelet sharpen ({args.wavelet_levels} levels ×{args.wavelet_boost})")
        stacked = wavelet_sharpen(stacked,
                                  levels=args.wavelet_levels,
                                  boost=args.wavelet_boost)

    # final unsharp mask
    print("→ Final unsharp mask")
    final = unsharp_mask(stacked, amount=args.unsharp_amount)

    cv2.imwrite(args.output, final)
    print(f"✅ Done → {args.output}")

if __name__ == '__main__':
    main()
