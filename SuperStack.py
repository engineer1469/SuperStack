#!/usr/bin/env python3
"""
stackSR.py – multi-frame super-resolution and denoising via stacking

Usage:
  pip install opencv-python numpy tqdm
  # will default to Input/ for frames and Output/stacked.png for output
  ./stackSR.py \
    [--input-folder PATH/to/images] \
    [--input-video PATH/to/video.mp4] \
    [--lap-thresh 100.0] \
    [--top-percent 50] \
    [--align-method ECC|ORB] \
    [--stack-method average|median] \
    [--unsharp-amount 1.5] \
    [--output PATH/to/result.png]
"""

import os, glob, argparse
import cv2, numpy as np
from tqdm import tqdm
from ser_reader import reader #Credit to https://github.com/Copper280z/pySER-Reader.git

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

def laplacian_score(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = yuv[...,0]
    return cv2.Laplacian(y, cv2.CV_64F).var()

def filter_by_quality(imgs, lap_thresh=None, top_percent=None):
    scores = [laplacian_score(im) for im in tqdm(imgs, desc="Scoring")]
    imgs_ret = []
    if lap_thresh is not None:
        for im, s in zip(imgs, scores):
            if s >= lap_thresh:
                imgs_ret.append(im)
    elif top_percent is not None:
        n = int(len(imgs) * (top_percent/100.0))
        idx = np.argsort(scores)[-n:]
        imgs_ret = [imgs[i] for i in sorted(idx)]
    else:
        imgs_ret = imgs
    return imgs_ret

def align_ecc(imgs, iterations=5000, eps=1e-10):
    ref = imgs[0]
    h, w = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    aligned = [ref]
    for im in tqdm(imgs[1:], desc="Aligning ECC"):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        M = np.eye(2,3, dtype=np.float32)
        _, M = cv2.findTransformECC(ref_gray, gray, M,
                                     cv2.MOTION_EUCLIDEAN,
                                     (cv2.TERM_CRITERIA_EPS |
                                      cv2.TERM_CRITERIA_COUNT,
                                      iterations, eps))
        warped = cv2.warpAffine(im, M, (w,h),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned.append(warped)
    return aligned

def align_orb(imgs, keep_kp=5000, match_ratio=0.9):
    ref = imgs[0]
    h, w = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(keep_kp)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    aligned = [ref]
    for im in tqdm(imgs[1:], desc="Aligning ORB"):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        keep = matches[:int(len(matches)*match_ratio)]
        src = np.float32([kp1[m.queryIdx].pt for m in keep]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in keep]).reshape(-1,1,2)
        M, _ = cv2.findHomography(dst, src, cv2.RANSAC)
        warped = cv2.warpPerspective(im, M, (w,h))
        aligned.append(warped)
    return aligned

def stack_images(imgs, method='average'):
    arr = np.stack([im.astype(np.float32) for im in imgs], axis=0)
    if method == 'median':
        res = np.median(arr, axis=0)
    else:
        res = np.mean(arr, axis=0)
    return np.clip(res, 0, 255).astype(np.uint8)

def unsharp_mask(img, ksize=(5,5), sigma=1.0, amount=1.5, thresh=0):
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    sharp = float(amount+1)*img - float(amount)*blurred
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    if thresh > 0:
        low_contrast = np.abs(img - blurred) < thresh
        sharp[low_contrast] = img[low_contrast]
    return sharp

def load_frames_from_ser(path):
    ser = reader(path)  # parses header & metadata
    frames = []
    for i in range(ser.header.frameCount):
        img = ser.getImg(i)  # H×W×numPlanes uint8/uint16

        # If single‐plane, it’s mono (colorID=0) or Bayer (8–11)
        if ser.header.numPlanes == 1:
            cid = ser.header.colorID
            if cid == 0:
                # pure mono → convert to BGR so rest of pipeline sees 3 channels
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif cid in (8,9,10,11):
                # standard Bayer patterns → choose correct demosaic code
                bayer_code = {
                  8:  cv2.COLOR_BAYER_RG2BGR,   # RGGB
                  9:  cv2.COLOR_BAYER_GR2BGR,   # GRBG
                  10: cv2.COLOR_BAYER_GB2BGR,   # GBRG
                  11: cv2.COLOR_BAYER_BG2BGR    # BGGR
                }[cid]
                img = cv2.cvtColor(img, bayer_code)
            else:
                # unknown single‐plane → treat as gray
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # If three planes, it’s already color (100=RGB, 101=BGR)
        else:
            cid = ser.header.colorID
            if cid == 100:
                # convert RGB→BGR
                img = img[..., ::-1]
            # if 101 (BGR) we’re good

        frames.append(img)

    return frames

def main():
    default_input  = 'Input'
    default_output = os.path.join('Output', 'stacked.png')

    p = argparse.ArgumentParser(prog="stackSR",
        description="Stacking & super-res without AI (now .SER-aware)")
    g = p.add_mutually_exclusive_group()
    g.add_argument('--input-folder', help="Folder of images", default=default_input)
    g.add_argument('--input-video',  help="Input video file")
    g.add_argument('--input-ser',    help="Input .ser file")
    p.add_argument('--lap-thresh',  type=float, help="Laplacian variance blur threshold")
    p.add_argument('--top-percent', type=float, help="Keep top P% sharpest frames")
    p.add_argument('--align-method', choices=['ECC','ORB'], default='ECC')
    p.add_argument('--stack-method', choices=['average','median'], default='average')
    p.add_argument('--unsharp-amount', type=float, default=1.5)
    p.add_argument('--output', help="Output filename", default=default_output)
    args = p.parse_args()

    # ensure Output dir exists
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # if user didn’t override --output, derive it from the input name
    if args.output == default_output:
        if args.input_ser:
            base = os.path.splitext(os.path.basename(args.input_ser))[0]
        elif args.input_video:
            base = os.path.splitext(os.path.basename(args.input_video))[0]
        else:
            base = os.path.basename(os.path.normpath(args.input_folder))
        args.output = os.path.join(out_dir, f"{base}.png")

    # Load frames based on input type:
    if args.input_ser:
        frames = load_frames_from_ser(args.input_ser)
    elif args.input_video:
        frames = list(load_frames_from_video(args.input_video))
    else:
        frames = list(load_images_from_folder(args.input_folder))

    if not frames:
        raise RuntimeError("No frames loaded – check your inputs.")
    good = filter_by_quality(frames,
                             lap_thresh=args.lap_thresh,
                             top_percent=args.top_percent)
    if not good:
        raise RuntimeError("All frames filtered out – loosen criteria.")

    aligned = align_ecc(good) if args.align_method=='ECC' else align_orb(good)
    stacked = stack_images(aligned, method=args.stack_method)
    final   = unsharp_mask(stacked, amount=args.unsharp_amount)
    cv2.imwrite(args.output, final)
    print(f"Done → {args.output}")

if __name__ == '__main__':
    main()