#COMPUTER VISION PROJECT ORINGS PROJECT BY JORDAN ANIZU (B00151878)

#AL IMPORTS FOR PROJECTS NO OPEN CV USED SO MANUALLY IMPLEMENTATION

#SEGMENT 1
import os
import time
import cv2
import numpy as np
from collections import deque

def to_grayscale(bgr):
    # CONVERT BGR IMAGE TO GRAYSCALE USING NUMPY
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)

def histogram_u8(gray):
    # BUILD 256 BIN HISTOGRAM FOR GRAYSCALE IMAGE 
    hist = np.zeros(256, dtype=np.int64)
    flat = gray.ravel()
    for v in flat:
        hist[int(v)] += 1
    return hist

def otsu_threshold_from_hist(hist):
    # COMPUTE OTSU THRESHOLD FROM HISTOGRAM 
    total = int(hist.sum())
    if total <= 0:
        return 128

    sum_total = 0.0
    for i in range(256):
        sum_total += float(i) * float(hist[i])

    sum_b = 0.0
    w_b = 0
    best_t = 128
    best_between = -1.0

    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += float(t) * float(hist[t])
        m_b = sum_b / float(w_b)
        m_f = (sum_total - sum_b) / float(w_f)

        diff = m_b - m_f
        between = float(w_b) * float(w_f) * diff * diff

        if between > best_between:
            best_between = between
            best_t = t

    return int(best_t)

def threshold_binary(gray, t):
    # THRESHOLD AND AUTO PICK FOREGROUND SIDE
    a = (gray <= t).astype(np.uint8)
    b = (gray > t).astype(np.uint8)
    if int(a.sum()) < int(b.sum()):
        return a
    return b

def pad_binary(img, pad):
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

def erode(binary, k=3, iters=1):
    # EROSION WITH SQUARE KERNEL
    if k % 2 == 0:
        raise ValueError("K MUST BE ODD")
    pad = k // 2
    out = binary.copy()

    for _ in range(iters):
        b = pad_binary(out, pad)
        h, w = out.shape
        nxt = np.zeros_like(out)
        for y in range(h):
            for x in range(w):
                window = b[y:y + k, x:x + k]
                nxt[y, x] = 1 if np.all(window == 1) else 0
        out = nxt

    return out

def dilate(binary, k=3, iters=1):
    #DILATION WITH SQUARE KERNEL
    if k % 2 == 0:
        raise ValueError("K MUST BE ODD")
    pad = k // 2
    out = binary.copy()

    for _ in range(iters):
        b = pad_binary(out, pad)
        h, w = out.shape
        nxt = np.zeros_like(out)
        for y in range(h):
            for x in range(w):
                window = b[y:y + k, x:x + k]
                nxt[y, x] = 1 if np.any(window == 1) else 0
        out = nxt

    return out


