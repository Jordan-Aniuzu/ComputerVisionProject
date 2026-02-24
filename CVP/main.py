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
#SEGMENT 2`
def opening(binary, k=3, iters=1):
    # OPENING REMOVES SMALL WHITE NOISE
    return dilate(erode(binary, k=k, iters=iters), k=k, iters=iters)

def closing(binary, k=5, iters=1):
    # CLOSING FILLS SMALL HOLES AND GAPS
    return erode(dilate(binary, k=k, iters=iters), k=k, iters=iters)

def connected_components(binary, connectivity=8):
    # LABEL CONNECTED COMPONENTS USING BFS
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    regions = []
    current = 0

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 1 and labels[y, x] == 0:
                current += 1
                q = deque()
                q.append((y, x))
                labels[y, x] = current #X AND Y MIN AND VALUES COMPUTED BELOW 

                area = 0
                miny = y
                maxy = y
                minx = x
                maxx = x

                while q:
                    cy, cx = q.popleft()
                    area += 1

                    if cy < miny:
                        miny = cy
                    if cy > maxy:
                        maxy = cy
                    if cx < minx:
                        minx = cx
                    if cx > maxx:
                        maxx = cx

                    for dy, dx in nbrs:
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                q.append((ny, nx))

                regions.append({
                    "label": current,
                    "area": int(area),
                    "bbox": (int(miny), int(minx), int(maxy), int(maxx))
                })

    return labels, regions

def extract_largest_component(binary):
    # PICK THE LARGEST FOREGROUND REGION AS THE O RING
    labels, regions = connected_components(binary, connectivity=8)
    if len(regions) == 0:
        return np.zeros_like(binary), None
    biggest = max(regions, key=lambda r: r["area"])
    mask = (labels == biggest["label"]).astype(np.uint8)
    return mask, biggest

def fill_holes(binary_fg):
    # FILL HOLES INSIDE THE FOREGROUND OBJECT USING BORDER FLOOD FILL
    h, w = binary_fg.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    for x in range(w):
        if binary_fg[0, x] == 0 and visited[0, x] == 0:
            visited[0, x] = 1
            q.append((0, x))
        if binary_fg[h - 1, x] == 0 and visited[h - 1, x] == 0:
            visited[h - 1, x] = 1
            q.append((h - 1, x))

    for y in range(h):
        if binary_fg[y, 0] == 0 and visited[y, 0] == 0:
            visited[y, 0] = 1
            q.append((y, 0))
        if binary_fg[y, w - 1] == 0 and visited[y, w - 1] == 0:
            visited[y, w - 1] = 1
            q.append((y, w - 1))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        cy, cx = q.popleft()
        for dy, dx in nbrs:
            ny = cy + dy
            nx = cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if visited[ny, nx] == 0 and binary_fg[ny, nx] == 0:
                    visited[ny, nx] = 1
                    q.append((ny, nx))

    holes = ((binary_fg == 0) & (visited == 0)).astype(np.uint8)
    filled = binary_fg.copy()
    filled[holes == 1] = 1
    return filled, holes


