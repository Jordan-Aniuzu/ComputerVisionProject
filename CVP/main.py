#COMPUTER VISION PROJECT ORINGS PROJECT BY JORDAN ANIZU (B00151878)

#AL IMPORTS FOR PROJECTS NO OPEN CV USED SO MANUALLY IMPLEMENTATION
# MARKING POINTS
#Finding the threshold using the image histogram and performing the thresholding (15 marks)
#Performing the binary morphology to close any interior holes (15 marks)
#Implementing connected component labelling to extract the regions (30 marks)
#Analyse the regions to classify the Oring as being a pass or fail (30 marks)
#Overall program structure (measure the image processing time add it to the output image as
#a text annotation). (10 marks

#MARKING POINTS
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


#SEGMENT 3

def perimeter_4(binary_fg):
    # APPROXIMATE PERIMETER BY COUNTING EDGE CONTACTS
    h, w = binary_fg.shape
    perim = 0
    for y in range(h):
        for x in range(w):
            if binary_fg[y, x] == 1:
                if y == 0 or binary_fg[y - 1, x] == 0:
                    perim += 1
                if y == h - 1 or binary_fg[y + 1, x] == 0:
                    perim += 1
                if x == 0 or binary_fg[y, x - 1] == 0:
                    perim += 1
                if x == w - 1 or binary_fg[y, x + 1] == 0:
                    perim += 1
    return int(perim)

def center_of_mass(binary_fg):
    # COMPUTE CENTER OF MASS OF FOREGROUND PIXELS
    ys, xs = np.where(binary_fg == 1)
    if len(xs) == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))

def radial_thickness_stats(binary_fg, cy, cx, angles=180):
    # MEASURE THICKNESS CONSISTENCY AROUND THE RING ACROSS ALL IMAGES 
    h, w = binary_fg.shape
    max_r = int(np.hypot(h, w))

    thicknesses = []
    gaps = 0

    for i in range(angles):
        theta = (2.0 * np.pi * float(i)) / float(angles)
        dy = float(np.sin(theta))
        dx = float(np.cos(theta))

        hits = []
        for r in range(1, max_r):
            y = int(round(cy + dy * float(r)))
            x = int(round(cx + dx * float(r)))
            if not (0 <= y < h and 0 <= x < w):
                break
            if binary_fg[y, x] == 1:
                hits.append(r)

        if len(hits) < 2:
            gaps += 1
            continue

        thicknesses.append(float(hits[-1] - hits[0]))

    if len(thicknesses) == 0:
        return {"mean": 0.0, "std": 0.0, "gaps": int(gaps), "count": 0}

    th = np.array(thicknesses, dtype=np.float32)
    return {
        "mean": float(np.mean(th)),
        "std": float(np.std(th)),
        "gaps": int(gaps),
        "count": int(th.size)
    }

def classify(mask, holes, bbox):
    # RULE BASED PASS FAIL CLASSIFICATION DUE T NO OPEN CV IMPLEMENMTATION
    area = int(np.sum(mask))
    hole_area = int(np.sum(holes))
    perim = perimeter_4(mask)

    miny, minx, maxy, maxx = bbox
    bh = int(maxy - miny + 1)
    bw = int(maxx - minx + 1)

    extent = float(area) / float(bh * bw + 1e-9)
    pa = float(perim) / float(area + 1e-9)

    com = center_of_mass(mask)
    if com is None:
        return "FAIL", ["NO_FOREGROUND"], {"area": area}

    cy, cx = com
    radial = radial_thickness_stats(mask, cy, cx, angles=180)

    total_disc = float(area + hole_area + 1e-9)
    hole_ratio = float(hole_area) / total_disc

    reasons = []

#SEGEMENT4
    if radial["gaps"] > 10: #SEGMANTIAON LIMIT #SUBJECT TO BE CHANGED DUE TO OUTPUT INCOSISTENCIES
        reasons.append("TOO_MANY_GAPS")

    if radial["count"] > 0 and radial["std"] > 3.5:
        reasons.append("THICKNESS_NOT_UNIFORM")

    if extent > 0.75:
        reasons.append("SEGMENTATION_LOOKS_SOLID")
    if extent < 0.10:
        reasons.append("SEGMENTATION_TOO_SMALL")

    if hole_ratio < 0.15:
        reasons.append("HOLE_TOO_SMALL")
    if hole_ratio > 0.75:
        reasons.append("HOLE_TOO_LARGE")

    if pa > 0.25:
        reasons.append("EDGE_TOO_ROUGH")

    info = {
        "area": area,
        "hole_area": hole_area,
        "perim": perim,
        "extent": extent,
        "pa": pa,
        "hole_ratio": hole_ratio,
        "radial_mean": radial["mean"],
        "radial_std": radial["std"],
        "radial_gaps": radial["gaps"]
    }

    if len(reasons) > 0:
        return "FAIL", reasons, info
    return "PASS", reasons, info


def process_image(path):
    # FULL PIPELINE FOR ONE IMAGE
    t0 = time.perf_counter()

    bgr = cv2.imread(path)
    if bgr is None:
        return None, "FAIL", ["READ_ERROR"], 0.0

    gray = to_grayscale(bgr)
    hist = histogram_u8(gray)
    t = otsu_threshold_from_hist(hist)

    binary = threshold_binary(gray, t)

    binary = opening(binary, k=3, iters=1)
    binary = closing(binary, k=5, iters=1)

    ring_mask, biggest = extract_largest_component(binary)
    if biggest is None:
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
        out = bgr.copy()
        cv2.putText(out, "FAIL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(out, "NO RING FOUND", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f"{elapsed_ms:.1f} ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out, "FAIL", ["NO_RING"], elapsed_ms

    filled, holes = fill_holes(ring_mask)

    label, reasons, info = classify(ring_mask, holes, biggest["bbox"])

    elapsed_ms = float((time.perf_counter() - t0) * 1000.0)

    out = bgr.copy()
    miny, minx, maxy, maxx = biggest["bbox"]
    cv2.rectangle(out, (minx, miny), (maxx, maxy), (255, 255, 255), 2)

    color = (0, 255, 0) if label == "PASS" else (0, 0, 255)
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(out, f"{elapsed_ms:.1f} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(
        out,
        f"AREA {info['area']} HOLE {info['hole_area']} EXT {info['extent']:.2f} GAP {info['radial_gaps']}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return out, label, reasons, elapsed_ms

def run_folder(input_dir, output_dir):
    # RUN PIPELINE ON ALL IMAGES IN A FOLDER
    os.makedirs(output_dir, exist_ok=True)

#IMAGE TYPES TO BE ACCEPTED
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()

    results = []
    for name in files:
        in_path = os.path.join(input_dir, name)
        out_img, label, reasons, ms = process_image(in_path)

        if out_img is None:
            continue

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, out_img)

        print(name, label, f"{ms:.1f} ms", reasons)
        results.append((name, label, ms, reasons))

    return results

if __name__ == "__main__": #RUNS BOTH FOLDERS  IN THE MIX
    run_folder(r"C:\Users\jordo\OneDrive\Desktop\CVP\Orings", "Orings2") #./ FOLDER ISSUE ERROR TO BE FIXED (COULNT FIND PATH USED RAW PATH TO READ FILE TO AVOID UNICODE ERROR)









