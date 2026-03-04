#COMPUTER VISION PROJECT ORINGS PROJECT BY JORDAN ANIZU (B00151878)
#AL IMPORTS FOR PROJECTS NO OPEN CV USED SO MANUALLY IMPLEMENTATION

# MARKING POINTS
#Finding the threshold using the image histogram and performing the thresholding (15 marks)
#Performing the binary morphology to close any interior holes (15 marks)
#Implementing connected component labelling to extract the regions (30 marks)
#Analyze the regions to classify the Oring as being a pass or fail (30 marks)
#Overall program structure (measure the image processing time add it to the output image as
#a text annotation). (10 marks
#https://github.com/Jordan-Aniuzu/ComputerVisionProject PROJECT GITHUB LINK
#MARKING POINTS




#SEGMENT 1

#IMPORTS USED THROUGHOUT THE PIPELINE
#OPENCV IS ONLY USED FOR IMREAD / IMWRITE / DRAWING TEXT AND SHAPES
#ALL IMAGE PROCESSING AND ANALYSIS IS DONE MANUALLY WITH RAW PYTHON + NUMPY

import os
import time
import cv2 #imports used in project (noopencv unless process or analyse the image )
import numpy as np
from collections import deque



def to_grayscale(bgr):
    #CONVERT BGR IMAGE TO GRAYSCALE USING NUMPY

    #SPLIT CHANNELS AND CONVERT TO FLOAT FOR SAFE WEIGHTED SUM
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    #FORMULA FOR GRAYSCALE
    gray = 0.114 * b + 0.587 * g + 0.299 * r

    #RETURN UINT8 IMAGE
    return np.clip(gray, 0, 255).astype(np.uint8)



def histogram_u8(gray):
    #BUILDS THE 256 BIN HISTOGRAM FOR GRAYSCALE IMAGE 

    #HISTOGRAM COUNTS HOW MANY PIXELS FALL INTO EACH INTENSITY VALUE 0-255
    hist = np.zeros(256, dtype=np.int64)

    #FLATTEN IMAGE INTO 1D ARRAY FOR SIMPLE LOOPING
    flat = gray.ravel()

    #MANUALLY COUNTAS PIXEL OCCURRENCES
    for v in flat:
        hist[int(v)] += 1

    return hist



def otsu_threshold_from_hist(hist):
    #COMPUTES OTSU THRESHOLD FROM HISTOGRAM 

    #THEOTSU FINDS THE THRESHOLD THAT MAXIMISES BETWEEN-CLASS VARIANCE
    total = int(hist.sum())
    if total <= 0:
        return 128

    #TOTAL WEIGHTED SUM OF ALL INTENSITIES
    sum_total = 0.0
    for i in range(256):
        sum_total += float(i) * float(hist[i])

    sum_b = 0.0
    w_b = 0
    best_t = 128
    best_between = -1.0

    #EVERY POSSIBLE THRESHOLD
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += float(t) * float(hist[t])

        #MEANS OF BACKGROUND AND FOREGROUND
        m_b = sum_b / float(w_b)
        m_f = (sum_total - sum_b) / float(w_f)

        diff = m_b - m_f

        #BETWEEN CLASS VARIANCE
        between = float(w_b) * float(w_f) * diff * diff

        #KEEP THE THRESHOLD WITH MAX BETWEEN-CLASS VARIANCE
        if between > best_between:
            best_between = between
            best_t = t

    return int(best_t)



def threshold_binary(gray, t):
    #THRESHOLD AND AUTO PICK FOREGROUND SIDE

    #CREATE BOTH POSSIBLE FOREGROUND CHOICES
    a = (gray <= t).astype(np.uint8)
    b = (gray > t).astype(np.uint8)

    #PICK THE SMALLER REGION AS FOREGROUND (RING IS USUALLY SMALLER THAN BACKGROUND)
    if int(a.sum()) < int(b.sum()):
        return a

    return b



def pad_binary(img, pad):
    #PAD WITH ZEROS SO KERNEL WINDOWS NEVER GO OUT OF BOUNDS
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)



def erode(binary, k=3, iters=1):
    #EROSION WITH SQUARE KERNEL

    #EROSION REMOVES SMALL FOREGROUND NOISE AND SHRINKS REGIONS
    if k % 2 == 0:
        raise ValueError("K MUST BE ODD")

    pad = k // 2
    out = binary.copy()

    for _ in range(iters):
        b = pad_binary(out, pad)
        h, w = out.shape
        nxt = np.zeros_like(out)

        #SLIDE KXK WINDOW AND REQUIRE ALL ONES FOR THE OUTPUT TO BE 1
        for y in range(h):
            for x in range(w):
                window = b[y:y + k, x:x + k]
                nxt[y, x] = 1 if np.all(window == 1) else 0

        out = nxt

    return out



def dilate(binary, k=3, iters=1):
    #DILATION WITH SQUARE KERNEL

    #DILATION FILLS SMALL GAPS AND EXPANDS REGIONS
    if k % 2 == 0:
        raise ValueError("K MUST BE ODD")

    pad = k // 2
    out = binary.copy()

    for _ in range(iters):
        b = pad_binary(out, pad)
        h, w = out.shape
        nxt = np.zeros_like(out)

        #SLIDE KXK WINDOW AND REQUIRE ANY ONE FOR THE OUTPUT TO BE 1
        for y in range(h):
            for x in range(w):
                window = b[y:y + k, x:x + k]
                nxt[y, x] = 1 if np.any(window == 1) else 0

        out = nxt

    return out




#SEGMENT 2`

#MORPHOLOGICAL COMBINATIONS AND CONNECTED COMPONENT LABELLING



def opening(binary, k=3, iters=1):
    #OPENING REMOVES SMALL WHITE NOISE

    #OPENING = ERODE THEN DILATE
    return dilate(erode(binary, k=k, iters=iters), k=k, iters=iters)



def closing(binary, k=5, iters=1):
    #CLOSING FILLS SMALL HOLES AND GAPS

    #CLOSING = DILATE THEN ERODE
    return erode(dilate(binary, k=k, iters=iters), k=k, iters=iters)



def connected_components(binary, connectivity=8):
    #LABEL CONNECTED COMPONENTS USING BFS

    #THIS IMPLEMENTS THE REGION EXTRACTION REQUIREMENT
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    regions = []
    current = 0

    #CHOOSE 4-CONNECTED OR 8-CONNECTED NEIGHBOURS
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    #SCAN IMAGE FOR NEW UNLABELLED FOREGROUND PIXELS
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

                #BFS FLOOD FILL TO LABEL ALL CONNECTED PIXELS
                while q:
                    cy, cx = q.popleft()
                    area += 1

                    #UPDATE REGION BOUNDING BOX
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

                #STORE REGION STATISTICS FOR LATER SELECTION
                regions.append({
                    "label": current,
                    "area": int(area),
                    "bbox": (int(miny), int(minx), int(maxy), int(maxx))
                })

    return labels, regions



def extract_largest_component(binary):
    #PICK THE LARGEST FOREGROUND REGION AS THE O RING

    #THIS IS BASED ON THE ASSUMPTION THAT THE O RING IS THE BIGGEST CONNECTED OBJECT
    labels, regions = connected_components(binary, connectivity=8)

    if len(regions) == 0:
        return np.zeros_like(binary), None

    biggest = max(regions, key=lambda r: r["area"])
    mask = (labels == biggest["label"]).astype(np.uint8)



    return mask, biggest



def fill_holes(binary_fg):
    #FILL HOLES INSIDE THE FOREGROUND OBJECT USING BORDER FLOOD FILL

    #THIS FINDS BACKGROUND THAT IS CONNECTED TO THE BORDER
    #ANY BACKGROUND NOT CONNECTED TO THE BORDER IS A HOLE INSIDE THE OBJECT
    h, w = binary_fg.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    #START FLOOD FILL FROM ALL BORDER BACKGROUND PIXELS
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

    #FLOOD FILL THROUGH BACKGROUND CONNECTED TO BORDER
    while q:
        cy, cx = q.popleft()
        for dy, dx in nbrs:
            ny = cy + dy
            nx = cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if visited[ny, nx] == 0 and binary_fg[ny, nx] == 0:
                    visited[ny, nx] = 1
                    q.append((ny, nx))

    #HOLES ARE BACKGROUND PIXELS NOT REACHED BY BORDER FLOOD FILL
    holes = ((binary_fg == 0) & (visited == 0)).astype(np.uint8)

    filled = binary_fg.copy()
    filled[holes == 1] = 1

    return filled, holes




#SEGMENT 3

#REGION MEASUREMENTS USED FOR RULE-BASED DEFECT CLASSIFICATION



def perimeter_4(binary_fg):
    #PERIMETER BY COUNTING EDGE CONTACTS

    #COUNTS HOW MANY FOREGROUND PIXELS TOUCH BACKGROUND IN 4 DIRECTIONS
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
    #COMPUTE CENTER OF MASS OF FOREGROUND PIXELS

    #USED AS A SIMPLE APPROXIMATION OF RING CENTER FOR RADIAL SAMPLING
    ys, xs = np.where(binary_fg == 1)

    if len(xs) == 0:
        return None

    return float(np.mean(ys)), float(np.mean(xs))



def radial_thickness_stats(binary_fg, cy, cx, angles=180):
    #MEASURE THICKNESS CONSISTENCY AROUND THE RING ACROSS ALL IMAGES 

    #THIS CASTS RAYS FROM THE CENTER AND MEASURES THICKNESS ALONG EACH DIRECTION
    h, w = binary_fg.shape
    max_r = int(np.hypot(h, w))

    thicknesses = []
    gaps = 0

    for i in range(angles):
        theta = (2.0 * np.pi * float(i)) / float(angles)
        dy = float(np.sin(theta))
        dx = float(np.cos(theta))

        hits = []

        #WALK OUTWARD FROM CENTER AND RECORD WHERE WE HIT FOREGROUND
        for r in range(1, max_r):
            y = int(round(cy + dy * float(r)))
            x = int(round(cx + dx * float(r)))

            if not (0 <= y < h and 0 <= x < w):
                break

            if binary_fg[y, x] == 1:
                hits.append(r)

        #IF LESS THAN 2 HITS THEN THIS RAY DID NOT SEE A FULL RING THICKNESS
        if len(hits) < 2:
            gaps += 1
            continue

        #THICKNESS APPROX = LAST HIT - FIRST HIT
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
    #RULE BASED PASS FAIL CLASSIFICATION DUE T NO OPEN CV IMPLEMENMTATION

    #BASIC REGION MEASUREMENTS
    area = int(np.sum(mask))
    hole_area = int(np.sum(holes))
    perim = perimeter_4(mask)

    #BOUNDING BOX USED FOR EXTENT FEATURE
    miny, minx, maxy, maxx = bbox
    bh = int(maxy - miny + 1)
    bw = int(maxx - minx + 1)

    extent = float(area) / float(bh * bw + 1e-9)
    pa = float(perim) / float(area + 1e-9)

    #CENTER FOR RADIAL ANALYSIS
    com = center_of_mass(mask)
    if com is None:
        return "FAIL", ["NO_FOREGROUND"], {"area": area}

    cy, cx = com
    radial = radial_thickness_stats(mask, cy, cx, angles=180)

    #HOLE RATIO MEASURES HOW LARGE THE INTERIOR VOID IS
    total_disc = float(area + hole_area + 1e-9)
    hole_ratio = float(hole_area) / total_disc

    reasons = []



#SEGEMENT4
    if radial["gaps"] > 10: #SEGMANTIAON LIMIT SUBJECT TO BE CHANGED DUE TO OUTPUT INCOSISTENCIES #UPDATE SEGMENTAION NUMBER OF 10 KEPT
        reasons.append("THERE IS TOO MANY GAPS")

    if radial["count"] > 0 and radial["std"] > 3.5:
        reasons.append("THE THICKNESS IS NOT UNIFORM")

    #REMOVED SEGMENTATION_LOOKS_SOLID FROM FAIL REASONS BECAUSE IT IS NOT A DEFECT TRIGGER


    if extent < 0.10:
        reasons.append("THE SEGMENTATION IS TOO SMALL")

    if hole_ratio < 0.15:
        reasons.append("THE HOLE IS TOO SMALL")
    if hole_ratio > 0.75:   #OUTPUT LOGIC STATMENTS
        reasons.append("THE HOLE IS TOO LARGE")

    if pa > 0.25:
        reasons.append("THE EDGE IS TOO ROUGH")

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
    #FULL PIPELINE FOR ONE IMAGE

    #MEASURE TIME FOR THE FULL PIPELINE TO SATISFY THE STRUCTURE REQUIREMENT
    t0 = time.perf_counter()

    #READ IMAGE USING OPENCV (ALLOWED)
    bgr = cv2.imread(path)
    if bgr is None:
        return None, "FAIL", ["READ_ERROR"], 0.0

    #GRAYSCALE + HISTOGRAM + OTSU THRESHOLD
    gray = to_grayscale(bgr)
    hist = histogram_u8(gray)
    t = otsu_threshold_from_hist(hist)

    #THRESHOLD INTO BINARY
    binary = threshold_binary(gray, t)

    #OPTIONAL CLEANUP USING MORPHOLOGY
    binary = opening(binary, k=3, iters=1)
    binary = closing(binary, k=5, iters=1)

    #EXTRACT THE MAIN O-RING REGION
    ring_mask, biggest = extract_largest_component(binary)

    if biggest is None:
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
        out = bgr.copy()
        cv2.putText(out, "FAIL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(out, "NO RING FOUND", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f"{elapsed_ms:.1f} ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out, "FAIL", ["NO_RING"], elapsed_ms

    #FILL INTERIOR HOLES AND RETURN HOLE MASK
    filled, holes = fill_holes(ring_mask)

    #CLASSIFY USING THE FILLED MASK SO REGION FEATURES ARE CONSISTENT WITH HOLE HANDLING
    label, reasons, info = classify(filled, holes, biggest["bbox"])

    elapsed_ms = float((time.perf_counter() - t0) * 1000.0)

    #OUTPUT IMAGE ANNOTATION USING OPENCV (ALLOWED)
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
    #RUN PIPELINE ON ALL IMAGES IN A FOLDER

    #CREATE OUTPUT DIRECTORY IF IT DOES NOT EXIST
    os.makedirs(output_dir, exist_ok=True)



#IMAGE TYPES TO BE ACCEPTED
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()

    results = []

    #PROCESS EVERY IMAGE AND SAVE ANNOTATED OUTPUT
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


