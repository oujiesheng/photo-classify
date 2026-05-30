#!/usr/bin/env python3
"""
照片去重 Web 服务 v3.0
增强：人物、建筑、物件、汽车等类别感知相似度 + 扫描进度接口
"""

import os
import json
import shutil
import hashlib
import time
import math
import threading
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

from PIL import Image, ImageOps, ImageFilter
import imagehash
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

PORT = 5001
BASE_DIR = Path(__file__).parent
PHOTO_DIR = BASE_DIR / "photos"
PHOTO_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
FEATURE_SIZE = (256, 256)
SCAN_JOBS = {}
SCAN_LOCK = threading.Lock()
FEATURE_CACHE = {}

CATEGORY_LABELS = {
    'people': '人物照片',
    'building': '建筑场景',
    'vehicle': '汽车/交通',
    'object': '物件/标签',
    'chart': '图表数据',
    'document': '文档截图',
    'food': '食物美食',
    'other': '其他照片',
}
CATEGORY_ICONS = {
    'people': '👤', 'building': '🏢', 'vehicle': '🚗', 'object': '📦',
    'chart': '📊', 'document': '📄', 'food': '🍜', 'other': '📷'
}

# ============================================================
# 基础工具
# ============================================================
def safe_float(v, default=0.0):
    try:
        if math.isnan(v) or math.isinf(v):
            return default
        return float(v)
    except Exception:
        return default

def normalize_vec(vec):
    arr = np.asarray(vec, dtype=float)
    s = np.linalg.norm(arr)
    return arr / (s + 1e-10)

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))

def corr_similarity(a, b):
    try:
        c = np.corrcoef(a, b)[0, 1]
        return max(0.0, min(1.0, safe_float(c)))
    except Exception:
        return 0.0

def preprocess(img, size=FEATURE_SIZE):
    img = ImageOps.exif_transpose(img)
    if img.mode == 'P':
        img = img.convert('RGBA')
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    gray = img.convert('L').resize(size, Image.LANCZOS)
    rgb = img.resize(size, Image.LANCZOS)
    return gray, rgb

def simple_ssim(a, b):
    a = a.astype(float)
    b = b.astype(float)
    c1 = 6.5025
    c2 = 58.5225
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2) + 1e-10)
    return max(0.0, min(1.0, float(score)))

# ============================================================
# 视觉特征提取
# ============================================================
def edge_maps(gray_arr):
    dx = np.abs(np.diff(gray_arr.astype(float), axis=1))
    dy = np.abs(np.diff(gray_arr.astype(float), axis=0))
    edge = np.zeros_like(gray_arr, dtype=float)
    edge[:, :-1] += dx
    edge[:-1, :] += dy
    edge_norm = np.clip(edge / 255.0, 0, 1)
    return edge, edge_norm

def line_features(edge_norm):
    h, w = edge_norm.shape
    strong = edge_norm > 0.18
    row_ratio = strong.mean(axis=1)
    col_ratio = strong.mean(axis=0)
    horizontal_lines = int(np.sum(row_ratio > 0.32))
    vertical_lines = int(np.sum(col_ratio > 0.32))
    grid_score = min(1.0, (horizontal_lines + vertical_lines) / 28.0)
    horiz_strength = float(np.mean(np.sort(row_ratio)[-10:])) if len(row_ratio) >= 10 else float(row_ratio.mean())
    vert_strength = float(np.mean(np.sort(col_ratio)[-10:])) if len(col_ratio) >= 10 else float(col_ratio.mean())
    return {
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines,
        'grid_score': grid_score,
        'line_strength': min(1.0, horiz_strength + vert_strength),
        'axis_like': horizontal_lines >= 1 and vertical_lines >= 1,
    }

def color_features(rgb_arr):
    hsv = Image.fromarray(rgb_arr.astype('uint8'), 'RGB').convert('HSV')
    hsv_arr = np.asarray(hsv)
    h = hsv_arr[:, :, 0].astype(float)
    s = hsv_arr[:, :, 1].astype(float)
    v = hsv_arr[:, :, 2].astype(float)
    brightness = v / 255.0
    white_ratio = float(np.mean((v > 210) & (s < 55)))
    dark_ratio = float(np.mean(v < 55))
    saturation = float(np.mean(s) / 255.0)
    warm_ratio = float(np.mean(((h <= 35) | (h >= 235)) & (s > 45) & (v > 60)))
    blue_ratio = float(np.mean((h >= 135) & (h <= 180) & (s > 50) & (v > 60)))
    green_ratio = float(np.mean((h >= 55) & (h <= 115) & (s > 45) & (v > 60)))
    gray_ratio = float(np.mean(s < 35))
    hist = []
    for ch in range(3):
        hh, _ = np.histogram(rgb_arr[:, :, ch].flatten(), bins=64, range=(0, 255))
        hist.extend((hh.astype(float) / (hh.sum() + 1e-10)).tolist())
    small = Image.fromarray(rgb_arr.astype('uint8'), 'RGB').resize((24, 24), Image.LANCZOS)
    palette = np.asarray(small).reshape(-1, 3)
    quant = (palette // 32).astype(int)
    unique_colors = len({tuple(x) for x in quant})
    return {
        'hist': normalize_vec(hist).tolist(),
        'white_ratio': white_ratio,
        'dark_ratio': dark_ratio,
        'mid_ratio': float(np.mean((brightness >= 0.22) & (brightness <= 0.82))),
        'bright_ratio': float(np.mean(brightness > 0.75)),
        'avg_brightness': float(brightness.mean()),
        'saturation': saturation,
        'warm_ratio': warm_ratio,
        'blue_ratio': blue_ratio,
        'green_ratio': green_ratio,
        'gray_ratio': gray_ratio,
        'unique_colors': unique_colors,
    }

def skin_masks(rgb_arr):
    r = rgb_arr[:, :, 0].astype(float)
    g = rgb_arr[:, :, 1].astype(float)
    b = rgb_arr[:, :, 2].astype(float)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
    ycbcr_mask = (cb >= 85) & (cb <= 135) & (cr >= 125) & (cr <= 175) & (y >= 60) & (y <= 220)
    hsv = Image.fromarray(rgb_arr.astype('uint8'), 'RGB').convert('HSV')
    hsv_arr = np.asarray(hsv)
    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]
    hsv_mask = (h >= 5) & (h <= 35) & (s >= 40) & (s <= 170) & (v >= 70) & (v <= 240)
    return ycbcr_mask, hsv_mask, v

def connected_skin_features(mask, v_arr):
    h, w = mask.shape
    total = h * w
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    center_largest = 0
    dark_spots = 0
    components = 0
    for yy in range(h):
        for xx in range(w):
            if not mask[yy, xx] or visited[yy, xx]:
                continue
            q = deque([(yy, xx)])
            visited[yy, xx] = True
            pts = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(pts) < 30:
                continue
            components += 1
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            area = len(pts)
            largest = max(largest, area)
            cx = (min(xs) + max(xs)) / 2 / w
            cy = (min(ys) + max(ys)) / 2 / h
            in_center = 0.25 <= cx <= 0.75 and 0.18 <= cy <= 0.78
            if in_center:
                center_largest = max(center_largest, area)
            dark_spots += int(np.sum(v_arr[ys, xs] < 80))
    return {
        'largest_region': largest / total,
        'center_region': center_largest / total,
        'dark_spot_ratio': dark_spots / total,
        'components': components,
    }

def category_features(path):
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key in FEATURE_CACHE:
        return FEATURE_CACHE[key]

    img = Image.open(path)
    original_w, original_h = img.size
    gray, rgb = preprocess(img)
    gray_arr = np.asarray(gray)
    rgb_arr = np.asarray(rgb)
    edge, edge_norm = edge_maps(gray_arr)
    lines = line_features(edge_norm)
    colors = color_features(rgb_arr)
    ymask, hmask, v_arr = skin_masks(rgb_arr)
    skin = connected_skin_features(ymask & hmask, v_arr)

    edge_density = float(np.mean(edge_norm > 0.16))
    variance = float(np.var(gray_arr))
    aspect = original_w / original_h if original_h else 1.0
    portrait = original_h > original_w * 1.12
    landscape = original_w > original_h * 1.18

    # 人物 v5 评分
    y_ratio = float(ymask.mean())
    h_ratio = float(hmask.mean())
    people_score = 0
    people_reject = y_ratio < 0.10 or h_ratio < 0.05 or skin['largest_region'] < 0.03
    if not people_reject:
        people_score += 25 if y_ratio > 0.20 else 10
        people_score += 25 if h_ratio > 0.15 else 10
        people_score += 25 if skin['center_region'] > 0.05 else 10
        if skin['dark_spot_ratio'] > 0.02:
            people_score += 15
        if portrait:
            people_score += 10
        if 0.2 <= colors['avg_brightness'] <= 0.8:
            people_score += 5
        if variance < 400:
            people_score -= 20
        if edge_density > 0.25:
            people_score -= 15
        if colors['avg_brightness'] < 0.12:
            people_score -= 15
        if colors['avg_brightness'] > 0.88:
            people_score -= 10
        if landscape and y_ratio < 0.08:
            people_score -= 10
    people_conf = max(0.0, min(1.0, people_score / 80.0))

    chart_conf = 0.0
    if 0.30 <= colors['white_ratio'] <= 0.86:
        chart_conf += 0.24
    if 0.07 <= edge_density <= 0.30:
        chart_conf += 0.22
    if lines['axis_like']:
        chart_conf += 0.22
    chart_conf += min(0.18, colors['unique_colors'] / 260.0)
    if lines['grid_score'] > 0.18:
        chart_conf += 0.14
    if colors['gray_ratio'] > 0.70 and edge_density < 0.10:
        chart_conf -= 0.20
    chart_conf = max(0.0, min(1.0, chart_conf))

    document_conf = 0.0
    if colors['white_ratio'] > 0.55:
        document_conf += 0.35
    if colors['gray_ratio'] > 0.45:
        document_conf += 0.20
    if edge_density > 0.09:
        document_conf += 0.20
    if colors['saturation'] < 0.28:
        document_conf += 0.20
    if chart_conf > 0.60:
        document_conf -= 0.25
    document_conf = max(0.0, min(1.0, document_conf))

    building_conf = 0.0
    if lines['grid_score'] > 0.12:
        building_conf += 0.26
    if lines['vertical_lines'] >= 3:
        building_conf += 0.18
    if lines['horizontal_lines'] >= 3:
        building_conf += 0.14
    if 0.10 <= edge_density <= 0.36:
        building_conf += 0.20
    if colors['gray_ratio'] > 0.24 or colors['blue_ratio'] > 0.08:
        building_conf += 0.10
    if landscape:
        building_conf += 0.08
    if people_conf > 0.65:
        building_conf -= 0.25
    building_conf = max(0.0, min(1.0, building_conf))

    vehicle_conf = 0.0
    bottom = gray_arr[int(gray_arr.shape[0] * 0.55):, :]
    bottom_edges = edge_norm[int(edge_norm.shape[0] * 0.55):, :]
    bottom_edge_density = float(np.mean(bottom_edges > 0.16))
    dark_bottom = float(np.mean(bottom < 65))
    if landscape:
        vehicle_conf += 0.15
    if 0.08 <= bottom_edge_density <= 0.34:
        vehicle_conf += 0.22
    if dark_bottom > 0.08:
        vehicle_conf += 0.18
    if colors['saturation'] > 0.22:
        vehicle_conf += 0.12
    if lines['vertical_lines'] <= 8 and lines['horizontal_lines'] <= 10:
        vehicle_conf += 0.12
    if colors['blue_ratio'] > 0.05 or colors['gray_ratio'] > 0.25:
        vehicle_conf += 0.10
    if people_conf > 0.65 or chart_conf > 0.65 or document_conf > 0.70:
        vehicle_conf -= 0.30
    vehicle_conf = max(0.0, min(1.0, vehicle_conf))

    object_conf = 0.0
    if 0.65 <= aspect <= 1.45:
        object_conf += 0.18
    if colors['white_ratio'] > 0.22:
        object_conf += 0.16
    if 0.08 <= edge_density <= 0.28:
        object_conf += 0.20
    if colors['unique_colors'] < 145:
        object_conf += 0.14
    if colors['saturation'] > 0.12:
        object_conf += 0.12
    if not landscape and not portrait:
        object_conf += 0.10
    if people_conf > 0.55 or chart_conf > 0.65:
        object_conf -= 0.25
    object_conf = max(0.0, min(1.0, object_conf))

    food_conf = 0.0
    if colors['warm_ratio'] > 0.22:
        food_conf += 0.30
    if colors['saturation'] > 0.28:
        food_conf += 0.24
    if colors['mid_ratio'] > 0.45:
        food_conf += 0.14
    if people_conf > 0.50 or document_conf > 0.65:
        food_conf -= 0.22
    food_conf = max(0.0, min(1.0, food_conf))

    confs = {
        'people': people_conf,
        'building': building_conf,
        'vehicle': vehicle_conf,
        'object': object_conf,
        'chart': chart_conf,
        'document': document_conf,
        'food': food_conf,
        'other': 0.18,
    }
    category = max(confs, key=confs.get)
    if confs[category] < 0.45:
        category = 'other'

    hash_vals = []
    for fn in (imagehash.phash, imagehash.average_hash, imagehash.dhash, imagehash.whash):
        try:
            hash_vals.append(str(fn(gray)))
        except Exception:
            hash_vals.append('0')
    for fn in (imagehash.colorhash,):
        try:
            hash_vals.append(str(fn(rgb)))
        except Exception:
            hash_vals.append('0')

    # 空间布局特征：分块亮度、边缘、颜色
    grid = []
    gh, gw = 4, 4
    for yy in range(gh):
        for xx in range(gw):
            ys = slice(yy * 256 // gh, (yy + 1) * 256 // gh)
            xs = slice(xx * 256 // gw, (xx + 1) * 256 // gw)
            grid.append(float(gray_arr[ys, xs].mean()) / 255.0)
            grid.append(float(edge_norm[ys, xs].mean()))
            block = rgb_arr[ys, xs]
            grid.extend((block.mean(axis=(0, 1)) / 255.0).tolist())

    feature = {
        'path': str(path),
        'width': original_w,
        'height': original_h,
        'aspect': aspect,
        'hashes': hash_vals,
        'gray': gray_arr,
        'edge_hist': normalize_vec(np.histogram(edge, bins=64, range=(0, 255))[0]).tolist(),
        'color_hist': colors['hist'],
        'layout': normalize_vec(grid).tolist(),
        'edge_density': edge_density,
        'variance': variance,
        'lines': lines,
        'colors': colors,
        'skin': {'ycbcr_ratio': y_ratio, 'hsv_ratio': h_ratio, **skin},
        'category': category,
        'category_label': CATEGORY_LABELS[category],
        'category_icon': CATEGORY_ICONS[category],
        'category_confidence': round(confs[category] * 100, 1),
        'category_scores': {CATEGORY_LABELS[k]: round(v * 100, 1) for k, v in confs.items() if k != 'other'},
    }
    FEATURE_CACHE[key] = feature
    return feature

# ============================================================
# 相似度计算
# ============================================================
def hash_similarity(hash_a, hash_b):
    vals = []
    for a, b in zip(hash_a, hash_b):
        try:
            ia = imagehash.hex_to_hash(a)
            ib = imagehash.hex_to_hash(b)
            max_bits = len(ia.hash.flatten())
            vals.append(1.0 - ((ia - ib) / max_bits))
        except Exception:
            vals.append(1.0 if a == b else 0.0)
    return max(0.0, min(1.0, float(np.mean(vals)))) if vals else 0.0

def category_similarity(fa, fb):
    ca = fa['category']
    cb = fb['category']
    same = ca == cb
    related = {('document', 'chart'), ('object', 'document'), ('building', 'vehicle')}
    base = 1.0 if same else (0.62 if (ca, cb) in related or (cb, ca) in related else 0.28)
    score_vec_a = [fa['category_scores'].get(CATEGORY_LABELS[k], 0) for k in ['people','building','vehicle','object','chart','document','food']]
    score_vec_b = [fb['category_scores'].get(CATEGORY_LABELS[k], 0) for k in ['people','building','vehicle','object','chart','document','food']]
    return max(0.0, min(1.0, 0.55 * base + 0.45 * cosine_similarity(score_vec_a, score_vec_b)))

def compare_features(fa, fb):
    arr_a = fa['gray']
    arr_b = fb['gray']
    if skimage_ssim:
        try:
            ssim_score = max(0.0, safe_float(skimage_ssim(arr_a, arr_b, data_range=255)))
        except Exception:
            ssim_score = simple_ssim(arr_a, arr_b)
    else:
        ssim_score = simple_ssim(arr_a, arr_b)

    components = {
        'hash': hash_similarity(fa['hashes'], fb['hashes']),
        'color': corr_similarity(fa['color_hist'], fb['color_hist']),
        'structure': ssim_score,
        'edge': corr_similarity(fa['edge_hist'], fb['edge_hist']),
        'layout': cosine_similarity(fa['layout'], fb['layout']),
        'category': category_similarity(fa, fb),
    }

    ca = fa['category']
    cb = fb['category']
    if ca == cb == 'people':
        weights = {'hash': 0.20, 'color': 0.14, 'structure': 0.18, 'edge': 0.12, 'layout': 0.16, 'category': 0.20}
        # 人物误删风险高：类别低或结构低时收紧
        if components['structure'] < 0.45 and components['hash'] < 0.62:
            components['category'] *= 0.80
    elif ca == cb == 'building':
        weights = {'hash': 0.18, 'color': 0.12, 'structure': 0.20, 'edge': 0.20, 'layout': 0.18, 'category': 0.12}
    elif ca == cb == 'vehicle':
        weights = {'hash': 0.22, 'color': 0.18, 'structure': 0.18, 'edge': 0.14, 'layout': 0.16, 'category': 0.12}
    elif ca == cb == 'object':
        weights = {'hash': 0.22, 'color': 0.20, 'structure': 0.16, 'edge': 0.16, 'layout': 0.14, 'category': 0.12}
    elif ca == cb == 'chart' or ca == cb == 'document':
        weights = {'hash': 0.25, 'color': 0.12, 'structure': 0.25, 'edge': 0.18, 'layout': 0.12, 'category': 0.08}
    else:
        weights = {'hash': 0.27, 'color': 0.19, 'structure': 0.19, 'edge': 0.13, 'layout': 0.12, 'category': 0.10}

    total = sum(components[k] * weights[k] for k in weights)
    if ca != cb:
        # 不同大类不要轻易合并，避免人物/汽车/建筑互相误判
        total *= 0.90 if components['category'] > 0.55 else 0.78
    return max(0.0, min(1.0, total)), {k: round(v * 100, 1) for k, v in components.items()}

def compute_similarity(img_path_a, img_path_b):
    try:
        fa = category_features(Path(img_path_a))
        fb = category_features(Path(img_path_b))
        score, _ = compare_features(fa, fb)
        return score
    except Exception:
        return 0.0

# ============================================================
# 扫描
# ============================================================
def _files_info(paths, feature_map=None):
    result = []
    for f in paths:
        try:
            img = Image.open(f)
            img = ImageOps.exif_transpose(img)
            size = f"{img.size[0]}x{img.size[1]}"
            file_size = f.stat().st_size / 1024
        except Exception:
            size = "unknown"
            file_size = 0
        feat = (feature_map or {}).get(str(f))
        item = {
            'path': str(f),
            'name': f.name,
            'size': size,
            'file_size_kb': round(file_size, 1),
            'is_dir': False,
        }
        if feat:
            item.update({
                'category': feat['category_label'],
                'category_icon': feat['category_icon'],
                'category_confidence': feat['category_confidence'],
            })
        result.append(item)
    return result

def choose_keep(paths):
    def rank(p):
        try:
            img = Image.open(p)
            pixels = img.size[0] * img.size[1]
        except Exception:
            pixels = 0
        return (pixels, p.stat().st_size, -len(p.name))
    return sorted(paths, key=rank, reverse=True)[0]

def scan_folder(folder_path, threshold=0.70, progress_cb=None):
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        return {'error': f'文件夹不存在: {folder_path}', 'groups': []}

    images = [f for f in sorted(folder.rglob('*')) if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]
    if not images:
        return {'error': f'文件夹内没有图片: {folder_path}', 'groups': []}

    if progress_cb:
        progress_cb(5, f'发现 {len(images)} 张图片，正在读取指纹...')

    md5_map = defaultdict(list)
    for idx, img_path in enumerate(images):
        try:
            md5_map[hashlib.md5(img_path.read_bytes()).hexdigest()].append(img_path)
        except Exception:
            pass
        if progress_cb and idx % 20 == 0:
            progress_cb(5 + int(15 * (idx + 1) / max(1, len(images))), f'MD5 检查 {idx + 1}/{len(images)}')

    feature_map = {}
    for idx, img_path in enumerate(images):
        try:
            feature_map[str(img_path)] = category_features(img_path)
        except Exception as e:
            print(f"feature error {img_path}: {e}")
        if progress_cb and (idx % 5 == 0 or idx == len(images) - 1):
            progress_cb(20 + int(30 * (idx + 1) / max(1, len(images))), f'类别识别 {idx + 1}/{len(images)}')

    groups = []
    for _, paths in md5_map.items():
        if len(paths) > 1:
            keep = choose_keep(paths)
            ordered = [keep] + [p for p in paths if p != keep]
            groups.append({
                'type': '相同文件',
                'similarity': 100.0,
                'category': feature_map.get(str(keep), {}).get('category_label', '未知'),
                'category_icon': feature_map.get(str(keep), {}).get('category_icon', '📷'),
                'details': {'hash': 100, 'color': 100, 'structure': 100, 'edge': 100, 'layout': 100, 'category': 100},
                'files': _files_info(ordered, feature_map),
                'keep': str(keep),
                'to_delete': [str(p) for p in ordered[1:]],
            })

    grouped_files = {finfo['path'] for g in groups for finfo in g['files']}
    remaining = [p for p in images if str(p) not in grouped_files]
    n = len(remaining)
    total_pairs = n * (n - 1) // 2
    done_pairs = 0

    if n > 1:
        processed = set()
        for i in range(n):
            if str(remaining[i]) in processed:
                continue
            similar_to_this = [remaining[i]]
            scores_to_this = []
            details_to_this = []
            for j in range(i + 1, n):
                if str(remaining[j]) in processed:
                    done_pairs += 1
                    continue
                fa = feature_map.get(str(remaining[i]))
                fb = feature_map.get(str(remaining[j]))
                if not fa or not fb:
                    score, details = 0.0, {}
                else:
                    score, details = compare_features(fa, fb)
                done_pairs += 1
                if score >= threshold:
                    similar_to_this.append(remaining[j])
                    scores_to_this.append(score)
                    details_to_this.append(details)
                    processed.add(str(remaining[j]))
                if progress_cb and (done_pairs % 20 == 0 or done_pairs == total_pairs):
                    pct = 50 + int(45 * done_pairs / max(1, total_pairs))
                    progress_cb(pct, f'相似度比较 {done_pairs}/{total_pairs}')
            if len(similar_to_this) > 1:
                keep = choose_keep(similar_to_this)
                ordered = [keep] + [p for p in similar_to_this if p != keep]
                avg_sim = np.mean(scores_to_this) * 100
                avg_details = {}
                if details_to_this:
                    for k in ['hash', 'color', 'structure', 'edge', 'layout', 'category']:
                        vals = [d.get(k, 0) for d in details_to_this]
                        avg_details[k] = round(float(np.mean(vals)), 1)
                base_feat = feature_map.get(str(ordered[0]), {})
                groups.append({
                    'type': '相似照片',
                    'similarity': round(float(avg_sim), 1),
                    'category': base_feat.get('category_label', '未知'),
                    'category_icon': base_feat.get('category_icon', '📷'),
                    'details': avg_details,
                    'files': _files_info(ordered, feature_map),
                    'keep': str(keep),
                    'to_delete': [str(p) for p in ordered[1:]],
                })
                processed.add(str(remaining[i]))

    categories = defaultdict(int)
    for feat in feature_map.values():
        categories[feat['category_label']] += 1

    groups.sort(key=lambda g: (-len(g['files']), -g['similarity']))
    if progress_cb:
        progress_cb(100, '扫描完成')
    return {
        'total_images': len(images),
        'total_groups': len(groups),
        'threshold': threshold * 100,
        'categories': dict(sorted(categories.items(), key=lambda x: -x[1])),
        'groups': groups,
    }

def start_scan_job(folder, threshold):
    job_id = hashlib.md5(f"{folder}|{threshold}|{time.time()}".encode()).hexdigest()[:12]
    with SCAN_LOCK:
        SCAN_JOBS[job_id] = {'status': 'running', 'progress': 0, 'message': '准备扫描...', 'result': None, 'error': None}

    def update(p, msg):
        with SCAN_LOCK:
            if job_id in SCAN_JOBS:
                SCAN_JOBS[job_id].update({'progress': int(p), 'message': msg})

    def runner():
        try:
            result = scan_folder(folder, threshold, update)
            with SCAN_LOCK:
                SCAN_JOBS[job_id].update({'status': 'done', 'progress': 100, 'message': '扫描完成', 'result': result})
        except Exception as e:
            with SCAN_LOCK:
                SCAN_JOBS[job_id].update({'status': 'error', 'error': str(e), 'message': f'扫描失败: {e}'})

    threading.Thread(target=runner, daemon=True).start()
    return job_id

# ============================================================
# HTTP Server
# ============================================================
class PhotoDedupHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        if path == '/' or path == '/index.html':
            self._serve_file(BASE_DIR / 'index.html', 'text/html')
        elif path == '/style.css':
            self._serve_file(BASE_DIR / 'style.css', 'text/css')
        elif path == '/app.js':
            self._serve_file(BASE_DIR / 'app.js', 'application/javascript')
        elif path == '/listdir':
            self._list_dir(params)
        elif path == '/scan':
            self._scan_folder(params)
        elif path == '/scan_status':
            self._scan_status(params)
        elif path.startswith('/preview/'):
            self._preview_image(path)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b''
        if parsed.path == '/upload':
            self._handle_upload(body)
        elif parsed.path == '/scan':
            self._handle_scan(body)
        elif parsed.path == '/scan_start':
            self._handle_scan_start(body)
        elif parsed.path == '/delete':
            self._handle_delete(body)
        else:
            self.send_error(404)

    def _serve_file(self, filepath, content_type):
        if not filepath.exists():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', f'{content_type}; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(filepath.read_bytes())

    def _list_dir(self, params):
        path = params.get('path', ['/'])[0]
        folder = Path(path).expanduser().resolve()
        if not folder.exists():
            self._json_response({'error': '路径不存在'}, 404)
            return
        items = []
        try:
            for item in sorted(folder.iterdir()):
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    try:
                        list(item.iterdir())
                        items.append({'name': item.name, 'path': str(item), 'is_dir': True})
                    except PermissionError:
                        pass
                elif item.suffix.lower() in SUPPORTED_EXTS:
                    items.append({'name': item.name, 'path': str(item), 'is_dir': False})
        except PermissionError:
            self._json_response({'error': '无权限访问'}, 403)
            return
        self._json_response({'current': str(folder), 'parent': str(folder.parent) if folder != folder.parent else None, 'items': items})

    def _scan_folder(self, params):
        folder = params.get('folder', [''])[0]
        threshold = float(params.get('threshold', ['0.70'])[0])
        if not folder:
            self._json_response({'error': '请提供文件夹路径'}, 400)
            return
        self._json_response(scan_folder(folder, threshold=threshold))

    def _handle_scan(self, body):
        data = json.loads(body or b'{}')
        folder = data.get('folder', '')
        threshold = float(data.get('threshold', 0.70))
        self._json_response(scan_folder(folder, threshold=threshold))

    def _handle_scan_start(self, body):
        data = json.loads(body or b'{}')
        folder = data.get('folder', '')
        threshold = float(data.get('threshold', 0.70))
        if not folder:
            self._json_response({'error': '请提供文件夹路径'}, 400)
            return
        job_id = start_scan_job(folder, threshold)
        self._json_response({'job_id': job_id})

    def _scan_status(self, params):
        job_id = params.get('job_id', [''])[0]
        with SCAN_LOCK:
            job = dict(SCAN_JOBS.get(job_id, {'status': 'missing', 'error': '任务不存在'}))
        self._json_response(job)

    def _handle_upload(self, body):
        ctype = self.headers.get('Content-Type', '')
        if 'boundary=' not in ctype:
            self._json_response({'error': '上传格式错误'}, 400)
            return
        boundary = ctype.split('boundary=')[1]
        parts = body.split(f'--{boundary}'.encode())
        saved = []
        for part in parts:
            if b'filename="' not in part:
                continue
            header_end = part.find(b'\r\n\r\n')
            if header_end == -1:
                continue
            headers = part[:header_end].decode(errors='ignore')
            file_data = part[header_end + 4:]
            if file_data.endswith(b'\r\n'):
                file_data = file_data[:-2]
            import re
            match = re.search(r'filename="([^"]+)"', headers)
            if match:
                filename = Path(match.group(1)).name
                filepath = PHOTO_DIR / filename
                filepath.write_bytes(file_data)
                saved.append(str(filepath))
        self._json_response({'uploaded': len(saved), 'files': saved, 'folder': str(PHOTO_DIR)})

    def _handle_delete(self, body):
        data = json.loads(body or b'{}')
        files = data.get('files', [])
        deleted, errors = 0, []
        trash = Path.home() / '.Trash'
        for fpath in files:
            fpath = Path(fpath)
            if not fpath.exists():
                continue
            try:
                if trash.exists():
                    dest = trash / fpath.name
                    if dest.exists():
                        stem, suffix = fpath.stem, fpath.suffix
                        counter = 1
                        while dest.exists():
                            dest = trash / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.move(str(fpath), str(dest))
                else:
                    fpath.unlink()
                deleted += 1
            except Exception as e:
                errors.append({'file': str(fpath), 'error': str(e)})
        self._json_response({'deleted': deleted, 'errors': errors})

    def _preview_image(self, path):
        file_path = unquote(path.replace('/preview/', '/', 1))
        fpath = Path(file_path).expanduser().resolve()
        if not fpath.exists():
            self.send_error(404)
            return
        self.send_response(200)
        ct_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp'}
        self.send_header('Content-Type', ct_map.get(fpath.suffix.lower(), 'application/octet-stream'))
        self.send_header('Cache-Control', 'max-age=3600')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(fpath.read_bytes())

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {args[0] if args else ''}")

def main():
    server = HTTPServer(('127.0.0.1', PORT), PhotoDedupHandler)
    print("🚀 照片去重服务 v3.0")
    print(f"   http://127.0.0.1:{PORT}")
    print("   类别感知相似度：人物/建筑/物件/汽车/图表/文档")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()

if __name__ == '__main__':
    main()
