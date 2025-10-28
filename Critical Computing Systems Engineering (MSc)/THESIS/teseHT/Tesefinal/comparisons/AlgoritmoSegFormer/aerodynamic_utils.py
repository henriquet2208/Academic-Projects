# aerodynamic_utils.py

import numpy as np
import cv2
from collections import namedtuple

# Definição de Point como tuplo (x, y)
Point = namedtuple("Point", ["x", "y"])
# Número de pontos para discretização na busca do camber_max
NUMBER_OF_POINTS = 500

def calculate_inclination(p1: Point, p2: Point) -> float:
    return (p2.y - p1.y) / (p2.x - p1.x)

def calculate_point_of_interception(m: float, p: Point) -> float:
    return p.y - m * p.x

def calculate_camber_max_point_by_tan(poly: np.poly1d,
                                      dpoly: np.poly1d,
                                      first: Point,
                                      last: Point) -> Point:
    m = calculate_inclination(first, last)
    start, end = first.x, last.x
    step = (end - start) / NUMBER_OF_POINTS
    closest_x = start
    min_disc = abs(m - dpoly(start))
    x = start
    for _ in range(NUMBER_OF_POINTS + 1):
        disc = abs(m - dpoly(x))
        if disc < min_disc:
            min_disc, closest_x = disc, x
        x += step
    y = poly(closest_x)
    return Point(int(closest_x), int(y))

def calculate_interception_point_camber(max_pt: Point,
                                        m: float,
                                        b: float) -> Point:
    m_perp = -1.0 / m
    b_perp = calculate_point_of_interception(m_perp, max_pt)
    x = (b_perp - b) / (m - m_perp)
    y = m * x + b
    return Point(int(x), int(y))

def calculate_camber(poly: np.poly1d,
                     max_pt: Point,
                     intercept_pt: Point,
                     first: Point,
                     last: Point) -> float:
    chord_len = np.hypot(last.x - first.x, last.y - first.y)
    cam_len = np.hypot(max_pt.x - intercept_pt.x,
                       max_pt.y - intercept_pt.y)
    return cam_len / chord_len * 100.0

def calculate_draft_right(first: Point,
                          last: Point,
                          intercept_pt: Point) -> float:
    chord_len = np.hypot(last.x - first.x, last.y - first.y)
    dist = np.hypot(last.x - intercept_pt.x,
                    last.y - intercept_pt.y)
    return dist / chord_len * 100.0

def calculate_twist(m: float) -> float:
    return np.degrees(np.arctan(abs(m)))


def compute_line_properties(mask: np.ndarray,
                            num_lines: int = 3,
                            poly_deg: int = 3,
                            merge_thresh: float = 75.0,
                            min_cluster_pts: int = 200):
    """
    Retorna exatamente `num_lines` seams, cada uma com:
       'first', 'last', 'max_pt', 'intercept_pt', 'camber', 'draft', 'twist'.

    1) Binariza a máscara.
    2) Fecha micro‑gaps (morphological close).
    3) Extrai todos os fragments via contours.
    4) Faz union‑find reunindo fragments cujos endpoints
       estão a < merge_thresh px.
    5) Forma clusters e para cada cluster:
         - junta pontos,
         - calcula center_x,
         - conta número de pts.
    6) Se existirem ≥ num_lines clusters grandes (>=min_cluster_pts pts),
       escolhe entre eles; senão, escolhe entre todos.
    7) Ordena pela proximidade de center_x ao centro da imagem,
       pega os top num_lines.
    8) Para cada cluster escolhido:
         - define `first` como o pixel de min‑x do cluster,
           e `last` como o pixel de max‑x,
         - ajusta polinômio, calcula camber_max, intercept_pt,
           camber%, draft%, twist°.
    9) Retorna a lista de length num_lines, ordenada top→bottom (por center_y).
    """
    H, W = mask.shape
    bin_mask = (mask == 1).astype(np.uint8) * 255

    # 2) fechar pequenos gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # 3) contours → fragments
    cnts, _ = cv2.findContours(bin_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    fragments = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 50:
            continue
        pts = cnt.reshape(-1,2)
        hull = cv2.convexHull(pts).reshape(-1,2)
        xmin, xmax = hull[:,0].min(), hull[:,0].max()
        # y moyenne at those x
        y_min = int(hull[hull[:,0]==xmin,1].mean())
        y_max = int(hull[hull[:,0]==xmax,1].mean())
        f = Point(int(xmin), y_min)
        l = Point(int(xmax), y_max)
        if f.x > l.x:
            f, l = l, f
        fragments.append({"pts": pts, "first": f, "last": l})

    if not fragments:
        return []

    # 4) union-find por proximidade de endpoints
    parent = list(range(len(fragments)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(fragments)):
        for j in range(i+1, len(fragments)):
            f1, l1 = fragments[i]["first"], fragments[i]["last"]
            f2, l2 = fragments[j]["first"], fragments[j]["last"]
            dists = [
                np.hypot(f1.x-f2.x, f1.y-f2.y),
                np.hypot(f1.x-l2.x, f1.y-l2.y),
                np.hypot(l1.x-f2.x, l1.y-f2.y),
                np.hypot(l1.x-l2.x, l1.y-l2.y),
            ]
            if min(dists) < merge_thresh:
                union(i, j)

    # 5) agrupar índices por raiz → clusters
    clusters = {}
    for idx in range(len(fragments)):
        r = find(idx)
        clusters.setdefault(r, []).append(idx)

    # construir infos de cluster (todos e grandes)
    cluster_all = []
    cluster_big = []
    for idxs in clusters.values():
        merged_pts = np.vstack([fragments[i]["pts"] for i in idxs])
        cx = merged_pts[:,0].mean()
        cluster_all.append((idxs, merged_pts, cx))
        if len(merged_pts) >= min_cluster_pts:
            cluster_big.append((idxs, merged_pts, cx))

    # 6) decide entre grandes ou todas
    candidates = cluster_big if len(cluster_big) >= num_lines else cluster_all
    if not candidates:
        return []

    # 7) pick closest to image center in X
    candidates.sort(key=lambda x: abs(x[2] - (W/2)))
    selected = candidates[:num_lines]

    all_seams = []
    for idxs, merged_pts, _ in selected:
        # 8a) chord endpoints = true min‑x / max‑x pixels
        xs = merged_pts[:,0]; ys = merged_pts[:,1]
        i_min = int(np.argmin(xs)); i_max = int(np.argmax(xs))
        first = Point(int(xs[i_min]), int(ys[i_min]))
        last  = Point(int(xs[i_max]), int(ys[i_max]))
        if first.x > last.x:
            first, last = last, first

        # 8b) ajuste polinômial e geometria
        xs_f = merged_pts[:,0].astype(float)
        ys_f = merged_pts[:,1].astype(float)
        coeff = np.polyfit(xs_f, ys_f, deg=poly_deg)
        poly  = np.poly1d(coeff)
        dpoly = poly.deriv()

        m = calculate_inclination(Point(first.x, poly(first.x)),
                                  Point(last.x,  poly(last.x)))
        b = calculate_point_of_interception(m, Point(first.x, poly(first.x)))

        max_pt       = calculate_camber_max_point_by_tan(poly, dpoly, first, last)
        intercept_pt = calculate_interception_point_camber(max_pt, m, b)
        camber_pct   = calculate_camber(poly, max_pt, intercept_pt, first, last)
        draft_pct    = calculate_draft_right(first, last, intercept_pt)
        twist_deg    = calculate_twist(m)

        all_seams.append({
            "coeff": coeff,
            "first": first,
            "last": last,
            "center_y": 0.5*(first.y + last.y),
            "m": m,
            "b": b,
            "max_pt": max_pt,
            "intercept_pt": intercept_pt,
            "camber": camber_pct,
            "draft": draft_pct,
            "twist": twist_deg
        })

    # 9) ordenar top→bottom
    all_seams.sort(key=lambda s: s["center_y"])
    return all_seams
