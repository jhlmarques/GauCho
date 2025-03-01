from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse
import re
import numpy as np

def draw_bboxes_from_series(series, ax, color='blue'):
    polygons = []
    for _, bbox in series.items():
        bbox = re.sub(' +', ' ', bbox)
        bbox = re.sub(r'\[ ?', '', bbox)
        bbox = re.sub(r' ?\]', '', bbox)
        bbox = bbox.split(' ')
        bbox = [float(i) for i in bbox]
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        poly = np.int16(np.array([p1, p2, p3, p4]))
        polygons.append(Polygon(poly))

    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=2)
    ax.add_collection(p)

    return ax

def draw_ellipses_from_series(series, ax, color='blue'):
    ellipses = []
    for _, bbox in series.items():
        bbox = re.sub(' +', ' ', bbox)
        bbox = re.sub(r'\[ ?', '', bbox)
        bbox = re.sub(r' ?\]', '', bbox)
        bbox = bbox.split(' ')
        bbox = [float(i) for i in bbox]
        xc, yc, w, h, ag = bbox[:5]
        ellipsis = Ellipse((xc,yc), w, h, angle=int(np.rad2deg(ag)))
        ellipses.append(ellipsis)

    p = PatchCollection(
        ellipses,
        facecolor='none',
        edgecolors=color,
        linewidths=2)
    ax.add_collection(p)

    return ax