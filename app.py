from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict
from PIL import Image
import numpy as np, io, cv2
from sklearn.cluster import KMeans

app = FastAPI(title="Analisador de Revestimentos", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def rgb_to_hex(rgb):
    r,g,b = [int(x) for x in rgb]
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def color_label(rgb):
    r,g,b = rgb/255.0
    h = (np.arctan2(np.sqrt(3)*(g-b), 2*r-g-b)+np.pi)/(2*np.pi)
    v = max(r,g,b); s = 0 if v==0 else 1-min(r,g,b)/v
    if v>0.9 and s<0.15: return "branco"
    if v<0.12: return "preto"
    if s<0.15: return "cinza"
    if h<0.08 or h>0.92: return "vermelho"
    if h<0.20: return "laranja/caramelo"
    if h<0.33: return "amarelo/bege"
    if h<0.45: return "verde oliva"
    if h<0.66: return "ciano/azulado"
    if h<0.80: return "azul"
    return "magenta/roxo"

def dominant_colors(img_rgb: np.ndarray, k: int=5) -> List[Dict]:
    h, w, _ = img_rgb.shape
    y0, y1 = int(h*0.2), int(h*0.8)
    x0, x1 = int(w*0.2), int(w*0.8)
    crop = img_rgb[y0:y1, x0:x1].reshape(-1,3)
    n = min(50000, crop.shape[0])
    sample = crop[np.random.choice(crop.shape[0], n, replace=False)]
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(sample)
    centers = km.cluster_centers_
    labels = km.predict(crop)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(counts)[::-1]
    total = counts.sum()
    out=[]
    for idx in order:
        rgb = centers[idx]
        out.append({
            "hex": rgb_to_hex(rgb),
            "rgb": [int(x) for x in rgb],
            "percent": round(100*counts[idx]/total, 2),
            "name": color_label(rgb)
        })
    return out

def grid_analysis(img_gray: np.ndarray):
    g = cv2.equalizeHist(img_gray)
    edges = cv2.Canny(g, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min(img_gray.shape)//6, maxLineGap=6)
    vert, hori = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dx==0 and dy==0: 
                continue
            if dx < dy*0.5: vert.append((x1,y1,x2,y2))
            elif dy < dx*0.5: hori.append((x1,y1,x2,y2))
    def spacings(ls, axis=0):
        if not ls: return None, 0
        coords = sorted([p[axis] for p in [(x1,y1,x2,y2) for x1,y1,x2,y2 in ls]])
        uniq = sorted(list(set(coords)))
        if len(uniq) < 3: return None, len(ls)
        d = np.diff(uniq)
        return float(np.median(d)), len(ls)
    dx_med, n_v = spacings(vert, axis=0)
    dy_med, n_h = spacings(hori, axis=1)
    aspect = None
    mtype = "indefinido"
    if dx_med and dy_med and dx_med>0 and dy_med>0:
        r = dx_med/dy_med
        aspect = round(r,2)
        if 0.8 <= r <= 1.25: mtype = "quadrado"
        elif r < 0.8: mtype = "filetes/horizontal"
        else: mtype = "retangular/vertical"
    return {
        "grid_type": mtype,
        "aspect_ratio": aspect,
        "vertical_lines": n_v,
        "horizontal_lines": n_h,
        "dx_spacing_px": None if dx_med is None else round(dx_med,2),
        "dy_spacing_px": None if dy_med is None else round(dy_med,2)
    }

def finish_guess(img_rgb: np.ndarray):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]/255.0
    bright_ratio = float((v>0.9).mean())
    return "brilhante" if bright_ratio>0.02 else "fosco/natural"

@app.get("/")
def root():
    return {"ok": True, "msg": "API online"}

@app.post("/analisar")
async def analisar(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w,h = img.size
    scale = 1024 / max(w,h)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)))
    arr = np.array(img)
    colors = dominant_colors(arr, k=5)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    grid = grid_analysis(gray)
    finish = finish_guess(arr)
    shape_hint = "revestimento"
    query = f'{shape_hint} {grid["grid_type"]} {colors[0]["name"]} site:leroymerlin.com.br OR site:jvtubos.com.br OR site:portobello.com.br OR site:eliane.com OR site:mercadolivre.com.br'
    return JSONResponse({
        "image_size": {"w": int(img.size[0]), "h": int(img.size[1])},
        "grid": grid,
        "finish_guess": finish,
        "dominant_colors": colors,
        "suggested_query": query
    })
