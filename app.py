
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from collections import Counter

app = Flask(__name__)

def get_dominant_colors(image, num_colors=5):
    # Reduz tamanho p/ acelerar
    img = image.copy()
    img.thumbnail((200, 200))
    arr = np.array(img)
    arr = arr.reshape((-1, 3))
    # Converte p/ lista de tuplas
    pixels = [tuple(p) for p in arr]
    most_common = Counter(pixels).most_common(num_colors)
    return [{"color": c, "percent": round(count/len(pixels)*100,2)} for c,count in most_common]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"ok","message":"API do analisador de revestimentos online."})

@app.route("/analisar", methods=["POST"])
def analisar():
    if "image" not in request.files:
        return jsonify({"error": "Envie um arquivo de imagem no campo 'image'."}), 400
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    w,h = img.size
    colors = get_dominant_colors(img)
    return jsonify({
        "width": w,
        "height": h,
        "dominant_colors": colors
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
