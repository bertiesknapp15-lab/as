import torch
from transformers import AutoProcessor, AutoModel
from flask import Flask, request, jsonify
from imageSplit import split_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/siglip-so400m-patch14-384"

CLASSES = [
    "scissors",
    "pencil",
    "calculator",
    "notebook"
]

# ------------- LOAD MODEL (GPU + FP16) -------------
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.eval()

# Encode labels ONCE
with torch.no_grad():
    text_tokens = processor(
        text=CLASSES,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    text_features = model.get_text_features(**text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
# ---------------------------------------------------

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        data = request.json
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image provided"}), 400

        split_result = split_image(image_url)
        if not split_result["status"]:
            return jsonify(split_result), 400

        images = split_result["data"]

        # ---------- IMAGE INFERENCE (BATCHED) ----------
        with torch.no_grad():
            inputs = processor(
                images=images,
                return_tensors="pt"
            ).to(DEVICE)

            img_features = model.get_image_features(**inputs)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            logits = img_features @ text_features.T
            probs = logits.softmax(dim=-1)
        # ----------------------------------------------

        response = []
        for p in probs:
            i = p.argmax().item()
            response.append({
                "score": float(p[i]),
                "label": CLASSES[i]
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030, debug=False)
