import torch
from transformers import CLIPModel, CLIPProcessor
from flask import Flask, request, jsonify
from PIL import Image
from imageSplit import split_image

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

CLASSES = [
    "scissors",
    "pencil",
    "calculator",
    "notebook"
]
# ----------------------------------------

# ---------------- MODEL LOAD (ONCE) ----------------
model = CLIPModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(DEVICE)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Encode text ONCE
with torch.no_grad():
    text_inputs = processor(
        text=CLASSES,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    text_features = model.get_text_features(**text_inputs)
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
        
        image_url = image_url.split(",")

        split_results = []

        for imageuri in image_url:
            split_results.append(split_image(imageuri))

            for split_result in split_results:
                if not split_result["status"]:
                    return jsonify(split_result), 400

        listProbs = []

        for split_result in split_results:
            images = split_result["data"]  # MUST be List[PIL.Image]

            # ----------- IMAGE INFERENCE (BATCHED) -----------
            with torch.no_grad():
                image_inputs = processor(
                    images=images,
                    return_tensors="pt"
                ).to(DEVICE)

                image_features = model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = image_features @ text_features.T
                listProbs.append(logits.softmax(dim=-1))
            # -------------------------------------------------

        listResponseJson = []
        for probs in listProbs:
            # SAME response structure as before
            response_json = []
            for prob in probs:
                idx = prob.argmax().item()
                response_json.append({
                    "score": float(prob[idx]),
                    "label": CLASSES[idx]
                })

            listResponseJson.append(response_json)

        
        return jsonify(listResponseJson)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030, debug=False)
