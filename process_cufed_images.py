import os
import json
from collections import defaultdict
from PIL import Image
import torch
import spacy
from transformers import AutoProcessor, VisionEncoderDecoderModel, pipeline

# Load spaCy model for extracting noun chunks (tags)
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face image captioning pipeline
caption_pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Directory where CUFED images are located
image_dir = os.path.join("cufed_images", "CUFED5")

# JSON Output Files
image_tag_file = "vitgpt2_image_tags.json"
cluster_tag_file = "vitgpt2_cluster_tags.json"

# Containers
image_tags = {}
cluster_tags = defaultdict(list)

# Extract noun phrase tags from caption
def extract_tags_from_caption(caption):
    doc = nlp(caption)
    tags = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
    tags = [tag.replace(".", "").replace(",", "") for tag in tags]
    return list(dict.fromkeys(tags))[:7]  # Max 7 unique tags

# Get first N cluster IDs
def get_first_n_clusters(image_dir, n=50):
    clusters = set()
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            cluster_id = filename.split("_")[0]
            clusters.add(cluster_id)
            if len(clusters) >= n:
                break
    return sorted(list(clusters))

first_n_clusters = set(get_first_n_clusters(image_dir, 35))

# Inference loop
for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    cluster_id = filename.split("_")[0]
    if cluster_id not in first_n_clusters:
        continue

    img_path = os.path.join(image_dir, filename)
    image = Image.open(img_path).convert("RGB")

    # Generate caption
    result = caption_pipe(image)
    caption = result[0]['generated_text'].strip().lower()

    # Extract tags
    tags = extract_tags_from_caption(caption)
    image_tags[filename] = tags
    cluster_tags[cluster_id].extend(tags)

# Deduplicate tags in each cluster
for cid in cluster_tags:
    cluster_tags[cid] = list(dict.fromkeys(cluster_tags[cid]))

# Save JSON outputs
with open(image_tag_file, "w") as f:
    json.dump(image_tags, f, indent=2)

with open(cluster_tag_file, "w") as f:
    json.dump(cluster_tags, f, indent=2)

print(f"✅ Done. Tags saved to:\n- {image_tag_file}\n- {cluster_tag_file}")
