from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json
import re
from collections import defaultdict

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Extract caption from image
def extract_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Convert caption to tags
def caption_to_tags(caption):
    stopwords = {"the", "a", "an", "in", "on", "with", "and", "of", "to", "at", "for", "from"}
    tokens = caption.lower().replace(".", "").split()
    tags = [word.strip(",") for word in tokens if word not in stopwords]
    return list(set(tags))

# Process images and generate both individual and cluster tags
def extract_tags(folder_path, start=50, end=125):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".png"]
    individual_tags = {}
    cluster_tags = defaultdict(set)  # album prefix → set of tags

    for filename in sorted(os.listdir(folder_path)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(filename)[0]

            # Extract 3-digit album prefix (e.g., '050' from '050_3.png')
            match = re.match(r"^(\d{3})_", base_name)
            if not match:
                continue
            album_prefix = match.group(1)
            file_index = int(album_prefix)

            if start <= file_index <= end:
                image_path = os.path.join(folder_path, filename)
                print(f"Processing {filename}...")

                try:
                    caption = extract_caption(image_path)
                    tags = caption_to_tags(caption)
                    individual_tags[filename] = tags
                    cluster_tags[album_prefix].update(tags)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Convert sets to sorted lists
    cluster_tags = {k: sorted(list(v)) for k, v in cluster_tags.items()}
    return individual_tags, cluster_tags

# Main
if __name__ == "__main__":
    folder = "cufed_images\CUFED5"  # Update this path if needed
    individual_tags, cluster_tags = extract_tags(folder, start=50, end=125)

    # Save individual image tags
    with open("full_individual_tags.json", "w") as f:
        json.dump(individual_tags, f, indent=2)

    # Save cluster tags (per album)
    with open("full_cluster_tags.json", "w") as f:
        json.dump(cluster_tags, f, indent=2)

    print("\n✅ Tag extraction complete!")
