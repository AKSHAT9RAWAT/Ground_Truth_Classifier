print("Script started.")

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Load your cluster tags (BLIP output)
    print("Loading cluster tags from blip_cluster_tags.json...")
    with open("blip_cluster_tags.json", "r") as f:
        cluster_tags = json.load(f)
    print(f"Loaded {len(cluster_tags)} clusters total.")

    # Limit to only the first 10 clusters
    all_cids = sorted(cluster_tags.keys())[:10]
    print(f"Processing first 10 cluster IDs: {all_cids}")

    # Load the model
    model_name = "microsoft/phi-2"
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model loaded.")

    def make_prompt(tag_list):
        tag_str = ', '.join(tag_list)
        return ("You are an assistant that classifies clusters of image tags as either "
    "story-worthy (label: 1) or not story-worthy (label: 0).\n\n"

    "Only label as story-worthy (1) if the tags suggest a clear, emotionally significant or memorable event, such as:\n"
    "- Celebrations (e.g., birthdays, weddings, graduations)\n"
    "- Performances or public events (e.g., concerts, sports matches)\n"
    "- Activities involving pets or children\n"
    "- Personal or family moments (e.g., reunions, special dinners)\n"
    "- Emotional experiences (e.g., hugging, crying, playing)\n\n"

    "Do NOT label as story-worthy (0) if:\n"
    "- The tags just describe a generic group of people with no activity or context\n"
    "- The scene is a random object, crowd, architecture, or generic setting\n"
    "- There is no sign of a personal, emotional, or event-based moment\n\n"

    "Tags: [{tag_str}]\n"
    "Answer with ONLY a single digit: 1 (story-worthy) or 0 (not story-worthy). Do not explain."
)

    def label_tags(tags):
        prompt = make_prompt(tags)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=4, temperature=0.0)
        resp = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nModel response: {resp}")
        for char in resp:
            if char == '1':
                return 1
            if char == '0':
                return 0
        print(f"⚠️ Failed to parse label from response: {resp}")
        return -1

    # Run for first 10 clusters
    output = []
    for cid in all_cids:
        tags = cluster_tags[cid]
        print(f"\n🧠 Labeling cluster {cid} with tags: {tags}")
        label = label_tags(tags)
        output.append({
            "cluster_id": cid,
            "tags": tags,
            "label": label
        })

    # Save results
    with open("blip_cluster_labels_phi2_top10.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Labeling for top 10 clusters complete. Results saved to blip_cluster_labels_phi2_top10.json")

except Exception as e:
    print(f"\n❌ Script failed with error: {e}")
