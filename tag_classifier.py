import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

class TagClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def preprocess_tags(self, tags):
        """Convert tags list to string"""
        cleaned_tags = [tag.strip() for tag in tags if tag.strip()]
        return " [SEP] ".join(cleaned_tags)
    
    def load_data(self, json_file):
        """Load data from JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item_id, item_data in data.items():
            tags = item_data.get('tags', [])
            label = item_data.get('label', 0)
            
            if tags:
                processed_text = self.preprocess_tags(tags)
                texts.append(processed_text)
                labels.append(label)
        
        return texts, labels
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train the model"""
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)
        
        # Tokenize training data
        train_encodings = self.tokenizer(
            train_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Create dataset
        class TagDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
                
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
                
            def __len__(self):
                return len(self.encodings.input_ids)
        
        train_dataset = TagDataset(train_encodings, train_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./tag_model",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            save_strategy="epoch",
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        trainer.train()
        trainer.save_model("./tag_model")
        self.tokenizer.save_pretrained("./tag_model")
        print("Model trained and saved!")
    
    def predict(self, tags):
        """Predict if tags are meaningful"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        text = self.preprocess_tags(tags)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            "tags": tags,
            "prediction": prediction,  # 1 = meaningful, 0 = not meaningful
            "confidence": confidence,
            "is_meaningful": prediction == 1
        }
    
    def load_model(self, model_path="./tag_model"):
        """Load trained model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")

def main():
    # Initialize classifier
    classifier = TagClassifier()
    
    # Load data
    print("Loading data...")
    texts, labels = classifier.load_data("final_groundtruth.json")
    
    print(f"Total samples: {len(texts)}")
    print(f"Meaningful (1): {sum(labels)}")
    print(f"Not meaningful (0): {len(labels) - sum(labels)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    print("Training model...")
    classifier.train(train_texts, train_labels)
    
    # Test examples
    print("\nTesting model...")
    test_cases = [
        ["a group", "people", "children", "graduation gowns", "the stage"],
        ["a group", "people", "a large crowd", "a large group"],
        ["a woman", "a graduation gown", "a black metal railing", "a man"],
        ["a wall", "pictures", "it", "four pictures", "a white wall"]
    ]
    
    for i, tags in enumerate(test_cases):
        result = classifier.predict(tags)
        print(f"\nTest {i+1}:")
        print(f"Tags: {tags}")
        print(f"Prediction: {'Meaningful' if result['is_meaningful'] else 'Not Meaningful'}")
        print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main() 