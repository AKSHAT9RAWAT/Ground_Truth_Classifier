import json
from tag_classifier import TagClassifier

def predict_tag_meaningfulness(tags_list):
    """
    Predict if a list of tags is meaningful.
    
    Args:
        tags_list: List of tag strings
        
    Returns:
        Dictionary with prediction results
    """
    # Load the trained model
    classifier = TagClassifier()
    try:
        classifier.load_model("./tag_model")
    except:
        print("No trained model found. Please run tag_classifier.py first to train the model.")
        return None
    
    # Make prediction
    result = classifier.predict(tags_list)
    return result

def test_clusters_from_json():
    """
    Test all clusters from test_tags.json and output 1/0 predictions
    """
    # Load test tags from JSON file
    try:
        with open('a.json', 'r') as f:
            test_clusters = json.load(f)
    except FileNotFoundError:
        print("test_tags.json file not found!")
        return
    except json.JSONDecodeError:
        print("Error reading test_tags.json file!")
        return
    
    # Load the trained model
    classifier = TagClassifier()
    try:
        classifier.load_model("./tag_model")
    except:
        print("No trained model found. Please run tag_classifier.py first to train the model.")
        return
    
    print("Testing clusters from test_tags.json...")
    print("=" * 60)
    
    results = {}
    
    for cluster_id, tags in test_clusters.items():
        try:
            # Make prediction
            result = classifier.predict(tags)
            prediction = result['prediction']  # This is 0 or 1
            confidence = result['confidence']
            
            results[cluster_id] = prediction
            
            # Print result
            status = "✅ Meaningful (1)" if prediction == 1 else "❌ Not Meaningful (0)"
            print(f"Cluster {cluster_id}: {status} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"Error predicting for cluster {cluster_id}: {e}")
            results[cluster_id] = -1  # Error indicator
    
    # Save results to file
    output_file = "cluster_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    meaningful_count = sum(1 for pred in results.values() if pred == 1)
    not_meaningful_count = sum(1 for pred in results.values() if pred == 0)
    error_count = sum(1 for pred in results.values() if pred == -1)
    
    print(f"\nSummary:")
    print(f"Meaningful clusters (1): {meaningful_count}")
    print(f"Not meaningful clusters (0): {not_meaningful_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    
    return results

def main():
    # Test clusters from test_tags.json
    test_clusters_from_json()

if __name__ == "__main__":
    main() 