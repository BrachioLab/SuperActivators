import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embedding_utils import percent_to_layer

def generate_percentage_bins_for_model(model_name, total_layers):
    """
    Generate percentage bins that map to each layer of a model.
    
    Returns a dictionary where keys are layer indices and values are
    the range of percentages that map to that layer.
    """
    layer_to_percent_range = {}
    
    # For each layer, find the percentage range that maps to it
    for layer_idx in range(total_layers):
        min_percent = None
        max_percent = None
        
        # Check all percentages from 1 to 100
        for percent in range(1, 101):
            if percent_to_layer(percent, total_layers) == layer_idx:
                if min_percent is None:
                    min_percent = percent
                max_percent = percent
        
        if min_percent is not None:
            layer_to_percent_range[layer_idx] = {
                "min_percent": min_percent,
                "max_percent": max_percent,
                "percent_range": f"{min_percent}-{max_percent}%"
            }
    
    return layer_to_percent_range

def main():
    # Define models and their total layer counts
    models = {
        "CLIP": 24,  # Standard CLIP ViT-L/14
        "Llama-Vision": 40,  # 32 transformer + 8 global transformer layers
        "Gemma-2": 42,  # For SAE analysis
        "Llama-Text": 32,  # Standard Llama-2 text model
        "Gemma-Text": 28,  # Gemma text model layers
        "Qwen-Text": 32,  # Qwen text model layers
    }
    
    all_mappings = {}
    
    print("Generating percentage-to-layer mappings for all models...\n")
    
    for model_name, total_layers in models.items():
        print(f"\n{model_name} ({total_layers} layers):")
        print("-" * 50)
        
        mapping = generate_percentage_bins_for_model(model_name, total_layers)
        all_mappings[model_name] = {
            "total_layers": total_layers,
            "layer_mappings": mapping
        }
        
        # Print the mapping
        for layer_idx in range(total_layers):
            if layer_idx in mapping:
                range_info = mapping[layer_idx]
                print(f"Layer {layer_idx:2d}: {range_info['percent_range']:>8s} ({range_info['min_percent']:3d}% - {range_info['max_percent']:3d}%)")
        
        # Also print some key percentages for this model
        print(f"\nKey percentages for {model_name}:")
        for key_percent in [25, 50, 75, 81, 92, 100]:
            layer = percent_to_layer(key_percent, total_layers)
            print(f"  {key_percent}% → Layer {layer}")
    
    # Save the mappings
    output_dir = "/workspace/Experiments/Model_Layer_Mappings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    output_file = os.path.join(output_dir, "percentthrumodel_to_layer_mappings.json")
    with open(output_file, 'w') as f:
        json.dump(all_mappings, f, indent=2)
    print(f"\nMappings saved to: {output_file}")
    
    # Also save a human-readable text version
    text_file = os.path.join(output_dir, "percentthrumodel_to_layer_mappings.txt")
    with open(text_file, 'w') as f:
        f.write("Percentage-to-Layer Mappings for All Models\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, model_data in all_mappings.items():
            f.write(f"{model_name} ({model_data['total_layers']} layers)\n")
            f.write("-" * 40 + "\n")
            
            for layer_idx, range_info in model_data['layer_mappings'].items():
                f.write(f"Layer {int(layer_idx):2d}: {range_info['percent_range']:>8s} "
                       f"({range_info['min_percent']:3d}% - {range_info['max_percent']:3d}%)\n")
            
            f.write("\n")
    
    print(f"Human-readable version saved to: {text_file}")
    
    # Create a reverse mapping (percent -> layer) for quick lookup
    reverse_mapping = {}
    for model_name, model_data in all_mappings.items():
        reverse_mapping[model_name] = {}
        for percent in range(1, 101):
            layer = percent_to_layer(percent, model_data['total_layers'])
            reverse_mapping[model_name][percent] = layer
    
    reverse_file = os.path.join(output_dir, "percent_to_layer_lookup.json")
    with open(reverse_file, 'w') as f:
        json.dump(reverse_mapping, f, indent=2)
    print(f"Reverse lookup table saved to: {reverse_file}")

if __name__ == "__main__":
    main()