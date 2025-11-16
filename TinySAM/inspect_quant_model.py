"""
Inspect the quantized model structure to understand the architecture.
"""
import torch
import sys

# Import quantization layers
import tinysam.quantization_layer as quantization_layer
sys.modules['quantization_layer'] = quantization_layer

# Load both models
print("Loading models...")
quant_model = torch.load('weights/tinysam_w8a8.pth', map_location='cpu')
print("✓ Quantized model loaded\n")

# Compare encoders
print("="*60)
print("Quantized Model Image Encoder Structure:")
print("="*60)
print(quant_model.image_encoder)

print("\n" + "="*60)
print("Comparing layer counts:")
print("="*60)

def count_layers(module):
    """Count different types of layers in a module."""
    from collections import defaultdict
    counts = defaultdict(int)
    
    for name, layer in module.named_modules():
        layer_type = type(layer).__name__
        if layer_type not in ['Sequential', 'ModuleList', 'TinyViT']:
            counts[layer_type] += 1
    
    return dict(counts)

quant_counts = count_layers(quant_model.image_encoder)

print("\nQuantized model layers:")
for layer_type, count in sorted(quant_counts.items()):
    print(f"  {layer_type}: {count}")

# Count total parameters
quant_params = sum(p.numel() for p in quant_model.image_encoder.parameters())
print(f"\nTotal parameters: {quant_params / 1e6:.2f}M")

