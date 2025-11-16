"""
Evaluate FLOPs (floating point operations) for TinySAM models.
Compares full precision vs quantized models.
"""

import torch
import sys
sys.path.append(".")
from tinysam import sam_model_registry

# Import quantization layers for loading quantized model
# The checkpoint was saved with module path 'quantization_layer', 
# but it's actually at 'tinysam.quantization_layer', so we need to alias it
try:
    import tinysam.quantization_layer as quantization_layer
    sys.modules['quantization_layer'] = quantization_layer
except ImportError:
    print("Warning: Could not import quantization_layer")
    pass

def count_flops_fvcore(model, input_shape=(1, 3, 1024, 1024)):
    """Count FLOPs using fvcore library."""
    try:
        from fvcore.nn import FlopCountAnalysis
        
        dummy_input = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        return total_flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("fvcore not installed. Install with: pip install fvcore")
        return None

def count_flops_thop(model, input_shape=(1, 3, 1024, 1024)):
    """Count FLOPs using thop library."""
    try:
        from thop import profile
        
        dummy_input = torch.randn(input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return None

def count_flops_manual(model, input_shape=(1, 3, 1024, 1024)):
    """Manual FLOPs counting - basic estimation."""
    # This is a simplified estimation for comparison
    # TinySAM uses TinyViT encoder which has known FLOPs
    print("Using manual estimation (install fvcore or thop for accurate counting)")
    return None

def evaluate_model_flops(checkpoint_path, model_name):
    """Evaluate FLOPs for a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...")
        if checkpoint_path.endswith('_w8a8.pth'):
            # Quantized model - load directly
            model = torch.load(checkpoint_path, map_location='cpu')
        else:
            # Regular model - use model registry
            model = sam_model_registry['vit_t'](checkpoint=checkpoint_path)
        
        model.eval()
        print("✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params / 1e6:.2f}M")
        
        # Try different FLOPs counting methods
        print("\nCounting FLOPs...")
        print("Note: Counting FLOPs for image encoder only (main computational component)")
        
        # Use image encoder for FLOPs counting (avoids complex input format issues)
        encoder = model.image_encoder
        
        # Try fvcore first (most accurate)
        flops = count_flops_fvcore(encoder)
        if flops is not None:
            print(f"✓ FLOPs (fvcore): {flops:.2f} GFLOPs")
            return flops
        
        # Fall back to thop
        flops = count_flops_thop(encoder)
        if flops is not None:
            print(f"✓ FLOPs (thop): {flops:.2f} GFLOPs")
            return flops
        
        # Manual estimation
        print("⚠ No FLOPs library available. Install fvcore or thop:")
        print("  pip install fvcore")
        print("  OR")
        print("  pip install thop")
        return None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main evaluation function."""
    print("="*60)
    print("TinySAM FLOPs Evaluation")
    print("="*60)
    
    # Model configurations
    models = [
        {
            'name': 'TinySAM (Full Precision)',
            'checkpoint': 'weights/tinysam_42.3.pth',
            'expected_flops': 42.0
        },
        {
            'name': 'Q-TinySAM (Quantized W8A8)',
            'checkpoint': 'weights/tinysam_w8a8.pth',
            'expected_flops': 20.3
        }
    ]
    
    results = {}
    
    # Evaluate each model
    for model_config in models:
        flops = evaluate_model_flops(
            model_config['checkpoint'],
            model_config['name']
        )
        results[model_config['name']] = {
            'measured': flops,
            'expected': model_config['expected_flops']
        }
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<35} {'Measured':<15} {'Expected':<15}")
    print("-"*60)
    
    for model_name, data in results.items():
        measured = f"{data['measured']:.2f} GFLOPs" if data['measured'] else "N/A"
        expected = f"{data['expected']:.2f} GFLOPs"
        print(f"{model_name:<35} {measured:<15} {expected:<15}")
    
    # Calculate speedup
    if all(r['measured'] is not None for r in results.values()):
        full_flops = results['TinySAM (Full Precision)']['measured']
        quant_flops = results['Q-TinySAM (Quantized W8A8)']['measured']
        speedup = full_flops / quant_flops
        print("-"*60)
        print(f"Quantization speedup: {speedup:.2f}x")
        print(f"FLOPs reduction: {(1 - quant_flops/full_flops)*100:.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    main()

