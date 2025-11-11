"""
Script để export Siamese-YOLOv8 model sang ONNX format.

Lưu ý: Export ONNX có thể gặp vấn đề với custom layers (Attention Pooling, etc.)
Cần xử lý đặc biệt hoặc viết custom ONNX operators.
"""
import torch
import torch.onnx
from pathlib import Path
from models.siamese_yolo import SiameseYOLOv8


def export_to_onnx(
    model_path: str,
    output_path: str,
    img_size: int = 640,
    device: str = 'cpu'
):
    """
    Export model sang ONNX format.
    
    Args:
        model_path: Đường dẫn đến PyTorch model (.pt)
        output_path: Đường dẫn output ONNX file
        img_size: Kích thước input image
        device: 'cpu' hoặc 'cuda'
    """
    print(f"📦 Loading model from {model_path}...")
    
    # Load model
    model = SiameseYOLOv8(model_size='n', feature_dim=256, num_ref_images=3)
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Dummy inputs
    print(f"🔧 Creating dummy inputs (size: {img_size})...")
    video_frame = torch.randn(1, 3, img_size, img_size).to(device)
    reference_images = torch.randn(1, 3, 3, img_size, img_size).to(device)
    
    # Export
    print(f"📤 Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            (video_frame, reference_images),
            output_path,
            input_names=['video_frame', 'reference_images'],
            output_names=['objectness', 'similarity', 'bbox'],
            dynamic_axes={
                'video_frame': {0: 'batch_size'},
                'reference_images': {0: 'batch_size'},
                'objectness': {0: 'batch_size'},
                'similarity': {0: 'batch_size'},
                'bbox': {0: 'batch_size'}
            },
            opset_version=11,  # ONNX opset version
            do_constant_folding=True,
            verbose=False
        )
        print(f"✅ ONNX model exported to: {output_path}")
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        print("⚠️  Lưu ý: Custom layers (Attention Pooling, etc.) có thể không được hỗ trợ.")
        print("   Cần viết custom ONNX operators hoặc TensorRT plugins.")


def export_to_tensorrt_script(onnx_path: str, engine_path: str):
    """
    Tạo script để compile ONNX sang TensorRT engine.
    
    Args:
        onnx_path: Đường dẫn đến ONNX file
        engine_path: Đường dẫn output TensorRT engine
    """
    script_content = f"""#!/bin/bash
# Script để compile ONNX sang TensorRT engine
# Sử dụng: bash export_tensorrt.sh

ONNX_PATH="{onnx_path}"
ENGINE_PATH="{engine_path}"

# Sử dụng trtexec (NVIDIA TensorRT tool)
trtexec \\
    --onnx=$ONNX_PATH \\
    --saveEngine=$ENGINE_PATH \\
    --fp16 \\
    --workspace=4096 \\
    --verbose

echo "✅ TensorRT engine saved to: $ENGINE_PATH"
"""
    
    script_path = Path("export_tensorrt.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"📝 TensorRT export script created: {script_path}")
    print("   Chạy: bash export_tensorrt.sh trên máy có TensorRT installed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Siamese-YOLOv8 to ONNX")
    parser.add_argument(
        "--model",
        default="runs/siamese_tracking/best.pt",
        help="Path to PyTorch model"
    )
    parser.add_argument(
        "--output",
        default="models/siamese_yolo.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--tensorrt-script",
        action="store_true",
        help="Generate TensorRT export script"
    )
    
    args = parser.parse_args()
    
    # Export to ONNX
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        img_size=args.img_size,
        device=args.device
    )
    
    # Generate TensorRT script if requested
    if args.tensorrt_script:
        engine_path = args.output.replace('.onnx', '.engine')
        export_to_tensorrt_script(args.output, engine_path)


