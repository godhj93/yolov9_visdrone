import torch_tensorrt
import torch
from termcolor import colored

def main():
    
    weights = 'runs/train/yolov9-c/weights/yolov9-c-converted_ptq.jit'
    model = torch.jit.load(weights)
    print(colored(f"Model loaded from {weights}", 'green'))

    B, C, H, W = (1, 3, 382, 674)
    print(colored(f"Input shape: {B, C, H, W}", 'green'))

    print(colored(f"Compiling the model to TensorRT", 'green'))
    trt_mod = torch_tensorrt.compile(
            model, 
            inputs=[torch.randn((1, C, H, W)).to('cuda')],
            enabled_precisions={torch.int8},  # Use int8 precision
            truncate_long_and_double=True,
        )
    
    print(colored(f"Model compiled to TensorRT", 'green'))
    # Get GPU name
    gpu_name = torch.cuda.get_device_name(0)
    weights_path = weights.replace('.jit', f'_qat_{gpu_name}.torchscript').replace(' ', '_')
    torch.jit.save(trt_mod, weights_path)
    print(colored(f"The model saved to {weights_path}", 'green'))


if __name__ == "__main__":
    
    main()