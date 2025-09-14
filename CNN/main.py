import os
import argparse
import torch
import wandb
from datetime import datetime

from models import create_model, get_model_info
from data import get_transforms, load_datasets
from training import train_model
from evaluation import evaluate_model
from benchmarking import benchmark_inference, compare_models
from utils import get_system_info, determine_input_size, setup_output_dirs



torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Image Classification with Various CNN Architectures')
    parser.add_argument('--model', type=str, default='resnet50', 
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
                               'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201',
                               'inception_v3', 'googlenet', 'alexnet', 'mobilenet_v2', 'efficientnet_b0',
                               'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
                      help='CNN architecture to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw','rmsprop'],
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--data_dir', type=str, default='../model_data', help='Data directory')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'both', 'inference_benchmark', 'compare'],
                        help='Mode to run (default: train)')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name for W&B')
    parser.add_argument('--img_size', type=int, default=224, choices=[224, 256, 299, 384, 512],
                        help='Input image size (224, 256, 299 for Inception, 384, 512)')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base model layers')
    parser.add_argument('--benchmark_batches', type=int, default=50, help='Number of batches for benchmarking')
    parser.add_argument('--benchmark_runs', type=int, default=5, help='Number of benchmark runs')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for all files')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--compare_models', type=str, default=None, 
                        help='Comma-separated list of models to compare (for compare mode)')
    # Add this to get_args() function in main.py
    parser.add_argument('--grayscale', action='store_true', 
                        help='Convert images to grayscale (single channel)')
    
    # Add these arguments to your main.py get_args() function
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM visualizations during evaluation')
    parser.add_argument('--gradcam_samples', type=int, default=5, help='Number of samples for Grad-CAM visualization')
    parser.add_argument('--gradcam_correct_and_incorrect', action='store_true', 
                        help='Generate Grad-CAM for both correctly and incorrectly classified samples')

    return parser.parse_args()

def main():
    args = get_args()
    
    # Create output directories
    output_dirs = setup_output_dirs(args.output_dir,args.mode)
    
    # Print system information
    system_info = get_system_info()
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    if args.mode == 'train' or args.mode == 'both':
        print("\n=== Starting Training ===")
        print(f"Model: {args.model}")
        print(f"Data Augmentation: {'Enabled' if args.augment else 'Disabled'}")
        print(f"Image Size: {args.img_size}x{args.img_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Optimizer: {args.optimizer}")
        print(f"Learning Rate: {args.lr}")
        print(f"Freeze Base: {args.freeze_base}")
        print(f"Mixed Precision: {args.mixed_precision}")
        
        model, device, class_names, results = train_model(args, output_dirs)
        
        if args.mode == 'both':
            print("\n=== Starting Evaluation ===")
            evaluate_model(args, model, device, class_names, output_dirs)
    
    elif args.mode == 'evaluate':
        print("\n=== Starting Evaluation ===")
        print(f"Model: {args.model}")
        print(f"Image Size: {args.img_size}x{args.img_size}")
        evaluate_model(args, output_dirs=output_dirs)
    
    elif args.mode == 'inference_benchmark':
        print("\n=== Starting Inference Benchmark ===")
        print(f"Model: {args.model}")
        print(f"Image Size: {args.img_size}x{args.img_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Benchmark Batches: {args.benchmark_batches}")
        print(f"Benchmark Runs: {args.benchmark_runs}")
        benchmark_inference(args, output_dirs)
    
    elif args.mode == 'compare':
        print("\n=== Comparing Models ===")
        if args.compare_models:
            models_to_compare = args.compare_models.split(',')
            print(f"Comparing models: {models_to_compare}")
            compare_models(args, models_to_compare, output_dirs)
        else:
            print("Please specify models to compare using --compare_models")

if __name__ == '__main__':
    main()