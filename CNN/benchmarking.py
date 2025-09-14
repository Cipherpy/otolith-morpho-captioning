import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast
from tqdm import tqdm
import glob
from tabulate import tabulate

from models import create_model, load_model, get_model_info
from data import determine_input_size, get_transforms, load_datasets, create_dataloaders, load_class_names
from utils import get_system_info, create_comparison_table, plot_inference_comparison, plot_model_size_vs_accuracy

def benchmark_inference(args, output_dirs):
    """
    Benchmark inference performance for a model
    """
    # Determine input size
    img_size = determine_input_size(args.model, args.img_size)
    augment_str = "augmented" if args.augment else "no_augment"
    
    print(f"=== Benchmarking {args.model} (size: {img_size}x{img_size}) ===")
    
    # Try to load an existing model if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load class names if available
    class_names_path = os.path.join(output_dirs['models'], f"class_labels_{args.model}.txt")
    class_names = load_class_names(class_names_path)
    
    if class_names is None:
        print("Class names file not found. Using default number of classes (1000).")
        num_classes = 1000
    else:
        num_classes = len(class_names)
    
    # Try to load an existing model first
    model_path = os.path.join(output_dirs['models'], f"{args.model}_{augment_str}_size{img_size}_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dirs['models'], f"{args.model}_{augment_str}_size{img_size}_final.pth")
    
    # If model doesn't exist, create a new one
    if not os.path.exists(model_path):
        print(f"No existing model found. Creating a new {args.model} model.")
        model = create_model(args.model, num_classes, pretrained=True)
    else:
        print(f"Loading model from {model_path}")
        model = load_model(model_path, args.model, num_classes, device)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Get model information
    model_info = get_model_info(model)
    print(f"Model: {args.model}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create fake input data for benchmarking
    print(f"Creating benchmark data with batch size {args.batch_size}...")
    
    # Check if we have real test data
    data_transforms = get_transforms(img_size, augment=False)
    test_data_path = os.path.join(args.data_dir, 'test')
    
    if os.path.exists(test_data_path):
        print(f"Using real test data from {test_data_path}")
        test_datasets = load_datasets(args.data_dir, {'test': data_transforms['test']})
        test_dataloader = create_dataloaders({'test': test_datasets['test']}, args.batch_size)['test']
        use_real_data = True
    else:
        print("Test data not found. Using synthetic data for benchmarking.")
        # Create synthetic data
        dummy_input = torch.randn(args.batch_size, 3, img_size, img_size)
        dummy_input = dummy_input.to(device)
        use_real_data = False
    
    # Warm-up runs
    print("Performing warm-up runs...")
    with torch.no_grad():
        if use_real_data:
            for inputs, _ in test_dataloader:
                inputs = inputs.to(device)
                if args.mixed_precision:
                    with autocast():
                        _ = model(inputs)
                else:
                    _ = model(inputs)
                break  # Only need one batch for warm-up
        else:
            for _ in range(5):
                if args.mixed_precision:
                    with autocast():
                        _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
    
    # Benchmark runs
    print(f"Starting {args.benchmark_runs} benchmark runs with {args.benchmark_batches} batches each...")
    batch_times = []
    
    for run in range(args.benchmark_runs):
        run_batch_times = []
        
        with torch.no_grad():
            if use_real_data:
                # Use real data from test dataloader
                batch_count = 0
                for inputs, _ in tqdm(test_dataloader, desc=f"Run {run+1}/{args.benchmark_runs}"):
                    if batch_count >= args.benchmark_batches:
                        break
                    
                    inputs = inputs.to(device)
                    
                    # Measure inference time
                    start_time = time.time()
                    if args.mixed_precision:
                        with autocast():
                            _ = model(inputs)
                    else:
                        _ = model(inputs)
                    
                    # Synchronize GPU to get accurate timings
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    run_batch_times.append(end_time - start_time)
                    batch_count += 1
            else:
                # Use synthetic data
                for i in tqdm(range(args.benchmark_batches), desc=f"Run {run+1}/{args.benchmark_runs}"):
                    # Measure inference time
                    start_time = time.time()
                    if args.mixed_precision:
                        with autocast():
                            _ = model(dummy_input)
                    else:
                        _ = model(dummy_input)
                    
                    # Synchronize GPU to get accurate timings
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    run_batch_times.append(end_time - start_time)
        
        batch_times.extend(run_batch_times)
    
    # Calculate statistics
    avg_time = np.mean(batch_times)
    std_time = np.std(batch_times)
    min_time = np.min(batch_times)
    max_time = np.max(batch_times)
    
    # Calculate throughput (images per second)
    throughput = args.batch_size / avg_time
    
    # Print results
    print("\n=== Inference Benchmark Results ===")
    print(f"Model: {args.model}")
    print(f"Input size: {img_size}x{img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Device: {device}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Standard deviation: {std_time*1000:.2f} ms")
    print(f"Min inference time: {min_time*1000:.2f} ms")
    print(f"Max inference time: {max_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/second")
    
    # Save results
    benchmark_results = {
        "model": args.model,
        "image_size": img_size,
        "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "device": str(device),
        "avg_inference_time_ms": avg_time * 1000,
        "std_inference_time_ms": std_time * 1000,
        "min_inference_time_ms": min_time * 1000,
        "max_inference_time_ms": max_time * 1000,
        "throughput_images_per_second": throughput,
        "total_params": model_info['total_params'],
        "model_size_mb": model_info['model_size_mb'],
        "system_info": get_system_info(),
        "benchmark_runs": args.benchmark_runs,
        "benchmark_batches": args.benchmark_batches
    }
    
    # Save benchmark results
    benchmark_file = os.path.join(output_dirs['benchmarks'], 
                                 f"benchmark_{args.model}_size{img_size}_bs{args.batch_size}.json")
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    print(f"Benchmark results saved to {benchmark_file}")
    
    # Create a histogram of inference times
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(batch_times) * 1000, bins=30, alpha=0.7)
    plt.axvline(avg_time * 1000, color='r', linestyle='dashed', linewidth=2)
    plt.text(avg_time * 1000 + 0.5, plt.ylim()[1] * 0.9, f'Mean: {avg_time*1000:.2f} ms', 
             color='r', fontsize=12)
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title(f'Inference Time Distribution - {args.model} (size: {img_size}x{img_size}, batch: {args.batch_size})')
    plt.grid(alpha=0.3)
    
    # Save histogram
    hist_file = os.path.join(output_dirs['plots'], 
                           f"benchmark_hist_{args.model}_size{img_size}_bs{args.batch_size}.png")
    plt.savefig(hist_file, dpi=300)
    print(f"Inference time histogram saved to {hist_file}")
    
    return benchmark_results

def compare_models(args, models_to_compare, output_dirs):
    """
    Compare multiple models based on their benchmark and evaluation results
    """
    print(f"Comparing models: {models_to_compare}")
    
    # Find all benchmark and evaluation result files
    benchmark_files = []
    eval_files = []
    
    for model_name in models_to_compare:
        # Find benchmark files
        benchmark_pattern = os.path.join(output_dirs['benchmarks'], f"benchmark_{model_name}_*.json")
        model_benchmark_files = glob.glob(benchmark_pattern)
        benchmark_files.extend(model_benchmark_files)
        
        # Find evaluation files
        eval_pattern = os.path.join(output_dirs['reports'], f"eval_summary_{model_name}_*.json")
        model_eval_files = glob.glob(eval_pattern)
        eval_files.extend(model_eval_files)
    
    print(f"Found {len(benchmark_files)} benchmark files and {len(eval_files)} evaluation files")
    
    # Combine benchmark and evaluation results
    combined_results = []
    
    # Process benchmark results
    for benchmark_file in benchmark_files:
        try:
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            model_name = benchmark_data['model']
            img_size = benchmark_data['image_size']
            batch_size = benchmark_data['batch_size']
            
            result = {
                'model': model_name,
                'image_size': img_size,
                'batch_size': batch_size,
                'avg_inference_time_ms': benchmark_data['avg_inference_time_ms'],
                'throughput_images_per_second': benchmark_data['throughput_images_per_second'],
                'model_size_mb': benchmark_data['model_size_mb'],
                'total_params': benchmark_data['total_params'],
                'mixed_precision': benchmark_data.get('mixed_precision', False)
            }
            
            # Look for matching eval file
            eval_pattern = os.path.join(output_dirs['reports'], f"eval_summary_{model_name}_*_size{img_size}.json")
            matching_eval_files = glob.glob(eval_pattern)
            
            if matching_eval_files:
                # Use the first matching eval file
                with open(matching_eval_files[0], 'r') as f:
                    eval_data = json.load(f)
                
                # Add accuracy metrics
                result['accuracy'] = eval_data.get('test_accuracy', 0)
                result['f1_score'] = eval_data.get('test_f1', 0)
                result['precision'] = eval_data.get('test_precision', 0)
                result['recall'] = eval_data.get('test_recall', 0)
                result['augmentation'] = eval_data.get('augmentation', False)
            
            combined_results.append(result)
            
        except Exception as e:
            print(f"Error processing {benchmark_file}: {e}")
    
    # If no benchmark files found but eval files exist
    if not combined_results and eval_files:
        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                model_name = eval_data['model']
                img_size = eval_data['image_size']
                
                result = {
                    'model': model_name,
                    'image_size': img_size,
                    'accuracy': eval_data.get('test_accuracy', 0),
                    'f1_score': eval_data.get('test_f1', 0),
                    'precision': eval_data.get('test_precision', 0),
                    'recall': eval_data.get('test_recall', 0),
                    'avg_inference_time_ms': eval_data.get('avg_inference_time', 0) * 1000,
                    'throughput_images_per_second': eval_data.get('samples_per_second', 0),
                    'augmentation': eval_data.get('augmentation', False)
                }
                
                combined_results.append(result)
                
            except Exception as e:
                print(f"Error processing {eval_file}: {e}")
    
    if not combined_results:
        print("No results found for comparison. Please run benchmarks or evaluations first.")
        return
    
    # Create DataFrame and sort by accuracy (if available), then by throughput
    df = pd.DataFrame(combined_results)
    if 'accuracy' in df.columns:
        df = df.sort_values(['accuracy', 'throughput_images_per_second'], ascending=[False, False])
    else:
        df = df.sort_values('throughput_images_per_second', ascending=False)
    
    # Save comparison to CSV
    comparison_file = os.path.join(output_dirs['csv'], "model_comparison.csv")
    df.to_csv(comparison_file, index=False)
    print(f"Comparison saved to {comparison_file}")
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    table_data = []
    columns = ['Model', 'Size', 'Accuracy', 'F1', 'Inf. Time (ms)', 'Throughput (img/s)', 'Model Size (MB)']
    
    for _, row in df.iterrows():
        model_entry = [
            row['model'],
            f"{row['image_size']}px",
            f"{row.get('accuracy', 'N/A'):.4f}" if 'accuracy' in row and not pd.isna(row['accuracy']) else "N/A",
            f"{row.get('f1_score', 'N/A'):.4f}" if 'f1_score' in row and not pd.isna(row['f1_score']) else "N/A",
            f"{row['avg_inference_time_ms']:.2f}",
            f"{row['throughput_images_per_second']:.2f}",
            f"{row['model_size_mb']:.2f}"
        ]
        table_data.append(model_entry)
    
    print(tabulate(table_data, headers=columns, tablefmt="grid"))
    
    # Create comparison plots
    if 'accuracy' in df.columns:
        # Plot accuracy vs inference time
        plt.figure(figsize=(12, 8))
        for i, row in df.iterrows():
            plt.scatter(row['avg_inference_time_ms'], row.get('accuracy', 0), 
                       s=row['model_size_mb']/5, alpha=0.7)
            plt.annotate(f"{row['model']}\n{row['image_size']}px", 
                        (row['avg_inference_time_ms'], row.get('accuracy', 0)),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy vs. Inference Time')
        plt.grid(alpha=0.3)
        
        accuracy_time_file = os.path.join(output_dirs['plots'], "comparison_accuracy_vs_time.png")
        plt.savefig(accuracy_time_file, dpi=300)
        print(f"Accuracy vs. inference time plot saved to {accuracy_time_file}")
        
        # Plot accuracy vs model size
        plt.figure(figsize=(12, 8))
        for i, row in df.iterrows():
            plt.scatter(row['model_size_mb'], row.get('accuracy', 0), 
                       s=row['throughput_images_per_second'], alpha=0.7)
            plt.annotate(f"{row['model']}\n{row['image_size']}px", 
                        (row['model_size_mb'], row.get('accuracy', 0)),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy vs. Size')
        plt.grid(alpha=0.3)
        
        accuracy_size_file = os.path.join(output_dirs['plots'], "comparison_accuracy_vs_size.png")
        plt.savefig(accuracy_size_file, dpi=300)
        print(f"Accuracy vs. model size plot saved to {accuracy_size_file}")
    
    # Plot throughput comparison
    plt.figure(figsize=(12, 8))
    models = df['model'] + "\n" + df['image_size'].astype(str) + "px"
    plt.barh(models, df['throughput_images_per_second'])
    plt.xlabel('Throughput (images/second)')
    plt.title('Model Throughput Comparison')
    plt.grid(alpha=0.3, axis='x')
    
    throughput_file = os.path.join(output_dirs['plots'], "comparison_throughput.png")
    plt.savefig(throughput_file, dpi=300)
    print(f"Throughput comparison plot saved to {throughput_file}")
    
    return df
