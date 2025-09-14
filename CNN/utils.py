import os
import platform
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
try:
    import GPUtil
except ImportError:
    print("GPUtil not found. GPU detailed information will be limited.")

def get_system_info():
    """Get detailed information about the system for benchmarking"""
    try:
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                system_info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
                
        try:
            if 'GPUtil' in globals():
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    system_info[f"gpu_{i}_memory"] = f"{gpu.memoryTotal}MB"
        except:
            pass
            
        # CPU info
        system_info["cpu_cores"] = psutil.cpu_count(logical=False)
        system_info["cpu_threads"] = psutil.cpu_count(logical=True)
        system_info["ram"] = f"{round(psutil.virtual_memory().total / (1024.0 ** 3))}GB"
        
        return system_info
    except Exception as e:
        print(f"Error gathering system info: {e}")
        return {"error": str(e)}

def determine_input_size(model_name, user_specified_size=224):
    """Determine the appropriate input size for the model"""
    # For inception and googlenet, the default is 299x299
    if model_name in ['inception_v3', 'googlenet'] and user_specified_size == 224:
        return 299
    # For other models, use the user-specified size or default to 224
    return user_specified_size

def setup_output_dirs(base_dir, mode='train'):
    """Create directory structure for all outputs"""
    # Only create a timestamped directory for training mode
    # For other modes, check if there are existing directories to use
    if mode == 'train' or mode == 'both':
        # Create a new timestamped directory for training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(base_dir, timestamp)
        print(f"Creating new output directory: {base_output_dir}")
    else:
        # For evaluate or benchmark modes, use the specified directory directly
        base_output_dir = base_dir
        print(f"Using specified output directory: {base_output_dir}")
    
    dirs = {
        'models': os.path.join(base_output_dir, 'models'),
        'plots': os.path.join(base_output_dir, 'plots'),
        'csv': os.path.join(base_output_dir, 'csv'),
        'reports': os.path.join(base_output_dir, 'reports'),
        'benchmarks': os.path.join(base_output_dir, 'benchmarks')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def plot_comparison(metrics_files, output_path, metric='val_accuracy', title='Model Comparison'):
    """
    Create comparison plots for different models or configurations
    
    Args:
        metrics_files: List of CSV files with metrics
        output_path: Where to save the comparison plot
        metric: Which metric to plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for file_path in metrics_files:
        try:
            df = pd.read_csv(file_path)
            model_name = os.path.basename(file_path).split('_')[2]  # Extract model name from filename
            img_size = file_path.split('size')[1].split('.')[0]  # Extract image size
            augment = 'aug' if 'augmented' in file_path else 'no_aug'
            
            label = f"{model_name} (size: {img_size}, {augment})"
            plt.plot(df['epoch'], df[metric], label=label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    
    return plt

def create_comparison_table(summary_files, output_path):
    """Create a comparison table from model summary files"""
    results = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract relevant info for comparison
            result = {
                'model': data['model'],
                'image_size': data['image_size'],
                'augmentation': data['augmentation'],
                'accuracy': data.get('test_accuracy', data.get('best_val_accuracy', 0)),
                'f1_score': data.get('test_f1', data.get('best_val_f1', 0)),
                'inference_time': data.get('avg_inference_time', 0),
                'samples_per_second': data.get('samples_per_second', 0),
                'model_size_mb': data.get('model_size_mb', 0),
                'total_params': data.get('total_params', 0),
                'trainable_params': data.get('trainable_params', 0),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if results:
        # Create DataFrame and sort by accuracy
        df = pd.DataFrame(results)
        df = df.sort_values('accuracy', ascending=False)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")
        
        return df
    else:
        print("No valid summary files found for comparison")
        return None

def plot_inference_comparison(summary_files, output_path):
    """Create an inference time comparison plot"""
    models = []
    inference_times = []
    throughputs = []
    accuracies = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            model_name = data['model']
            img_size = data['image_size']
            augment = 'aug' if data['augmentation'] else 'no_aug'
            models.append(f"{model_name}\n{img_size}px\n{augment}")
            
            inference_times.append(data.get('avg_inference_time', 0))
            throughputs.append(data.get('samples_per_second', 0))
            accuracies.append(data.get('test_accuracy', data.get('best_val_accuracy', 0)))
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if models:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Inference time plot
        y_pos = np.arange(len(models))
        ax1.barh(y_pos, inference_times, align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel('Average Inference Time (seconds)')
        ax1.set_title('Inference Time by Model')
        
        # Throughput plot
        ax2.barh(y_pos, throughputs, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(models)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel('Throughput (samples/second)')
        ax2.set_title('Model Throughput')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Inference comparison plot saved to {output_path}")
        
        return fig
    else:
        print("No valid summary files found for inference comparison")
        return None

def plot_model_size_vs_accuracy(summary_files, output_path):
    """Create a scatter plot of model size vs. accuracy"""
    models = []
    model_sizes = []
    accuracies = []
    param_counts = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            models.append(data['model'])
            model_sizes.append(data.get('model_size_mb', 0))
            accuracies.append(data.get('test_accuracy', data.get('best_val_accuracy', 0)))
            param_counts.append(data.get('total_params', 0))
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if models:
        plt.figure(figsize=(12, 8))
        
        # Size of scatter points proportional to parameter count
        sizes = [count/1e5 for count in param_counts]
        
        scatter = plt.scatter(model_sizes, accuracies, s=sizes, alpha=0.6)
        
        # Annotate points with model names
        for i, model in enumerate(models):
            plt.annotate(model, (model_sizes[i], accuracies[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy')
        plt.title('Model Size vs. Accuracy')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300)
        print(f"Size vs. accuracy plot saved to {output_path}")
        
        return plt
    else:
        print("No valid summary files found for size-accuracy comparison")
        return None

def format_time(seconds):
    """Format seconds into hours:minutes:seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"