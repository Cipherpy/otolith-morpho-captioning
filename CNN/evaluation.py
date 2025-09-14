import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from torch.cuda.amp import autocast

from models import create_model, load_model
from data import determine_input_size, get_transforms, load_datasets, create_dataloaders, load_class_names
from utils import get_system_info
from visualization import visualize_gradcam, get_target_layer_name



def evaluate_model(args, model=None, device=None, class_names=None, output_dirs=None):
    """
    Evaluate a trained model on the test dataset
    """
    start_time = time.time()
    
    # Determine appropriate image size
    img_size = determine_input_size(args.model, args.img_size)
    grayscale_str = "grayscale" if args.grayscale else "color"
    augment_str = "augmented" if args.augment else "no_augment"
    
    # If model is not provided, load it
    if model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load class names
        class_names_path = os.path.join(output_dirs['models'], f"class_labels_{args.model}.txt")
        class_names = load_class_names(class_names_path)
        
        if class_names is None:
            raise ValueError(f"Class names file not found at {class_names_path}. Please train the model first.")
        
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names: {class_names}")
        
        # Try to load the best model first, if it exists, otherwise load the final model
        model_path = os.path.join(output_dirs['models'], f"{args.model}_{augment_str}_{grayscale_str}_size{img_size}_best.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(output_dirs['models'], f"{args.model}_{augment_str}_{grayscale_str}_size{img_size}_final.pth")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}. Please train the model first.")
            
        print(f"Loading model from {model_path}")
        
        # Load the model
        model = load_model(model_path, args.model, num_classes, device)
    
    # Initialize W&B if needed
    if not args.no_wandb:
        wandb.init(project="image-classification-eval", 
                  name=f"eval_{args.model}_{augment_str}_size{img_size}",
                  config={
                      "model": args.model,
                      "batch_size": args.batch_size,
                      "data_dir": args.data_dir,
                      "augmentation": args.augment,
                      "img_size": img_size
                  })
    
    # Create test transforms
    data_transforms = get_transforms(img_size, augment=False, grayscale=args.grayscale)  # No augmentation for evaluation
    
    # Load test dataset
    test_datasets = load_datasets(args.data_dir, data_transforms)
    if 'test' not in test_datasets:
        raise ValueError(f"Test dataset not found in {args.data_dir}/test. Please make sure the test data is available.")
    
    # Create dataloader
    test_dataloader = create_dataloaders({'test': test_datasets['test']}, args.batch_size)['test']
    
    # Prepare for evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    batch_times = []
    
    # Variables for top-k accuracy
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    # Store a batch for Grad-CAM visualization if requested
    gradcam_images = None
    gradcam_labels = None
    
    # Start evaluation
    print("Evaluating model on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing"):
            batch_start = time.time()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Save a batch for Grad-CAM if requested and not already stored
            if args.gradcam and gradcam_images is None:
                gradcam_images = inputs.clone().cpu()
                gradcam_labels = labels.clone().cpu()

            # Forward pass (with mixed precision if enabled)
            if args.mixed_precision:
                with autocast():
                    if args.model == 'inception_v3' or args.model == 'googlenet':
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Take only main output, not aux
                    else:
                        outputs = model(inputs)
            else:
                if args.model == 'inception_v3' or args.model == 'googlenet':
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take only main output, not aux
                else:
                    outputs = model(inputs)
            
            # Calculate probabilities and predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Calculate top-k accuracy
            _, top1_preds = torch.max(outputs, 1)
            
            # For top-5 accuracy
            if outputs.size(1) >= 5:  # If we have at least 5 classes
                _, top5_indices = torch.topk(outputs, 5, dim=1)
                batch_top5_correct = sum(labels.view(-1, 1) == top5_indices).float().sum().item()
                top5_correct += batch_top5_correct
            else:
                # If fewer than 5 classes, top-5 is same as regular accuracy
                top5_correct += torch.sum(preds == labels).item()
            
            # Add to top-1 correct count
            top1_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            # Store predictions, labels and probabilities for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    # Calculate top-k accuracy
    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples
    
    # Calculate standard metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    test_precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    test_recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    test_f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    
    # Print results
    print("\n=== Test Set Metrics ===")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-score:  {test_f1:.4f}")
    
    # Print top-k metrics
    print("\n=== Top-K Accuracy Metrics ===")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    
    # Calculate per-class metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Calculate inference timing statistics
    avg_inference_time = np.mean(batch_times)
    total_inference_time = sum(batch_times)
    samples_per_second = len(all_labels) / total_inference_time
    
    print(f"\nInference Performance:")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average batch inference time: {avg_inference_time:.4f} seconds")
    print(f"Throughput: {samples_per_second:.2f} samples/second")
    
    # Log to W&B
    if not args.no_wandb:
        wandb.log({
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_precision_micro": test_precision_micro,
            "test_recall_micro": test_recall_micro,
            "test_f1_micro": test_f1_micro,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "avg_inference_time": avg_inference_time,
            "total_inference_time": total_inference_time,
            "samples_per_second": samples_per_second,
            "Total inference time (in seconds)": total_inference_time,
            "Average batch inference time (in seconds)": avg_inference_time,
            "Throughput (samples/second)": samples_per_second
        })
        
        # Log per-class metrics
        for i, class_name in enumerate(class_names):
            if class_name in report:
                wandb.log({
                    f"class_{class_name}_precision": report[class_name]['precision'],
                    f"class_{class_name}_recall": report[class_name]['recall'],
                    f"class_{class_name}_f1": report[class_name]['f1-score']
                })
    
    # Create confusion matrix plots
    plt.figure(figsize=(14, 12))
    
    # Plot raw counts
    
    ax=sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix -({args.model})", fontsize=14)
    
  
    
    plt.tight_layout()
    
    # Save confusion matrix plot
    conf_matrix_file = os.path.join(output_dirs['plots'], 
                                  f"confusion_matrix_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.png")
    plt.savefig(conf_matrix_file, dpi=300)
    print(f"Confusion matrix saved to {conf_matrix_file}")
    
    # Create confidence score distribution plot
    plt.figure(figsize=(10, 6))
    
    # Get confidence scores for predicted classes
    confidences = []
    is_correct = []
    
    for i, (probs, true_label, pred) in enumerate(zip(all_probs, all_labels, all_preds)):
        confidence = probs[pred]  # Confidence for the predicted class
        confidences.append(confidence)
        is_correct.append(pred == true_label)
    
    # Create separate lists for correct and incorrect predictions
    correct_conf = [conf for conf, corr in zip(confidences, is_correct) if corr]
    incorrect_conf = [conf for conf, corr in zip(confidences, is_correct) if not corr]
    
    # Plot histograms
    plt.hist(correct_conf, alpha=0.7, bins=20, label='Correct Predictions', 
             color='green', range=(0, 1))
    plt.hist(incorrect_conf, alpha=0.7, bins=20, label='Incorrect Predictions', 
             color='red', range=(0, 1))
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save confidence plot
    conf_plot_file = os.path.join(output_dirs['plots'], 
                                f"confidence_dist_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.png")
    plt.savefig(conf_plot_file, dpi=300)
    print(f"Confidence distribution saved to {conf_plot_file}")
    
    # Log to W&B if enabled
    if not args.no_wandb:
        wandb.log({"confidence_distribution": wandb.Image(conf_plot_file)})
    
    # Save per-class metrics
    per_class_df = pd.DataFrame(report).transpose()
    per_class_file = os.path.join(output_dirs['csv'], 
                                f"per_class_metrics_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.csv")
    per_class_df.to_csv(per_class_file)
    print(f"Per-class metrics saved to {per_class_file}")

    # Generate Grad-CAM visualizations if requested
    if args.gradcam:
        print("\n=== Generating Grad-CAM Visualizations ===")
        
        # Create directory for visualizations
        gradcam_dir = os.path.join(output_dirs['plots'], 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Get target layer name for the model
        target_layer_name = get_target_layer_name(args.model, model)
        print(f"Using target layer: {target_layer_name}")
        
        # Check if we need to save the sample indices for consistent visualization
        sample_index_file = os.path.join("outputs", 
                                        f"gradcam_sample_indices_{args.model}.json")
        
        # Dictionary to store sample indices
        gradcam_samples_dict = {
            "correct_samples": [],
            "incorrect_samples": []
        }
        
        # Check if the sample indices file already exists
        use_stored_indices = os.path.exists(sample_index_file)
        
        if use_stored_indices:
            # Load existing sample indices
            import json
            with open(sample_index_file, 'r') as f:
                stored_indices = json.load(f)
            
            print(f"Using stored sample indices from {sample_index_file}")
            
            # Get existing indices
            correct_indices = stored_indices.get("correct_samples", [])
            incorrect_indices = stored_indices.get("incorrect_samples", [])
            
            # Initialize lists to store the actual samples
            correct_samples = []
            incorrect_samples = []
            
            # Convert test dataset to list for easier indexing
            all_images = []
            all_image_labels = []
            
            model.eval()
            with torch.no_grad():
                # Collect all test images and labels
                for inputs, labels in tqdm(test_dataloader, desc="Collecting all test samples"):
                    for i in range(inputs.size(0)):
                        all_images.append(inputs[i:i+1].clone())
                        all_image_labels.append(labels[i].item())
            
            # Get model predictions for all samples
            predictions = []
            for img in tqdm(all_images, desc="Getting predictions for all samples"):
                img = img.to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())
            
            # Get the stored samples based on indices
            for idx in correct_indices:
                if idx < len(all_images):
                    true_label = all_image_labels[idx]
                    pred_label = predictions[idx]
                    if true_label == pred_label:  # Double-check it's still correct
                        correct_samples.append((all_images[idx], true_label, pred_label))
                    else:
                        # The prediction might have changed since last run
                        incorrect_samples.append((all_images[idx], true_label, pred_label))
            
            for idx in incorrect_indices:
                if idx < len(all_images):
                    true_label = all_image_labels[idx]
                    pred_label = predictions[idx]
                    if true_label != pred_label:  # Double-check it's still incorrect
                        incorrect_samples.append((all_images[idx], true_label, pred_label))
            
            print(f"Retrieved {len(correct_samples)} correct and {len(incorrect_samples)} incorrect samples from stored indices")
        
        else:
            # No stored indices, collect new samples
            # Initialize lists to store correct and incorrect predictions
            correct_samples = []
            incorrect_samples = []
            
            # Arrays to keep track of sample indices
            sample_indices = []
            
            # Collect samples for visualization
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(tqdm(test_dataloader, desc="Collecting samples for Grad-CAM")):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Get predictions
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    # Separate correct and incorrect predictions
                    for i in range(inputs.size(0)):
                        sample_idx = batch_idx * test_dataloader.batch_size + i
                        sample_indices.append(sample_idx)
                        
                        if preds[i] == labels[i]:
                            correct_samples.append((inputs[i:i+1].clone(), labels[i].item(), preds[i].item()))
                            gradcam_samples_dict["correct_samples"].append(sample_idx)
                        else:
                            incorrect_samples.append((inputs[i:i+1].clone(), labels[i].item(), preds[i].item()))
                            gradcam_samples_dict["incorrect_samples"].append(sample_idx)
                        
                        # Stop if we have enough samples
                        if len(correct_samples) >= args.gradcam_samples and (
                            not args.gradcam_correct_and_incorrect or 
                            len(incorrect_samples) >= args.gradcam_samples):
                            break
                    
                    # Check if we have enough samples
                    if len(correct_samples) >= args.gradcam_samples and (
                        not args.gradcam_correct_and_incorrect or 
                        len(incorrect_samples) >= args.gradcam_samples):
                        break
            
            # Save the sample indices for future runs
            import json
            with open(sample_index_file, 'w') as f:
                json.dump(gradcam_samples_dict, f, indent=4)
            print(f"Saved sample indices to {sample_index_file}")
        
        # Generate visualizations for correct predictions
        print(f"Generating Grad-CAM for {min(len(correct_samples), args.gradcam_samples)} correctly classified samples")
        for i, (input_tensor, true_label, pred_label) in enumerate(correct_samples[:args.gradcam_samples]):
            vis_path = os.path.join(gradcam_dir, 
                                f"gradcam_{args.model}_correct_{i}_class{true_label}.png")
            
            try:
                visualize_gradcam(
                    model=model,
                    image_tensor=input_tensor,
                    class_idx=pred_label,
                    class_names=class_names,
                    target_layer_name=target_layer_name,
                    output_path=vis_path
                )
                print(f"Generated Grad-CAM for correct sample {i+1}/{min(len(correct_samples), args.gradcam_samples)}")
                
                # Log to W&B if enabled
                if not args.no_wandb:
                    wandb.log({f"gradcam_correct_{i}": wandb.Image(vis_path)})
                    
            except Exception as e:
                print(f"Error generating Grad-CAM for correct sample {i}: {e}")
        
        # Generate visualizations for incorrect predictions if requested
        if args.gradcam_correct_and_incorrect and incorrect_samples:
            print(f"Generating Grad-CAM for {min(len(incorrect_samples), args.gradcam_samples)} incorrectly classified samples")
            for i, (input_tensor, true_label, pred_label) in enumerate(incorrect_samples[:args.gradcam_samples]):
                vis_path = os.path.join(gradcam_dir, 
                                    f"gradcam_{args.model}_incorrect_{i}_true{true_label}_pred{pred_label}.png")
                
                try:
                    visualize_gradcam(
                        model=model,
                        image_tensor=input_tensor,
                        class_idx=pred_label,  # Visualize what the model thought it saw
                        class_names=class_names,
                        target_layer_name=target_layer_name,
                        output_path=vis_path
                    )
                    print(f"Generated Grad-CAM for incorrect sample {i+1}/{min(len(incorrect_samples), args.gradcam_samples)}")
                    
                    # Log to W&B if enabled
                    if not args.no_wandb:
                        wandb.log({f"gradcam_incorrect_{i}": wandb.Image(vis_path)})
                        
                except Exception as e:
                    print(f"Error generating Grad-CAM for incorrect sample {i}: {e}")
        
        print(f"Saved Grad-CAM visualizations to {gradcam_dir}")

    # Create a summary of results
    eval_summary = {
        "model": args.model,
        "image_size": img_size,
        "augmentation": args.augment,
        "grayscale": args.grayscale,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_precision_micro": test_precision_micro,
        "test_recall_micro": test_recall_micro,
        "test_f1_micro": test_f1_micro,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
        "samples_per_second": samples_per_second,
        "batch_size": args.batch_size,
        "test_samples": len(all_labels),
        "evaluation_time": time.time() - start_time
    }
    
    # Save evaluation summary
    eval_summary_file = os.path.join(output_dirs['reports'], 
                                   f"eval_summary_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.json")
    import json
    with open(eval_summary_file, 'w') as f:
        json.dump(eval_summary, f, indent=4)
    print(f"Evaluation summary saved to {eval_summary_file}")
    
    # Log confusion matrix to W&B
    if not args.no_wandb:
        wandb.log({"confusion_matrix": wandb.Image(conf_matrix_file)})
        wandb.finish()
    
    return eval_summary