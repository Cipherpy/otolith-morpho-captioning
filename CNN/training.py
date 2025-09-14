import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from models import create_model, get_model_info, get_trainable_params
from data import determine_input_size, get_transforms, load_datasets, create_dataloaders, save_class_names
from utils import get_system_info

def train_model(args, output_dirs):
    # Determine appropriate image size for the model
    img_size = determine_input_size(args.model, args.img_size)

    grayscale_str = "grayscale" if args.grayscale else "color"
    
    # Create run name for W&B
    augment_str = "augmented" if args.augment else "no_augment"
    run_name = args.run_name if args.run_name else f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{augment_str}_{grayscale_str}_size{img_size}"
    
    # Initialize wandb if enabled
    if not args.no_wandb:
        wandb.init(project="Otolith-classification-benchmark", 
                  name=run_name,
                  config={
                      "model": args.model,
                      "epochs": args.epochs,
                      "batch_size": args.batch_size,
                      "learning_rate": args.lr,
                      "optimizer": args.optimizer,
                      "data_dir": args.data_dir,
                      "augmentation": args.augment,
                       "grayscale": args.grayscale,  # Add this line
                      "img_size": img_size,
                      "freeze_base": args.freeze_base,
                      "mixed_precision": args.mixed_precision,
                      "system_info": get_system_info()
                  })

    # Get transforms based on image size and augmentation setting
    data_transforms = get_transforms(img_size, args.augment)

    # Load datasets
    image_datasets = load_datasets(args.data_dir, data_transforms)
    
    # Create dataloaders
    dataloaders = create_dataloaders(image_datasets, args.batch_size)

    # Get number of classes and class names
    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Using input size: {img_size}x{img_size}")

    # Save class names to a text file for later use
    class_names_path = os.path.join(output_dirs['models'], f"class_labels_{args.model}.txt")
    save_class_names(class_names, class_names_path)

    # Create the model
    model = create_model(args.model, num_classes, pretrained=True, freeze_base=args.freeze_base)
    
    # Get model information
    model_info = get_model_info(model)
    print(f"Model: {args.model}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    if not args.no_wandb:
        wandb.log({"model_total_params": model_info['total_params'],
                  "model_trainable_params": model_info['trainable_params'],
                  "model_size_mb": model_info['model_size_mb']})

    # Create Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Get trainable parameters
    params_to_update = get_trainable_params(model, args.model, args.freeze_base)
    
    # Set optimizer based on argument
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params_to_update, lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params_to_update, lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable multi-GPU usage if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Enable mixed precision training with gradient scaler if requested
    scaler = GradScaler() if args.mixed_precision else None

    # Log model to W&B
    if not args.no_wandb:
        wandb.watch(model, log="all", log_freq=10)

    history = []
    epoch_times = []
    batch_times = {
        'train': [],
        'val': []
    }

    # Training + Validation
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        epoch_data = {'epoch': epoch+1}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            phase_start_time = time.time()
            
            # Use tqdm for progress bar
            with tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch+1}/{args.epochs}") as tepoch:
                for inputs, labels in tepoch:
                    batch_start_time = time.time()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    # Perform forward pass with or without mixed precision
                    with torch.set_grad_enabled(phase == 'train'):
                        if args.model == 'inception_v3' or args.model == 'googlenet':
                            if phase == 'train':
                                # When training inception/googlenet, we need to handle the auxiliary output
                                if args.mixed_precision:
                                    with autocast():
                                        outputs, aux_outputs = model(inputs)
                                        loss1 = criterion(outputs, labels)
                                        loss2 = criterion(aux_outputs, labels)
                                        loss = loss1 + 0.4 * loss2
                                else:
                                    outputs, aux_outputs = model(inputs)
                                    loss1 = criterion(outputs, labels)
                                    loss2 = criterion(aux_outputs, labels)
                                    loss = loss1 + 0.4 * loss2
                            else:
                                # During validation/testing, auxiliary output is not used
                                if args.mixed_precision:
                                    with autocast():
                                        outputs = model(inputs)
                                        loss = criterion(outputs, labels)
                                else:
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                        else:
                            if args.mixed_precision:
                                with autocast():
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            if args.mixed_precision:
                                # Use gradient scaler for mixed precision training
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                    batch_end_time = time.time()
                    batch_times[phase].append(batch_end_time - batch_start_time)
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Store predictions and labels for metrics calculation
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    tepoch.set_postfix(loss=loss.item(), accuracy=torch.sum(preds == labels.data).item()/inputs.size(0))

            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            # Calculate additional metrics
            epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            # Store metrics in epoch data
            epoch_data[f'{phase}_loss'] = epoch_loss
            epoch_data[f'{phase}_accuracy'] = epoch_acc.item()
            epoch_data[f'{phase}_precision'] = epoch_precision
            epoch_data[f'{phase}_recall'] = epoch_recall
            epoch_data[f'{phase}_f1'] = epoch_f1
            epoch_data[f'{phase}_time_seconds'] = phase_duration
            epoch_data[f'{phase}_avg_batch_time'] = np.mean(batch_times[phase])

            print(f"{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}, "
                 f"Time: {phase_duration:.2f}s, Avg Batch: {np.mean(batch_times[phase]):.4f}s")
            
            # Log metrics to W&B
            if not args.no_wandb:
                wandb.log({
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc.item(),
                    f"{phase}_precision": epoch_precision,
                    f"{phase}_recall": epoch_recall,
                    f"{phase}_f1": epoch_f1,
                    f"{phase}_time_seconds": phase_duration,
                    f"{phase}_avg_batch_time": np.mean(batch_times[phase]),
                    "epoch": epoch+1,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Update learning rate based on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)
                # Save best model
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    model_filename = os.path.join(output_dirs['models'], 
                                                f"{args.model}_{augment_str}_{grayscale_str}_size{img_size}_best.pth")
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save(model.module.state_dict(), model_filename)
                    else:
                        torch.save(model.state_dict(), model_filename)
                    print(f"Saved best model to {model_filename}")
                    
                    if not args.no_wandb:
                        # Log to W&B - save the model
                        wandb.save(model_filename)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_data['epoch_time_seconds'] = epoch_duration
        
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        if not args.no_wandb:
            wandb.log({"epoch_time_seconds": epoch_duration})
        
        history.append(epoch_data)

    # Calculate average times
    avg_epoch_time = np.mean(epoch_times)
    avg_train_batch_time = np.mean(batch_times['train'])
    avg_val_batch_time = np.mean(batch_times['val'])
    
    print(f"\nTraining completed in {sum(epoch_times):.2f} seconds")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Average training batch time: {avg_train_batch_time:.4f} seconds")
    print(f"Average validation batch time: {avg_val_batch_time:.4f} seconds")
    
    # Save time metrics to W&B
    if not args.no_wandb:
        wandb.log({
            "total_training_time": sum(epoch_times),
            "avg_epoch_time": avg_epoch_time,
            "avg_train_batch_time": avg_train_batch_time,
            "avg_val_batch_time": avg_val_batch_time
        })

    # Save training history to CSV with timing info
    history_df = pd.DataFrame(history)
    history_filename = os.path.join(output_dirs['csv'], 
                              f"training_history_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.csv")
    history_df.to_csv(history_filename, index=False)
    print(f"Training history saved as {history_filename}")

    # Create results summary
    results_summary = {
        "model": args.model,
        "image_size": img_size,
        "augmentation": args.augment,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "freeze_base": args.freeze_base,
        "mixed_precision": args.mixed_precision,
        "total_params": model_info['total_params'],
        "trainable_params": model_info['trainable_params'],
        "model_size_mb": model_info['model_size_mb'],
        "best_val_loss": best_val_loss,
        "best_val_accuracy": max(history_df['val_accuracy']),
        "best_val_f1": max(history_df['val_f1']),
        "total_train_time": sum(epoch_times),
        "avg_epoch_time": avg_epoch_time,
        "avg_train_batch_time": avg_train_batch_time,
        "avg_val_batch_time": avg_val_batch_time,
        "system_info": get_system_info()
    }
    
    # Save results summary to JSON
    import json
    with open(os.path.join(output_dirs['reports'], 
                          f"results_summary_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per Epoch')
    
    # Precision/Recall plot
    plt.subplot(2, 2, 3)
    plt.plot(history_df['epoch'], history_df['train_precision'], label='Train Precision')
    plt.plot(history_df['epoch'], history_df['val_precision'], label='Validation Precision')
    plt.plot(history_df['epoch'], history_df['train_recall'], label='Train Recall')
    plt.plot(history_df['epoch'], history_df['val_recall'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision & Recall per Epoch')
    
    # Timing plot
    plt.subplot(2, 2, 4)
    plt.plot(history_df['epoch'], history_df['train_time_seconds'], label='Train Time')
    plt.plot(history_df['epoch'], history_df['val_time_seconds'], label='Validation Time')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Time per Epoch')
    
    plt.tight_layout()
    metrics_plot_filename = os.path.join(output_dirs['plots'], 
                                       f"metrics_plot_{args.model}_{augment_str}_{grayscale_str}_size{img_size}.png")
    plt.savefig(metrics_plot_filename, dpi=300)
    print(f"Metrics plots saved as {metrics_plot_filename}")
    
    # Save the final model
    final_model_filename = os.path.join(output_dirs['models'], 
                                      f"{args.model}_{augment_str}_{grayscale_str}_size{img_size}_final.pth")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_filename)
    else:
        torch.save(model.state_dict(), final_model_filename)
    print(f"Final model saved as {final_model_filename}")
    
    if not args.no_wandb:
        # Log final plots to W&B
        wandb.log({"metrics_plot": wandb.Image(metrics_plot_filename)})
        wandb.save(final_model_filename)
        wandb.finish()

    # Return model and device for potential immediate evaluation
    return model, device, class_names, results_summary