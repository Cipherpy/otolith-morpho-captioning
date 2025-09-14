import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

def determine_input_size(model_name, user_specified_size=224):
    """Determine the appropriate input size for the model"""
    # For inception and googlenet, the default is 299x299
    if model_name in ['inception_v3', 'googlenet'] and user_specified_size == 224:
        return 299
    # For other models, use the user-specified size or default to 224
    return user_specified_size

def get_transforms(img_size, augment=False, grayscale=False):
    """Create image transformations based on image size, augmentation flag, and color mode"""
    
    # Add grayscale conversion if requested
    if grayscale:
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Only brightness/contrast for grayscale
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Still use same normalization
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Original color transforms (your existing code)
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return {
        'train': train_transform,
        'val': test_transform,
        'test': test_transform
    }

def load_datasets(data_dir, transforms_dict):

    print(data_dir)
    """Load image datasets from the specified directory"""
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_dict[x])
        for x in ['train', 'val', 'test'] if os.path.exists(os.path.join(data_dir, x))
    }
    
    return image_datasets

def create_dataloaders(image_datasets, batch_size, num_workers=4):
    """Create DataLoader objects for each dataset split"""
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        for x in image_datasets.keys()
    }
    
    return dataloaders

def save_class_names(class_names, output_path):
    """Save class names to a text file"""
    with open(output_path, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Class names saved to {output_path}")

def load_class_names(file_path):
    """Load class names from a text file"""
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names

def get_sample_batch(dataloader):
    """Get a sample batch from the dataloader for visualization or testing"""
    data_iter = iter(dataloader)
    return next(data_iter)

def calculate_dataset_stats(dataloader):
    """Calculate dataset mean and standard deviation for normalization"""
    mean = 0.0
    std = 0.0
    num_samples = 0
    
    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_size
    
    mean /= num_samples
    std /= num_samples
    
    return mean, std

def preprocess_image(image_path, transform):
    """Preprocess a single image for inference"""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def visualize_batch(dataloader, class_names, num_images=16):
    """Get a batch of images and their labels for visualization"""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    # Get a batch of training data
    images, labels = next(iter(dataloader))
    
    # Make a grid from batch
    out = make_grid(images[:num_images], nrow=4)
    
    # Convert tensor to numpy array for displaying with matplotlib
    out = out.numpy().transpose((1, 2, 0))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    out = std * out + mean
    out = np.clip(out, 0, 1)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(out)
    
    # Print labels
    if class_names:
        class_labels = [class_names[labels[i]] for i in range(num_images)]
        plt.title(' '.join('%5s' % class_labels[j] for j in range(num_images)))
    
    plt.axis('off')
    
    return plt