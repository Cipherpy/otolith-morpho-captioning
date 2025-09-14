import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.hooks = []
        self.gradients = None
        self.activations = None
        self.target_layer = None
        
        # Find the target layer
        self.target_layer = self._find_target_layer(model, target_layer_name)
        
        if self.target_layer is None:
            # Try to get a default layer if the specific one is not found
            print(f"Warning: Layer {target_layer_name} not found. Trying to find a suitable layer...")
            self.target_layer = self._find_default_target_layer(model)
            
        if self.target_layer is None:
            raise ValueError(f"Could not find a suitable layer for Grad-CAM in the model")
        
        # Register hooks
        self.hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))
        
        print(f"Successfully attached Grad-CAM to layer: {type(self.target_layer).__name__}")
    
    def _find_target_layer(self, model, target_layer_name):
        """Recursively find a layer by name"""
        if '.' in target_layer_name:
            # Handle nested layer names like 'features.29'
            parts = target_layer_name.split('.')
            current = model
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            
            return current
        else:
            # Handle top-level names like 'layer4'
            if hasattr(model, target_layer_name):
                return getattr(model, target_layer_name)
            else:
                return None
    
    def _find_default_target_layer(self, model):
        """Find a suitable layer for Grad-CAM if the specified one is not found"""
        # Try to find the last convolutional layer
        suitable_layers = []
        
        # If model is wrapped with DataParallel, get the module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        # Helper function to recursively find conv layers
        def find_conv_layers(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, torch.nn.Conv2d):
                    suitable_layers.append((full_name, child))
                else:
                    find_conv_layers(child, full_name)
        
        find_conv_layers(model)
        
        if suitable_layers:
            # Return the last convolutional layer
            print(f"Using layer {suitable_layers[-1][0]} for Grad-CAM")
            return suitable_layers[-1][1]
        
        return None
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input tensor (already preprocessed)
            class_idx: Target class index (None for highest probability class)
            
        Returns:
            heatmap: Numpy array containing the heatmap
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Handle models that return tuple (like inception)
        if isinstance(output, tuple):
            output = output[0]
        
        # Get the class index if not provided
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        
        # Generate heatmap
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)  # ReLU to keep only positive influence
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Convert to numpy
        heatmap = cam.cpu().numpy()
        
        return heatmap
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

def get_target_layer_name(model_name, model=None):
    """Get the appropriate target layer name for Grad-CAM based on model architecture"""
    # If model is provided, we'll check its structure
    if model is not None:
        # Try to find the target layer by inspecting the model structure
        # Check for common layer patterns in different models
        if hasattr(model, 'layer4'):
            return 'layer4'  # ResNet pattern
        elif hasattr(model, 'features') and len(list(model.features.children())) > 0:
            # For models with a 'features' module (VGG, DenseNet, etc.)
            # Return the last convolutional layer in features
            features = model.features
            last_conv_idx = None
            
            # Find the last conv layer
            for i, module in enumerate(features.modules()):
                if isinstance(module, torch.nn.Conv2d):
                    last_conv_idx = i
            
            if last_conv_idx is not None:
                return f'features.{last_conv_idx}'
            else:
                # If no conv layer found, return the last module in features
                return f'features.{len(list(features.children()))-1}'
        
        # If model is wrapped with DataParallel
        if isinstance(model, torch.nn.DataParallel):
            return get_target_layer_name(model_name, model.module)
    
    # If we couldn't determine from the model structure, use the model name
    if model_name.startswith('resnet'):
        return 'layer4'
    elif model_name.startswith('vgg'):
        return 'features.29'  # For VGG16
    elif model_name.startswith('densenet'):
        return 'features.denseblock4'
    elif model_name == 'inception_v3':
        return 'Mixed_7c'
    elif model_name == 'googlenet':
        return 'inception5b'
    elif model_name == 'alexnet':
        return 'features.10'
    elif model_name == 'mobilenet_v2':
        return 'features.18'
    elif model_name.startswith('efficientnet'):
        return 'features.7'
    else:
        # Default fallback
        return 'features'  # Generic fallback for unknown models

def visualize_gradcam(model, image_tensor, class_idx, class_names, target_layer_name, output_path):
    """
    Generate and save Grad-CAM visualization
    
    Args:
        model: The CNN model
        image_tensor: Preprocessed image tensor
        class_idx: Target class index
        class_names: List of class names
        target_layer_name: Name of the target layer for Grad-CAM
        output_path: Path to save the visualization
    
    Returns:
        Output path of the saved visualization
    """
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer_name)
    
    # Generate heatmap
    heatmap = grad_cam.generate_heatmap(image_tensor, class_idx)
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    # Convert input tensor to image
    img = image_tensor.squeeze(0).cpu()
    
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy
    img = img.permute(1, 2, 0).numpy()
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Resize heatmap to match image size using cv2
    import cv2
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3]
    
    # Create overlay
    overlay = heatmap_colored * 0.4 + img * 0.6
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title(f"Original Image\nClass: {class_names[class_idx]}")
    ax1.axis('off')
    
    # Heatmap
    ax2.imshow(heatmap_colored)
    ax2.set_title("Grad-CAM Heatmap")
    ax2.axis('off')
    
    # Overlay
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path