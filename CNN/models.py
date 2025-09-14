import torch
import torch.nn as nn
from torchvision import models

def create_model(model_name, num_classes, pretrained=True, freeze_base=False):
    """Create and configure the selected model architecture"""
    # Initialize the appropriate model
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        
        # Freeze layers if specified
        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace final layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name.startswith('vgg'):
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
            
        # Freeze layers if specified
        if freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name.startswith('densenet'):
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained=pretrained)
            
        # Freeze layers if specified
        if freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=pretrained, aux_logits=True)
        
        # Freeze layers if specified
        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace final layer and aux layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
        
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=pretrained, aux_logits=True)
        
        # Freeze layers if specified
        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace final layer and aux layers
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        
        # Freeze layers if specified
        if freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze layers if specified
        if freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
            
        # Freeze layers if specified
        if freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def get_model_info(model):
    """Get model parameters and size information"""
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size in MB
    model_size_bytes = 0
    for param in model.parameters():
        model_size_bytes += param.nelement() * param.element_size()
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb
    }

def load_model(model_path, model_name, num_classes, device):
    """Load a saved model from disk"""
    model = create_model(model_name, num_classes, pretrained=False)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def get_trainable_params(model, model_name, freeze_base):
    """Get the trainable parameters for the model based on architecture and freeze setting"""
    if not freeze_base:
        return model.parameters()
    
    if model_name.startswith('resnet'):
        return model.fc.parameters()
    elif model_name.startswith('vgg'):
        return model.classifier[6].parameters()
    elif model_name.startswith('densenet'):
        return model.classifier.parameters()
    elif model_name == 'inception_v3' or model_name == 'googlenet':
        # For inception and googlenet, we need to train both the main classifier and aux classifier
        if hasattr(model, 'AuxLogits'):
            return list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())
        return model.fc.parameters()
    elif model_name == 'alexnet':
        return model.classifier[6].parameters()
    elif model_name == 'mobilenet_v2' or model_name.startswith('efficientnet'):
        return model.classifier[1].parameters()
    else:
        return model.parameters()