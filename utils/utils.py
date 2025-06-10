"""
Utility functions for model training, optimization, and evaluation metrics.
"""

from torch import optim
import torch
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    recall_score,
    precision_score, 
    f1_score, 
    balanced_accuracy_score
)
from itertools import product

def get_learning_rate(optimizer):
    """
    Extract the current learning rate from the optimizer.
    
    Args:
        optimizer: PyTorch optimizer object
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def configure_optimizers(model, learning_rate, weight_decay, gamma, lr_decay_every_x_epochs):
    """
    Configure optimizer and learning rate scheduler for model training.
    
    Args:
        model: PyTorch model
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization parameter
        gamma (float): Learning rate decay factor
        lr_decay_every_x_epochs (int): Number of epochs between learning rate updates
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=lr_decay_every_x_epochs
    )
    
    return optimizer, scheduler

def soft_iou_loss(predictions, targets):
    """
    Compute soft IoU (Intersection over Union) loss.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        torch.Tensor: Mean IoU loss
    """
    batch_size = predictions.size()[0]
    predictions = predictions.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    intersection = torch.sum(torch.mul(predictions, targets), dim=-1, keepdim=False)
    union = torch.sum(torch.mul(predictions, predictions) + targets, dim=-1, keepdim=False) - intersection
    
    return torch.mean(1 - intersection / union)

def one_vs_rest_metrics(y_true, y_pred, y_score):
    """
    Compute one-vs-rest classification metrics for each class.
    
    This function evaluates the model's performance by treating each class
    as the positive class and all others as negative, computing metrics
    for each such binary classification task.
    
    Args:
        y_true (array-like): True class labels, shape (n_samples,)
        y_pred (array-like): Predicted class labels, shape (n_samples,)
        y_score (array-like): Predicted class probabilities, shape (n_samples, n_classes)
        
    Returns:
        pd.DataFrame: One-vs-rest metrics for each class, including:
            - AUC (Area Under ROC Curve)
            - F1 Score
            - Number of true positives
            - Number of predicted positives
    """
    n_classes = y_score.shape[1]
    metrics_list = []

    for class_idx in range(n_classes):
        # Convert to binary classification for current class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        y_score_binary = y_score[:, class_idx]

        # Compute binary classification metrics
        class_metrics = binary_classification_metrics(
            y_true=y_true_binary,
            y_pred=y_pred_binary,
            y_score=y_score_binary,
            pos_label=1
        )

        # Add sample counts
        class_metrics['n_true_positives'] = sum(y_true_binary)
        class_metrics['n_predicted_positives'] = sum(y_pred_binary)
        
        metrics_list.append(class_metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.index.name = 'class_index'
    return metrics_df

def binary_classification_metrics(y_true, y_pred, y_score, pos_label=1):
    """
    Compute binary classification metrics.
    
    Args:
        y_true (array-like): True labels, shape (n_samples,)
        y_pred (array-like): Predicted labels, shape (n_samples,)
        y_score (array-like): Predicted probabilities, shape (n_samples,)
        pos_label (int): Label of the positive class
        
    Returns:
        dict: Dictionary containing:
            - AUC (Area Under ROC Curve)
            - F1 Score
    """
    return {
        'auc': roc_auc_score(y_true=y_true, y_score=y_score),
        'f1': f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)
    }

def get_overall_multiclass_metrics(y_true, y_pred, y_score):
    """
    Compute overall metrics for multiclass classification.
    
    Args:
        y_true (array-like): True class labels, shape (n_samples,)
        y_pred (array-like): Predicted class labels, shape (n_samples,)
        y_score (array-like): Predicted class probabilities, shape (n_samples, n_classes)
        
    Returns:
        dict: Dictionary containing:
            - Accuracy
            - Balanced Accuracy
            - Macro-average AUC (One-vs-Rest)
            - Weighted-average AUC (One-vs-Rest)
    """
    return {
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
        'auc_ovr_macro': roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            average='macro',
            multi_class='ovr'
        ),
        'auc_ovr_weighted': roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            average='weighted',
            multi_class='ovr'
        )
    }