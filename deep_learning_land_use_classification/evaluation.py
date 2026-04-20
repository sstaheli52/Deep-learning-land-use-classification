from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy_metrics_singlelabel(model, loader, device, num_classes=None):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, dict): # Some models return dicts instead of tensors
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(images["pixel_values"])
            else: # Model returns tensors directly
                images = images.to(device)
                outputs = model(images)
            preds = outputs.argmax(dim=1) # Use argmax for single-label classification

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metric_labels = np.arange(num_classes) if num_classes is not None else None

    precision = precision_score(all_labels, all_preds, average=None, labels=metric_labels, zero_division=0)
    recall = recall_score( all_labels, all_preds, average=None, labels=metric_labels, zero_division=0)
    f1 = f1_score( all_labels, all_preds, average=None, labels=metric_labels, zero_division=0)
    precision_macro = precision_score( all_labels, all_preds, average='macro', labels=metric_labels, zero_division=0)
    recall_macro = recall_score( all_labels, all_preds, average='macro', labels=metric_labels, zero_division=0)
    f1_macro = f1_score( all_labels, all_preds, average='macro', labels=metric_labels, zero_division=0)

    return precision, recall, f1, precision_macro, recall_macro, f1_macro


def compute_accuracy_metrics_multilabel(model, loader, device, threshold=0.5):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, dict): # Some models return dicts instead of tensors
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(images["pixel_values"])
            else: # Model returns tensors directly
                images = images.to(device)
                outputs = model(images)
            probs = torch.sigmoid(outputs) # Apply sigmoid to get probabilities for multi-label classification
            preds = (probs > threshold).int().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return precision, recall, f1, precision_macro, recall_macro, f1_macro

def get_confusion_matrix_singlelabel(model, loader, device, num_classes=None):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, dict):
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(images["pixel_values"])
            else:
                images = images.to(device)
                outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    matrix_labels = np.arange(num_classes) if num_classes is not None else None
    return confusion_matrix(all_labels, all_preds, labels=matrix_labels)

def plot_confusion_matrix(model, loader, class_names, device):
    cm = get_confusion_matrix_singlelabel(model, loader, device, num_classes=len(class_names))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
