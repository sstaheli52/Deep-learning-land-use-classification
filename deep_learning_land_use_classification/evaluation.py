from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_accuracy_metrics_singlelabel(model, loader, device):
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
    
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return precision, recall, f1, precision_macro, recall_macro, f1_macro


def compute_accuracy_metrics_multilabel(model, loader, device, threshold=0.5):
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
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())  # explicit .cpu() added

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return precision, recall, f1, precision_macro, recall_macro, f1_macro

def get_confusion_matrix_singlelabel(model, loader, device):
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

    return confusion_matrix(all_labels, all_preds)
