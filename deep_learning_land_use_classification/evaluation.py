from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np

def compute_accuracy_metrics(model, loader, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs > threshold).int().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-class metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Macro averages (treat all classes equally)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return precision, recall, f1, precision_macro, recall_macro, f1_macro