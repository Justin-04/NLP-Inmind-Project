import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def _forward_model(model, sequences, lengths=None):
    if lengths is None:
        return model(sequences)
    try:
        return model(sequences, lengths)
    except TypeError:
        return model(sequences)


def collect_predictions(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                sequences, lengths, targets = batch
            else:
                sequences, targets = batch
                lengths = None

            sequences = sequences.to(device)
            targets = targets.to(device)
            if lengths is not None:
                lengths = lengths.to(device)

            logits = _forward_model(model, sequences, lengths)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    return np.asarray(all_probs), np.asarray(all_targets)


def evaluate_model(model, data_loader, device='cpu', threshold=0.5, title='Evaluation Dataset', save_path=None):
    probs, targets = collect_predictions(model, data_loader, device=device)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    cm = confusion_matrix(targets, preds)

    print(f"{title.upper()} RESULTS :")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{title} Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_').lower()}_confusion_matrix.png"))
    plt.show()

    return acc, prec, rec, f1, cm


def find_best_threshold(model, data_loader, device='cpu', thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    probs, targets = collect_predictions(model, data_loader, device=device)

    best = None
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)

        candidate = {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

        # Prefer the one with the highest F1 if tied prefer higher recall
        if best is None or candidate['f1'] > best['f1'] or (candidate['f1'] == best['f1'] and candidate['recall'] > best['recall']):
            best = candidate

    return best

def plot_loss_curves(history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "loss_curves.png"))
    plt.show()


def perform_error_analysis(model, data_loader, raw_texts, device='cpu', threshold=0.5, save_path=None):
    probs, targets = collect_predictions(model, data_loader, device=device)
    preds = (probs >= threshold).astype(int)
    
    errors = []
    
    for i in range(len(targets)):
        if preds[i] != targets[i]:
            errors.append({
                'Text': raw_texts.iloc[i] if isinstance(raw_texts, pd.Series) else raw_texts[i],
                'Actual_Label': 'Spam' if targets[i] == 1 else 'Ham',
                'Predicted_Label': 'Spam' if preds[i] == 1 else 'Ham',
                'Spam_Probability': probs[i]
            })
            
    error_df = pd.DataFrame(errors)
    
    if not error_df.empty:
        print(f"\nFound {len(error_df)} misclassified datapoints out of {len(targets)} total datapoints.")
        if save_path:
            csv_path = os.path.join(save_path, "error_analysis.csv")
            error_df.to_csv(csv_path, index=False)
            print(f"Saved error analysis to: {csv_path}")
    else:
        print("\nNo errors found.")
        
    return error_df