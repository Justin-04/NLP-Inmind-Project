import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data, build_vocab, get_dataloaders, load_pretrained_embeddings
from rnn import RNN
from lstm import LSTM
from gru import GRU
from evaluation import evaluate_model, find_best_threshold, plot_loss_curves, perform_error_analysis

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BIDIRECTIONAL = False
DROPOUT = 0.0
FINE_TUNE_EMBEDDINGS = True

SEED = 42
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import os

# Data load
print("\n Preparing Data")
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "data", "train_balanced.csv")
val_path = os.path.join(script_dir, "..", "data", "validation.csv")
test_path = os.path.join(script_dir, "..", "data", "test.csv")
w2v_path = r"C:\Users\MA21\Documents\nlp_data\GoogleNews-vectors-negative300.bin.gz"

train_df, val_df, test_df = load_data(train_path, val_path, test_path)
vocab = build_vocab(train_df["lemmatized_text"])

embed_dim = 300
pretrained_weights = load_pretrained_embeddings(vocab, w2v_path, embed_dim)

train_loader, val_loader, test_loader = get_dataloaders(
    train_df, val_df, test_df, vocab, max_len=50, batch_size=BATCH_SIZE
)

# Training funct
def forward_model(model, sequences, lengths=None):
    if lengths is None: return model(sequences)
    try: return model(sequences, lengths)
    except TypeError: return model(sequences)

from datetime import datetime
import json

def train_and_evaluate(model, model_name):
    # Setup Logging Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(script_dir, "..", "results", f"{model_name.replace(' ', '_')}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save Config
    config = {
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "NUM_LAYERS": NUM_LAYERS,
        "BIDIRECTIONAL": BIDIRECTIONAL,
        "DROPOUT": DROPOUT,
        "FINE_TUNE_EMBEDDINGS": FINE_TUNE_EMBEDDINGS
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print(f"\n Training: {model_name}\n")
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for sequences, lengths, labels in train_loader:
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = forward_model(model, sequences, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for seq, length, labels in val_loader:
                seq, length, labels = seq.to(device), length.to(device), labels.to(device)
                preds = forward_model(model, seq, length)
                val_loss = criterion(preds, labels)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

    print("\n Training done")
    plot_loss_curves(history, save_path=save_path)

    print("\n Finding best Threshold on validation set")
    val_threshold_result = find_best_threshold(model, val_loader, device=device)
    best_thresh = val_threshold_result['threshold']
    
    print(f"\n Final Evaluation on Test Set (Threshold: {best_thresh:.2f})")
    acc, prec, rec, f1, cm = evaluate_model(model, test_loader, device=device, threshold=best_thresh, title=f'{model_name} Test Set', save_path=save_path)
    
    print("\n Error Analysis")
    perform_error_analysis(model, test_loader, test_df["lemmatized_text"], device=device, threshold=best_thresh, save_path=save_path)
    
    # Save Metrics and Model
    metrics = {
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(rec),
        'test_f1': float(f1),
        'best_threshold': float(best_thresh)
    }
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    print(f"\n Logs and results saved to: {save_path}")

# RNN
rnn_model = RNN(
    vocab_size=len(vocab), embedding_dim=embed_dim, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT,
    pretrained_weights=pretrained_weights, fine_tune=FINE_TUNE_EMBEDDINGS
)
#train_and_evaluate(rnn_model, "RNN Baseline")


# LSTM
lstm_model = LSTM(
    vocab_size=len(vocab), embedding_dim=embed_dim, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT,
    pretrained_weights=pretrained_weights, fine_tune=FINE_TUNE_EMBEDDINGS
)
train_and_evaluate(lstm_model, "LSTM Model")


# GRU
gru_model = GRU(
    vocab_size=len(vocab), embedding_dim=embed_dim, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=DROPOUT,
    pretrained_weights=pretrained_weights, fine_tune=FINE_TUNE_EMBEDDINGS
)
#train_and_evaluate(gru_model, "GRU Model")
