import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import gensim
import os

def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df["labeled"] = train_df["label"].map({"spam": 1, "ham": 0})
    val_df["labeled"] = val_df["label"].map({"spam": 1, "ham": 0})
    test_df["labeled"] = test_df["label"].map({"spam": 1, "ham": 0})
    
    return train_df, val_df, test_df

def build_vocab(train_texts):
    # Builds a vocabulary mapping word to integer index from training texts
    all_words = " ".join(train_texts.fillna("").astype(str)).split()
    word_counts = Counter(all_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    
    # Index 0 is reserved for PAD, Index 1 is reserved for UNK
    vocab = {word: idx + 2 for idx, word in enumerate(sorted_words)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def text_to_padded_sequences(texts, vocab, max_len=50):
    # Converts a list of strings into padded tensor sequences
    sequences = []
    lengths = []
    for text in texts:
        tokens = str(text).split()
        token_ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens][:max_len]
        if not token_ids:
            token_ids = [vocab["<PAD>"]]
        lengths.append(len(token_ids))
        sequences.append(torch.tensor(token_ids, dtype=torch.long))
        
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab["<PAD>"])
    return padded_sequences, torch.tensor(lengths, dtype=torch.long)

def load_pretrained_embeddings(vocab, path_to_model, embed_dim=300):
    # Loads Word2Vec embeddings and creates a weight matrix for PyTorch Embedding layer
    print(f"Loading external Word2Vec model from {path_to_model}...")
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    print("Word2Vec Model loaded successfully!")

    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    for word, i in vocab.items():
        if word in w2v_model:
            embedding_matrix[i] = w2v_model[word]

    pretrained_weights = torch.tensor(embedding_matrix, dtype=torch.float32)
    pretrained_weights[vocab["<PAD>"]] = 0.0
    pretrained_weights[vocab["<UNK>"]] = torch.randn(embed_dim) * 0.05 # Random vector for Unknowns
    
    return pretrained_weights

def get_dataloaders(train_df, val_df, test_df, vocab, max_len=50, batch_size=32):
    # Converts dataframes to PyTorch DataLoaders, with class weighting for training
    # Convert text to sequences
    X_train, train_lengths = text_to_padded_sequences(train_df["lemmatized_text"], vocab, max_len)
    y_train = torch.tensor(train_df["labeled"].values, dtype=torch.float32)

    X_val, val_lengths = text_to_padded_sequences(val_df["lemmatized_text"], vocab, max_len)
    y_val = torch.tensor(val_df["labeled"].values, dtype=torch.float32)

    X_test, test_lengths = text_to_padded_sequences(test_df["lemmatized_text"], vocab, max_len)
    y_test = torch.tensor(test_df["labeled"].values, dtype=torch.float32)

    # Create Datasets
    train_dataset = TensorDataset(X_train, train_lengths, y_train)
    val_dataset = TensorDataset(X_val, val_lengths, y_val)
    test_dataset = TensorDataset(X_test, test_lengths, y_test)

    # Weighted Sampler for training (handles imbalanced classes)
    train_label_counts = train_df["labeled"].value_counts().to_dict()
    sample_weights = train_df["labeled"].map(lambda label: 1.0 / train_label_counts[label]).astype(np.float32).to_numpy()
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
