import json
import string
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rnn_model  import GloveRNN
from lstm_model import GloveLSTM
from gru_model  import GloveGRU


class W2vRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hn = self.rnn(packed)
        else:
            _, hn = self.rnn(embedded)
        return self.fc(hn[-1]).squeeze(-1)


class W2vLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(embedded)
        last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(last_hidden).squeeze(-1)


class W2vGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hn = self.gru(packed)
        else:
            _, hn = self.gru(embedded)
        return self.fc(hn[-1]).squeeze(-1)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOVE_MAX_SEQ = 30

W2V_MAX_SEQ = 50

MODEL_REGISTRY = {
    "rnn": {
        "class":       GloveRNN,
        "weight_file": "best_rnn.pth",
        "embedding":   "glove",
        "name":        "Simple RNN — GloVe",
        "description": "Vanilla single-layer RNN with GloVe 100-dim embeddings.",
        "metrics": {
            "accuracy":  0.9303,
            "precision": 0.6549,
            "recall":    0.8409,
            "f1":        0.7363,
        },
        "hyperparameters": {
            "layers":          1,
            "hidden_size":     32,
            "bidirectional":   False,
            "embedding_dim":   100,
            "embedding_type":  "GloVe (frozen)",
            "max_seq_length":  GLOVE_MAX_SEQ,
        },
    },
    "lstm": {
        "class":       GloveLSTM,
        "weight_file": "best_lstm.pth",
        "embedding":   "glove",
        "name":        "Bidirectional LSTM — GloVe",
        "description": "Bidirectional single-layer LSTM with GloVe 100-dim embeddings.",
        "metrics": {
            "accuracy":  0.9566,
            "precision": 0.7670,
            "recall":    0.8977,
            "f1":        0.8272,
        },
        "hyperparameters": {
            "layers":          1,
            "hidden_size":     64,
            "bidirectional":   True,
            "embedding_dim":   100,
            "embedding_type":  "GloVe (frozen)",
            "max_seq_length":  GLOVE_MAX_SEQ,
        },
    },
    "gru": {
        "class":       GloveGRU,
        "weight_file": "best_gru.pth",
        "embedding":   "glove",
        "name":        "GRU — GloVe",
        "description": "Single-layer GRU with GloVe 100-dim embeddings.",
        "metrics": {
            "accuracy":  0.9553,
            "precision": 0.7455,
            "recall":    0.9318,
            "f1":        0.8283,
        },
        "hyperparameters": {
            "layers":          1,
            "hidden_size":     32,
            "bidirectional":   False,
            "embedding_dim":   100,
            "embedding_type":  "GloVe",
            "max_seq_length":  GLOVE_MAX_SEQ,
        },
    },

    "rnn_w2v": {
        "class":       W2vRNN,
        "weight_file": "RNN_Word2vec.pth",
        "embedding":   "w2v",
        "name":        "RNN — Word2Vec",
        "description": "Single-layer RNN with Word2Vec 300-dim embeddings.",
        "metrics": {
            "accuracy":  0.9816,
            "precision": 0.9740,
            "recall":    0.8621,
            "f1":        0.9146,
        },
        "hyperparameters": {
            "layers":          1,
            "hidden_size":     64,
            "bidirectional":   False,
            "embedding_dim":   300,
            "embedding_type":  "Word2Vec",
            "max_seq_length":  W2V_MAX_SEQ,
        },
    },
    "lstm_w2v": {
        "class":       W2vLSTM,
        "weight_file": "LSTM_Word2Vec.pth",
        "embedding":   "w2v",
        "name":        "LSTM — Word2Vec",
        "description": "2-layer bidirectional LSTM with Word2Vec 300-dim embeddings.",
        "metrics": {
            "accuracy":  0.9882,
            "precision": 0.9643,
            "recall":    0.9310,
            "f1":        0.9474,
        },
        "hyperparameters": {
            "layers":          2,
            "hidden_size":     32,
            "bidirectional":   True,
            "embedding_dim":   300,
            "embedding_type":  "Word2Vec (fine-tuned)",
            "max_seq_length":  W2V_MAX_SEQ,
        },
    },
    "gru_w2v": {
        "class":       W2vGRU,
        "weight_file": "GRU_Word2vec.pth",
        "embedding":   "w2v",
        "name":        "GRU — Word2Vec",
        "description": "2-layer GRU with Word2Vec 300-dim embeddings.",
        "metrics": {
            "accuracy":  0.9895,
            "precision": 0.9759,
            "recall":    0.9310,
            "f1":        0.9529,
        },
        "hyperparameters": {
            "layers":          2,
            "hidden_size":     32,
            "bidirectional":   False,
            "embedding_dim":   300,
            "embedding_type":  "Word2Vec",
            "max_seq_length":  W2V_MAX_SEQ,
        },
    },
}


glove_vocab: dict = {}   
w2v_vocab:   dict = {}   
loaded_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global glove_vocab, w2v_vocab, loaded_models

    try:
        with open("vocab.json", "r", encoding="utf-8") as f:
            glove_vocab = json.load(f)
        print(f"[startup] GloVe vocab loaded: {len(glove_vocab)} words")
    except FileNotFoundError:
        raise RuntimeError("vocab.json not found in api/ folder.")

    try:
        with open("vocab_w2v.json", "r", encoding="utf-8") as f:
            w2v_vocab = json.load(f)
        print(f"[startup] Word2Vec vocab loaded: {len(w2v_vocab)} words")
    except FileNotFoundError:
        raise RuntimeError("vocab_w2v.json not found in api/ folder.")

    glove_vocab_size = len(glove_vocab) + 1   # GloVe: +1 for padding row 0

    for model_id, cfg in MODEL_REGISTRY.items():
        emb_dim = cfg["hyperparameters"]["embedding_dim"]
        hidden  = cfg["hyperparameters"]["hidden_size"]

        try:
            state = torch.load(cfg["weight_file"], map_location=DEVICE, weights_only=True)
            vocab_size = state["embedding.weight"].shape[0]

            m = cfg["class"](vocab_size=vocab_size, embedding_dim=emb_dim, hidden_size=hidden)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            loaded_models[model_id] = m
            print(f"[startup] Loaded: {cfg['name']} ({cfg['weight_file']}) vocab_size={vocab_size}")
        except FileNotFoundError:
            print(f"[startup] WARNING: '{cfg['weight_file']}' not found — {model_id} skipped.")

    yield  

    loaded_models.clear()


app = FastAPI(
    title="SMS Spam Detection API",
    description="Spam classifier — GloVe models: rnn, lstm, gru | Word2Vec models: rnn_w2v, lstm_w2v, gru_w2v",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#   GloVe  — lemmatized text, strip punctuation, OOV id = 1, max_len = 30
#   Word2Vec — lemmatized text, no punctuation strip (tokens already clean),
#              UNK id = 1 (<UNK> key), max_len = 50
def preprocess_glove(text: str):
    text_clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    words      = text_clean.split()
    token_ids  = [glove_vocab.get(w, 1) for w in words]   # 1 = <OOV>
    return _pad_and_tensorise(token_ids, GLOVE_MAX_SEQ)


def preprocess_w2v(text: str):
    """Tokenise for Word2Vec models (max_len=50, UNK=1)."""
    words     = text.lower().split()
    token_ids = [w2v_vocab.get(w, w2v_vocab.get("<UNK>", 1)) for w in words]
    return _pad_and_tensorise(token_ids, W2V_MAX_SEQ)


def _pad_and_tensorise(token_ids: list, max_len: int):
    real_length = min(len(token_ids), max_len)
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    input_tensor  = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    length_tensor = torch.tensor([real_length], dtype=torch.long)
    return input_tensor, length_tensor



class PredictRequest(BaseModel):
    text:  str
    model: Optional[str] = "gru"


class PredictResponse(BaseModel):
    text:             str
    model_used:       str
    model_name:       str
    prediction:       str
    spam_probability: float
    ham_probability:  float
    confidence_score: float
    label:            int



@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model_id = (payload.model or "gru").lower()

    if model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_id}'. Choose from: {list(MODEL_REGISTRY)}",
        )
    if model_id not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' weights not loaded (file missing at startup).",
        )

    cfg = MODEL_REGISTRY[model_id]
    m   = loaded_models[model_id]

    preprocess = preprocess_w2v if cfg["embedding"] == "w2v" else preprocess_glove

    try:
        input_tensor, lengths = preprocess(payload.text)
        with torch.no_grad():
            logits    = m(input_tensor, lengths)
            spam_prob = torch.sigmoid(logits).item()

        ham_prob   = 1.0 - spam_prob
        prediction = "spam" if spam_prob >= 0.5 else "ham"
        confidence = spam_prob if prediction == "spam" else ham_prob

        return PredictResponse(
            text             = payload.text,
            model_used       = model_id,
            model_name       = cfg["name"],
            prediction       = prediction,
            spam_probability = round(spam_prob, 4),
            ham_probability  = round(ham_prob, 4),
            confidence_score = round(confidence, 4),
            label            = 1 if prediction == "spam" else 0,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model_info")
def get_model_info(model: str = "gru"):
    model_id = model.lower()
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'.")

    cfg = MODEL_REGISTRY[model_id]
    return {
        "model_id":    model_id,
        "name":        cfg["name"],
        "description": cfg["description"],
        "is_loaded":   model_id in loaded_models,
        "performance_metrics": {
            "accuracy":  cfg["metrics"]["accuracy"],
            "precision": cfg["metrics"]["precision"],
            "recall":    cfg["metrics"]["recall"],
            "f1_score":  cfg["metrics"]["f1"],
        },
        "hyperparameters": {
            **cfg["hyperparameters"],
            "vocab_size": (
                len(glove_vocab) + 1
                if cfg["embedding"] == "glove"
                else len(w2v_vocab)
            ),
        },
    }


@app.get("/list_models")
def list_models():
    return [
        {
            "id":          mid,
            "name":        cfg["name"],
            "description": cfg["description"],
            "embedding":   cfg["embedding"],
            "is_loaded":   mid in loaded_models,
            "metrics": {
                "accuracy":  cfg["metrics"]["accuracy"],
                "precision": cfg["metrics"]["precision"],
                "recall":    cfg["metrics"]["recall"],
                "f1_score":  cfg["metrics"]["f1"],
            },
        }
        for mid, cfg in MODEL_REGISTRY.items()
    ]


@app.get("/health")
def health_check():
    return {
        "status":        "healthy" if loaded_models else "degraded",
        "loaded_models": list(loaded_models.keys()),
        "device":        str(DEVICE),
    }
