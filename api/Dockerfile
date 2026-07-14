FROM python:3.11-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install/packages \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install/packages /usr/local

COPY app.py         .
COPY rnn_model.py   .
COPY lstm_model.py  .
COPY gru_model.py   .
COPY vocab.json     .
COPY vocab_w2v.json .
COPY best_rnn.pth   .
COPY best_lstm.pth  .
COPY best_gru.pth   .
COPY RNN_Word2vec.pth  .
COPY LSTM_Word2Vec.pth .
COPY GRU_Word2vec.pth  .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
