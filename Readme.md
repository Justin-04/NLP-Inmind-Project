## SMS Spam Classification
=========================================================================
# Preprocessing & TF-IDF Baseline
---------------------------------
Target Encoding

The original labels were converted into numerical values:
where ham : 0   spam: 1


# Text Preprocessing
--------------------
The SMS messages were cleaned before feature extraction. The preprocessing pipeline included:

Converting text to lowercase
Handling missing values
Removing unnecessary characters
Tokenizing the messages
Removing stopwords
Preserving negation words such as not, no, and nor
Applying lemmatization
Removing empty messages
Removing duplicate processed messages

The final processed text was stored in the lemmatized_text column.

# Dataset Splitting
-------------------

The dataset was divided into:
Training set —> used to fit the models (70%)
Validation set —> used for hyperparameter tuning and model selection(15%)
Test set —> kept unchanged for final evaluation
Stratified splitting was used to preserve the class distribution in each subset.(15%)

# Handling Class Imbalance
--------------------------
--> The original dataset contained more ham messages than spam messages.
Random undersampling was applied only to the "training set".
--> The validation and test sets were not balanced, so their distributions remained representative of real-world data.

# TF-IDF with Logistic Regression
---------------------------------
TF-IDF was used with Logistic Regression as the classical baseline model.
Logistic Regression learns one coefficient for every TF-IDF feature.
--> A positive coefficient pushes the prediction toward spam.
--> A negative coefficient pushes the prediction toward ham.

# Experimental TF-IDF RNN
-------------------------

RNN models expect input with three dimensions:

(batch size, sequence length, input size)

Standard TF-IDF produces a two-dimensional matrix:

(samples, TF-IDF features)

For an additional experiment, the TF-IDF matrices were reshaped into:

(samples, TF-IDF features, 1)

## The extra third dimension does not add new information, it only makes 
------------------------------------------------------------------------
# each TF-IDF feature appear as one artificial sequence step. 
-------------------------------------------------------------

The RNN therefore processes:

TF-IDF feature 1 → TF-IDF feature 2 → TF-IDF feature 3 → ...

This is not the original word order of the message.

# TF-IDF RNN Limitation
-----------------------

The TF-IDF RNN failed to learn a useful separation between ham and spam and collapsed into predicting only one class.
This occurred because:
1 TF-IDF does not preserve original word order.
2 The artificial sequence represents vocabulary columns instead of token positions.
3 The sequence is extremely long and sparse.
4 Most TF-IDF values are zero.
5 A simple RNN may suffer from vanishing gradients.

# Deep Learning with Word2Vec
-----------------------------
We ran a list of tests using three different models: Vanilla RNN, LSTM, and GRU.
Instead of using TF-IDF we used pre-trained Google News Word2Vec embeddings (300 dimensions) so the models could actually understand semantic meaning. 

# Part 1: The Vanilla RNN
-------------------------
1 Simple RNN Baseline (1 Layer, 32 Hidden, Frozen Embeddings):
--> Scored 83.15% F1. We froze the embeddings to see how well the raw structure works. It did okay but it forgets things quickly.

2 RNN Tuning 1 (2 Layers, 32 Hidden, 0.3 Dropout, Frozen):
--> Scored 78.26% F1. Because standard RNNs don't have memory gates making them deeper caused the "vanishing gradient" problem and it couldn't learn at all.

3 RNN Tuning 2 (1 Layer, 64 Hidden, 0.0 Dropout, Fine-Tuned):
--> Scored 91.46% F1. To fix the vanishing gradients we went back to 1 layer but made it wider (64 hidden) and let the model update the embeddings. This worked well.

# Part 2: Advanced Architectures (LSTM)
---------------------------------------
1 Baseline (1 Layer, 32 Hidden, Frozen):
--> Scored 83.70%. Because of its internal "memory gates" it immediately beat the Vanilla RNN.
2 Tuning 1 (2 Layers, 32 Hidden, 0.3 Dropout, Fine-Tuned):
--> Scored 91.33%. 
3 Tuning 2 (1 Layer, 64 Hidden, 0.0 Dropout, Fine-Tuned):
--> Scored 93.33%. It handled the wider 1-layer layout well improving on the 2 layers architecture.

# Part 3: Advanced Architectures (GRU)
--------------------------------------
1 Baseline (1 Layer, 32 Hidden, Frozen):
--> Scored 84.75%.
2 Tuning 1 (2 Layers, 32 Hidden, 0.3 Dropout, Fine-Tuned):
--> Scored 95.29%. Its simpler gate design made it much easier to optimize with the Word2Vec embeddings compared to the heavier LSTM.
3 Tuning 2 (1 Layer, 64 Hidden, 0.0 Dropout, Fine-Tuned):
--> Dropped to 91.01%. Showing the Unidirectional GRU strongly prefers the 2-layer setup.

# Part 4: Bidirectional Sequence Modeling (LSTM)
------------------------------------------------
We let the models read the text forwards and backwards simultaneously to get more context.
1 Baseline (1 Layer, 32 Hidden, Frozen):
--> Scored 85.42%. Reading in both directions definitely helped the baseline.
2 Tuning 1 (2 Layers, 32 Hidden, 0.3 Dropout, Fine-Tuned):
--> Scored 94.74%. It handled the extreme complexity of bidirectionality and 2 layers perfectly.
3 Tuning 2 (1 Layer, 64 Hidden, 0.0 Dropout, Fine-Tuned):
--> Dropped to 91.66%. Performance went down compared to the 2 layers architecture.

# Part 5: Bidirectional Sequence Modeling (GRU)
-----------------------------------------------
1 Baseline (1 Layer, 32 Hidden, Frozen):
--> Scored 86.81%.
2 Tuning 1 (2 Layers, 32 Hidden, 0.3 Dropout, Fine-Tuned):
--> Crashed to 88.30%. Because 2 layers + bidirectionality + fine-tuning created too many parameters for our dataset, causing it to overfit heavily.
3 Tuning 2 (1 Layer, 64 Hidden, 0.0 Dropout, Fine-Tuned):
--> Jumped to 94.67%. Making it wider and shallower (1 layer, 64 hidden) completely fixed the overfitting issue from the 2 layers architecture.

# Final Results Table
---------------------
| Architecture | Bi-dir | Layers | Hidden | Embeddings | F1-Score | Status             |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Simple RNN   | No     | 1      | 32     | Frozen     | 83.15%   | Initial Baseline   |
| Simple RNN   | No     | 2      | 32     | Frozen     | 78.26%   | Vanishing Grads    |
| Simple RNN   | No     | 1      | 64     | Fine-Tuned | 91.46%   | Optimal RNN        |
| LSTM         | No     | 1      | 32     | Frozen     | 83.70%   | Baseline           |
| LSTM         | No     | 2      | 32     | Fine-Tuned | 91.33%   | Depth focus        |
| LSTM         | No     | 1      | 64     | Fine-Tuned | 93.33%   | Width focus        |
| LSTM         | Yes    | 1      | 32     | Frozen     | 85.42%   | Baseline           |
| LSTM         | Yes    | 2      | 32     | Fine-Tuned | 94.74%   | Optimal Bi-LSTM    |
| LSTM         | Yes    | 1      | 64     | Fine-Tuned | 91.66%   | Width focus        |
| GRU          | No     | 1      | 32     | Frozen     | 84.75%   | Baseline           |
| GRU          | No     | 2      | 32     | Fine-Tuned | 95.29%   | OVERALL WINNER     |
| GRU          | No     | 1      | 64     | Fine-Tuned | 91.01%   | Width focus        |
| GRU          | Yes    | 1      | 32     | Frozen     | 86.81%   | Baseline           |
| GRU          | Yes    | 2      | 32     | Fine-Tuned | 88.30%   | Overfit            |
| GRU          | Yes    | 1      | 64     | Fine-Tuned | 94.67%   | Optimal Bi-GRU     |

# Conclusion
------------
The absolute best model was the Unidirectional GRU with:
--> 2 Layers
--> 32 Hidden Units
--> 0.3 Dropout
--> Fine-Tuned Embeddings

It hit a 95.29% F1-Score. It worked the best because its simple gate setup was perfectly balanced to learn the Word2Vec embeddings without getting too complicated and memorizing the training data.

===============================================================================================
# SMS Spam Classification with GloVe and Recurrent Neural Networks

## Objective

This part of the project uses pretrained GloVe word embeddings with recurrent neural networks to classify SMS messages as **ham** or **spam**. The tested architectures are a Simple RNN, single- and two-layer GRUs, single- and two-layer LSTMs, and a bidirectional LSTM.

The target labels are encoded as:

```text
ham  -> 0
spam -> 1
```

## Vocabulary and Token IDs

A vocabulary is built using only the training set to prevent data leakage. Every word receives an integer token ID:

```text
0           -> padding
1           -> <OOV> (unknown word)
2 and above -> training-vocabulary words
```

For example:

```text
free prize urltoken -> [12, 81, 30]
```

Token IDs are only lookup indices; their numerical values do not represent a word's meaning or importance. Validation or test words that were not seen during training receive the `<OOV>` ID.

## Sequence Length, Padding, and Truncation

The training-message length distribution was examined, and a fixed sequence length of 30 was selected because it covers more than 95% of the messages:

```python
MAX_LENGTH = 30
```

Short messages are padded with zeros:

```text
Original: [12, 81, 30]
Padded:   [12, 81, 30, 0, 0, 0, ..., 0]
```

Messages longer than 30 tokens are truncated to their first 30 token IDs. The padded model input therefore has the shape:

```text
(number_of_messages, 30)
```

### Storing the real length

Because padding makes every row appear to contain 30 tokens, the usable length of every original sequence is stored separately:

```python
def get_lengths(sequences):
    lengths = [min(len(sequence), MAX_LENGTH) for sequence in sequences]
    return torch.tensor(lengths, dtype=torch.long)

train_lengths_tensor = get_lengths(train_sequences)
validation_lengths_tensor = get_lengths(validation_sequences)
test_lengths_tensor = get_lengths(test_sequences)
```

For example:

```text
Original sequence: [12, 81, 30]
Padded sequence:   [12, 81, 30, 0, 0, ..., 0]
Stored length:     3
```

If a message originally contains more than 30 tokens, its stored usable length is 30 because the rest of the message was truncated.

## What GloVe Provides

GloVe is a pretrained word-embedding method. It provides one dense numerical vector for each known word. This notebook uses **100-dimensional** GloVe vectors:

```text
word -> 100 floating-point values
```

Conceptually:

```text
free  -> [0.27, 0.11, -0.42, ..., 0.09]
claim -> [0.14, -0.31, 0.08, ..., 0.22]
prize -> [0.35, -0.18, 0.29, ..., -0.12]
```

These values encode contextual and semantic patterns learned from a large text corpus. Words used in similar contexts generally have similar vectors.

GloVe provides a vector for each individual word. It does not directly provide a whole-message representation, preserve the order of the current message, or predict whether a message is spam. The recurrent model combines the word vectors in their original order and learns the final classification.

### Vocabulary coverage

The training vocabulary contains 2,885 entries, and 2,572 were found in the GloVe file:

```text
GloVe vocabulary coverage: 89.15%
```

Words found in GloVe receive their pretrained vectors. Words not found receive small randomly initialized vectors.

## Embedding Matrix

An embedding matrix is created with the shape:

```text
(vocabulary_size, 100)
```

Every row corresponds to one token ID:

```text
Row 0  -> zero vector used for padding
Row 1  -> vector used for <OOV>
Row 2  -> vector used for numbertoken
Row 12 -> vector used for "free"
Row 30 -> vector used for urltoken
```

Rows are initially random, the padding row is replaced with zeros, and rows for words found in GloVe are replaced with their pretrained vectors:

```python
for word, token_id in word_index.items():
    glove_vector = embeddings_index.get(word)
    if glove_vector is not None:
        embedding_matrix[token_id] = glove_vector
```

Custom placeholders such as `urltoken` and `numbertoken` are unlikely to be present in the original GloVe vocabulary. When they are not found, they keep their random vectors.



## Input Expected by RNN, GRU, and LSTM

RNN, GRU, and LSTM layers expect sequential input with three dimensions:

```text
(batch_size, sequence_length, input_size)
```

In this notebook:

```text
batch_size      = usually 32 messages
sequence_length = 30 token positions
input_size      = 100 GloVe values per token
```

The direct model input initially contains integer token IDs:

```text
token_ids shape: (32, 30)
```

The embedding layer looks up the 100-dimensional vector for every ID:

```python
embedded = self.embedding(token_ids)
```

The resulting input passed to the recurrent layer has the shape:

```text
embedded shape: (32, 30, 100)
```

The dimensions mean:

```text
32  -> SMS messages processed together
30  -> token positions in each message
100 -> numerical features representing each token
```

## Batch Size

The default batch size is 32:

```python
BATCH_SIZE = 32
```

The model processes 32 messages, calculates their predictions and combined loss, updates its weights, and then processes the next batch. During hyperparameter tuning, batch sizes of 16 and 32 were compared.

Each training batch contains three tensors:

```python
for token_ids, labels, lengths in train_loader:
```

Their approximate shapes are:

```text
token_ids -> (32, 30)
labels    -> (32,)
lengths   -> (32,)
```

## Ignoring Padding

The embedded sequences and their real lengths are packed before entering the recurrent layer:

```python
packed = nn.utils.rnn.pack_padded_sequence(
    embedded,
    lengths.cpu(),
    batch_first=True,
    enforce_sorted=False
)
```

This tells the model where each real message ends, so padding is not treated as meaningful input and the final hidden state corresponds to the last real token.

## Recurrent Architectures

### Simple RNN

The Simple RNN processes one word vector at a time and combines it with the previous hidden state:

```text
current word vector + previous hidden state -> new hidden state
```

It provides a basic sequential baseline but can struggle to preserve information over long sequences because of vanishing gradients.

### GRU

The GRU accepts the same `(batch, 30, 100)` input. Its gates control which previous information to keep, which information to forget, and which new information to add. This generally allows it to preserve useful context better than a Simple RNN while using fewer gates than an LSTM.

The initial experiments tested a single-layer GRU and a two-layer GRU with dropout.

### LSTM

The LSTM also accepts the same input. It maintains both a hidden state and a separate cell state used as longer-term memory. Its gates control what information is stored, forgotten, and exposed.

The initial experiments tested single- and two-layer LSTMs.

### Bidirectional LSTM

The bidirectional LSTM reads each message in both directions:

```text
Forward:  first token -> last token
Backward: last token  -> first token
```

The final forward and backward hidden states are concatenated. With 32 hidden units per direction, this creates a 64-value message representation.

## Classification Output

After processing the real tokens, the model uses its final hidden state as a representation of the message. A linear layer converts that representation into one raw score, called a logit:

```python
return self.output(hidden).squeeze(1)
```

During evaluation, sigmoid converts the logit into a probability, and a threshold produces the class:

```python
predictions = (torch.sigmoid(logits) >= 0.5).int()
```

```text
Probability < 0.5 -> ham
Probability >= 0.5 -> spam
```

The complete shape transformation is:

```text
Token IDs:       (32, 30)
                     |
                     v  Embedding lookup
GloVe vectors:   (32, 30, 100)
                     |
                     v  RNN / GRU / LSTM
Message vectors: (32, hidden_size)
                     |
                     v  Linear layer
Logits:          (32,)
                     |
                     v  Sigmoid and threshold
Predictions:     32 ham/spam classes
```

## Training and Early Stopping

All models use:

```text
Loss function:  BCEWithLogitsLoss
Optimizer:      Adam
Learning rate:  initially 0.001
Maximum epochs: 20
```

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable calculation. The model therefore returns raw logits during training instead of applying sigmoid itself.

Validation loss is measured after every epoch. Training stops after validation loss fails to improve for two consecutive epochs, and the weights from the best validation epoch are restored. This helps reduce overfitting.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix

F1 is particularly important because the test data contains many more ham messages than spam messages. Accuracy alone can appear high even when a model performs poorly on the minority spam class.

## Initial Architecture Comparison

The first experiments used frozen GloVe embeddings and primarily used 32 hidden units.

| Architecture | Layers | Bidirectional | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simple RNN | 1 | No | 93.16% | 66.36% | 82.95% | 73.74% |
| Single-Layer GRU | 1 | No | 95.53% | 74.55% | 93.18% | 82.83% |
| Two-Layer GRU | 2 | No | 95.53% | 74.11% | 94.32% | **83.00%** |
| Single-Layer LSTM | 1 | No | 94.08% | 67.48% | 94.32% | 78.67% |
| Two-Layer LSTM | 2 | No | 95.13% | 72.57% | 93.18% | 81.59% |
| Bidirectional LSTM | 1 | Yes | 93.68% | 66.39% | 92.05% | 77.14% |

The two-layer GRU achieved the best initial F1 score at 83.00%.

## Hyperparameter Tuning

Configurations were compared using validation F1. The tested hyperparameters included:

- Hidden sizes: 16, 32, and 64
- Learning rates: 0.001 and 0.0005
- Batch sizes: 16 and 32
- Recurrent layers: 1 and 2
- Dropout: 0.0 and 0.3

Each tuning trial used at most five epochs with early stopping based on validation loss.

| Model | Hidden Size | Learning Rate | Batch Size | Layers | Dropout | Validation F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RNN | 64 | 0.001 | 32 | 1 | 0.0 | 79.57% |
| GRU | 32 | 0.001 | 32 | 1 | 0.0 | 83.80% |
| LSTM | 32 | 0.001 | 16 | 1 | 0.0 | **85.88%** |
| Bidirectional LSTM | 64 | 0.001 | 32 | 1 | 0.0 | 83.33% |

The single-layer LSTM with batch size 16 achieved the highest validation F1 during tuning.

## Final Tuned Results

Fresh models were created using the best validation configuration for each architecture, retrained with early stopping, and evaluated on the held-out test set.

| Model | Hidden | Batch | Layers | Best Epoch | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GRU | 32 | 32 | 1 | 5 | 95.53% | 74.55% | 93.18% | **82.83%** |
| Bidirectional LSTM | 64 | 32 | 1 | 3 | **95.66%** | **76.70%** | 89.77% | 82.72% |
| LSTM | 32 | 16 | 1 | 5 | 94.74% | 71.43% | 90.91% | 80.00% |
| RNN | 64 | 32 | 1 | 5 | 93.68% | 66.13% | **93.18%** | 77.36% |

## Conclusion

The final tuned GRU achieved the best test F1 score:

```text
Architecture:  Single-layer GRU
Hidden size:   32
Learning rate: 0.001
Batch size:    32
Dropout:       0.0
Best epoch:    5
Accuracy:      95.53%
Precision:     74.55%
Recall:        93.18%
F1 score:      82.83%
```

The tuned bidirectional LSTM was very close, with an F1 score of 82.72%, and achieved the highest accuracy and precision. Overall, the GRU provided the best balance between precision and recall in the final GloVe experiments. Its gating mechanism captured sequential information more effectively than the Simple RNN while remaining simpler than the LSTM architectures.

