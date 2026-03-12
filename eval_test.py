"""Evaluate trained FFNN and RNN models on the test set."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import string
import pickle
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)

unk = '<UNK>'

# ── FFNN ──────────────────────────────────────────────────────────────────────

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden = self.activation(self.W1(input_vector))
        output = self.W2(hidden)
        predicted_vector = self.softmax(output)
        return predicted_vector


def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data_ffnn(train_path, val_path, test_path):
    with open(train_path) as f:
        training = json.load(f)
    with open(val_path) as f:
        validation = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    tra = [(e["text"].split(), int(e["stars"] - 1)) for e in training]
    val = [(e["text"].split(), int(e["stars"] - 1)) for e in validation]
    tst = [(e["text"].split(), int(e["stars"] - 1)) for e in test]
    return tra, val, tst


def train_ffnn(hidden_dim, epochs, train_path, val_path, test_path):
    print(f"\n=== FFNN hidden_dim={hidden_dim}, epochs={epochs} ===")
    train_data, valid_data, test_data = load_data_ffnn(train_path, val_path, test_path)

    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data  = convert_to_vector_representation(test_data,  word2index)

    model = FFNN(input_dim=len(vocab), h=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    results = []
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        correct = total = 0
        minibatch_size = 16
        N = len(train_data)
        for mb in range(N // minibatch_size):
            optimizer.zero_grad()
            loss = None
            for ei in range(minibatch_size):
                iv, gl = train_data[mb * minibatch_size + ei]
                pv = model(iv)
                pl = torch.argmax(pv)
                correct += int(pl == gl)
                total += 1
                el = model.compute_Loss(pv.view(1, -1), torch.tensor([gl]))
                loss = el if loss is None else loss + el
            (loss / minibatch_size).backward()
            optimizer.step()
        train_acc = correct / total

        model.eval()
        correct = total = 0
        for iv, gl in valid_data:
            pv = model(iv)
            correct += int(torch.argmax(pv) == gl)
            total += 1
        val_acc = correct / total

        print(f"  Epoch {epoch+1}: train={train_acc:.4f}  val={val_acc:.4f}")
        results.append((epoch+1, train_acc, val_acc))

    # Test evaluation
    model.eval()
    correct = total = 0
    for iv, gl in test_data:
        pv = model(iv)
        correct += int(torch.argmax(pv) == gl)
        total += 1
    test_acc = correct / total
    print(f"  Test accuracy: {test_acc:.4f}")
    return results, test_acc


# ── RNN ───────────────────────────────────────────────────────────────────────

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        output, hidden = self.rnn(inputs)
        output = self.W(output)
        output = torch.sum(output, dim=0)
        predicted_vector = self.softmax(output)
        return predicted_vector


def load_data_rnn(train_path, val_path, test_path):
    with open(train_path) as f:
        training = json.load(f)
    with open(val_path) as f:
        validation = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    tra = [(e["text"].split(), int(e["stars"] - 1)) for e in training]
    val = [(e["text"].split(), int(e["stars"] - 1)) for e in validation]
    tst = [(e["text"].split(), int(e["stars"] - 1)) for e in test]
    return tra, val, tst


def train_rnn(hidden_dim, train_path, val_path, test_path):
    print(f"\n=== RNN hidden_dim={hidden_dim} (early stopping) ===")
    train_data, valid_data, test_data = load_data_rnn(train_path, val_path, test_path)

    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    model = RNN(50, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0
    results = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        correct = total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = loss_count = 0

        for mb in tqdm(range(N // minibatch_size), desc=f"Epoch {epoch+1} train"):
            optimizer.zero_grad()
            loss = None
            for ei in range(minibatch_size):
                input_words, gold_label = train_data[mb * minibatch_size + ei]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                el = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                correct += int(torch.argmax(output) == gold_label)
                total += 1
                loss = el if loss is None else loss + el
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        train_acc = correct / total

        model.eval()
        correct = total = 0
        for input_words, gold_label in tqdm(valid_data, desc=f"Epoch {epoch+1} val"):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            correct += int(torch.argmax(output) == gold_label)
            total += 1
        val_acc = correct / total

        print(f"  Epoch {epoch+1}: train={train_acc:.4f}  val={val_acc:.4f}")
        results.append((epoch + 1, train_acc, val_acc))

        if val_acc < last_validation_accuracy and train_acc > last_train_accuracy:
            stopping_condition = True
            print(f"  Early stopping! Best val acc: {last_validation_accuracy:.4f}")
        else:
            last_validation_accuracy = val_acc
            last_train_accuracy = train_acc
        epoch += 1

    # Test evaluation using model at early-stop point
    model.eval()
    correct = total = 0
    for input_words, gold_label in tqdm(test_data, desc="Test"):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        correct += int(torch.argmax(output) == gold_label)
        total += 1
    test_acc = correct / total
    print(f"  Test accuracy: {test_acc:.4f}")
    return results, test_acc


if __name__ == "__main__":
    TRAIN = "training.json"
    VAL   = "validation.json"
    TEST  = "test.json"

    # FFNN experiments
    ffnn_100, ffnn_100_test = train_ffnn(100, 5, TRAIN, VAL, TEST)
    ffnn_200, ffnn_200_test = train_ffnn(200, 5, TRAIN, VAL, TEST)

    # RNN experiments
    rnn_32, rnn_32_test = train_rnn(32, TRAIN, VAL, TEST)
    rnn_64, rnn_64_test = train_rnn(64, TRAIN, VAL, TEST)

    print("\n========== SUMMARY ==========")
    print("FFNN h=100:", ffnn_100, "test:", ffnn_100_test)
    print("FFNN h=200:", ffnn_200, "test:", ffnn_200_test)
    print("RNN  h=32:", rnn_32,  "test:", rnn_32_test)
    print("RNN  h=64:", rnn_64,  "test:", rnn_64_test)
