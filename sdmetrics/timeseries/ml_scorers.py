"""Machine Learning Detection based metrics for Time Series."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _stack(row):
    return np.stack(row.to_numpy())  # noqa


def _to_numpy(dataframe):
    return np.stack(dataframe.apply(_stack, axis=1))  # noqa


def _x_to_packed_sequence(X, torch):
    sequences = []
    for _, row in X.iterrows():
        sequence = []
        for _, values in row.iteritems():
            sequence.append(values)

        sequences.append(torch.FloatTensor(sequence).T)

    return torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)


def lstm_classifier(X_train, X_test, y_train, y_test):
    """ML Scorer based on a simple LSTM based NN implemented using torch."""
    try:
        import torch
    except ImportError:
        raise ImportError('Please install torch with `pip install torch`')

    input_dim = len(X_train.columns)
    output_dim = len(set(y_train))
    hidden_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = torch.nn.LSTM(input_dim, hidden_dim).to(device)
    linear = torch.nn.Linear(hidden_dim, output_dim).to(device)

    X_train = _x_to_packed_sequence(X_train, torch).to(device)
    X_test = _x_to_packed_sequence(X_test, torch).to(device)

    transformer = LabelEncoder()
    column = 'target'
    y_train = pd.DataFrame(y_train, columns=[column])
    y_test = pd.DataFrame(y_test, columns=[column])

    y_train = transformer.fit_transform(y_train[column])
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(transformer.transform(y_test)).to(device)

    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=1e-2)

    for _ in range(1024):
        _, (y, _) = lstm(X_train)
        y_pred = linear(y[0])
        loss = torch.nn.functional.cross_entropy(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, (y, _) = lstm(X_test)
    y_pred = linear(y[0])
    y_pred = torch.argmax(y_pred, axis=1)
    return (y_test == y_pred).sum().item() / len(y_test)
