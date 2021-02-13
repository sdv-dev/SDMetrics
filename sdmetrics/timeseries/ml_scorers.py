"""Machine Learning Detection based metrics for Time Series."""

import rdt
import torch
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier

try:
    from sktime.transformers.series_as_features.compose import ColumnConcatenator
except (ImportError, AttributeError):
    from sktime.transformations.panel.compose import ColumnConcatenator


def tsf_classifier(X_train, X_test, y_train, y_test):
    """ML Scorer based on sktime pipeline with a TimeSeriesForestClassifier."""
    steps = [
        ('concatenate', ColumnConcatenator()),
        ('classify', TimeSeriesForestClassifier(n_estimators=100))
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def _x_to_packed_sequence(X):
    sequences = []
    for _, row in X.iterrows():
        sequence = []
        for _, values in row.iteritems():
            sequence.append(values)

        sequences.append(torch.FloatTensor(sequence).T)

    return torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)


def lstm_classifier(X_train, X_test, y_train, y_test):
    """ML Scorer based on a simple LSTM based NN implemented using torch."""
    input_dim = len(X_train.columns)
    output_dim = len(set(y_train))
    hidden_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = torch.nn.LSTM(input_dim, hidden_dim).to(device)
    linear = torch.nn.Linear(hidden_dim, output_dim).to(device)

    X_train = _x_to_packed_sequence(X_train).to(device)
    X_test = _x_to_packed_sequence(X_test).to(device)

    transformer = rdt.transformers.categorical.LabelEncodingTransformer()
    y_train = torch.LongTensor(transformer.fit_transform(y_train).to_numpy()).to(device)
    y_test = torch.LongTensor(transformer.transform(y_test).to_numpy()).to(device)

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
