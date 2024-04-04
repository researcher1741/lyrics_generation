from torch import nn

class RhymingProbe(nn.Module):
    def __init__(self, hidden_size, activation_fun=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if activation_fun is None or activation_fun == 'linear':
            self.activation_fun = lambda x : x
        elif activation_fun == 'sigmoid':
            self.activation_fun = nn.Sigmoid()
        elif activation_fun == 'tanh':
            self.activation_fun = nn.Tanh()
        else:
            RuntimeError('Activation function ' + activation_fun +' not supported, please choose from \{linear, sigmoid, tanh\}')

        self.linear = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_embeddings, labels = None):
        logits = self.activation_fun(self.linear(input_embeddings))
        loss = None
        if labels:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels)
        return {'logits': logits, 'loss': loss}