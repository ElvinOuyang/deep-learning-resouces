"""
The LSTM network in this script takes in the word sequence
and assigns most likely POS tag as output
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1122)


def prepare_sequence(seq, to_ix):
    """Change tokenized word list into index sequence LongTensor"""
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
HIDDEN_LAYERS = 1
BATCH_SIZE = 1


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 hidden_layers, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            hidden_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return(autograd.Variable(torch.zeros(
               self.hidden_layers, self.batch_size, self.hidden_dim)),
               autograd.Variable(torch.zeros(
                self.hidden_layers, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(self.batch_size, len(sentence), -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


model = LSTMTagger(
    EMBEDDING_DIM,
    HIDDEN_DIM,
    len(word_to_ix),
    len(tag_to_ix),
    HIDDEN_LAYERS,
    BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def return_label(label_ix, label_to_ix):
    """
Returns original labels with the index as a list
label_ix: torch.tensor with shape (sequence * tag_len)
label_to_ix: dictionary that matches tags with numbers
    """
    decisions = label_ix.topk(1)[1].data.numpy().tolist()
    tags = []
    for decision in decisions:
        for key, value in label_to_ix.items():
            if value == decision[0]:
                tags.append(key)
    return tags


# print initial tags from the model
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
tag_labels = return_label(tag_scores, tag_to_ix)
tag_dict = {training_data[0][0][i]: tag_labels[i]
            for i in range(len(training_data[0][0]))}
print(tag_dict)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        # initiate hidden layer
        model.hidden = model.init_hidden()
        # prepare inputs/targets
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# print tags after the training
inputs = prepare_sequence(training_data[1][0], word_to_ix)
tag_scores = model(inputs)
tag_labels = return_label(tag_scores, tag_to_ix)
tag_dict = {training_data[1][0][i]: tag_labels[i]
            for i in range(len(training_data[1][0]))}
print(tag_dict)
