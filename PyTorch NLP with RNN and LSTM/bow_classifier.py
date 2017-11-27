"""
This is a short script that assigns index values to corpus words and target
labels, runs a basic logistic regression using Linear layer and output log
probabilites using LogSoftMax layer.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1122)

# build a logistic regression bow classifier
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# create word-to-ix as a "dictionary" of each word's index
# each vocab has a unique index represented in the matrix
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


# takes in vector of the len(word_to_ix), spits out log_probs of each class
class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))


# turn a sentence (list of words) into a vector using word_to_ix
# alternatively, can use CountVectorizer
# return data is a FloatTensor of shape 1 (batch size) by len(word_to_ix)
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


# turn the labels into a dictionary where each label has a unique index
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


# initiate the model and check what's inside
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
for param in model.parameters():
    print(param)

# Run the model once to check the output (log_probs)
sample = data[0]
print(sample)
bow_vector = make_bow_vector(sample[0], word_to_ix)
print(bow_vector)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)
label_to_ix = {'SPANISH': 0, 'ENGLISH': 1}

# Make predictions on the test data
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad()
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

print(next(model.parameters())[:, word_to_ix["creo"]])
