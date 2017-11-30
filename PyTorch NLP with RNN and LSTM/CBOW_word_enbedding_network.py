"""
This script deals with Continuous Bag of Words (CBOW) embedding in a NLP
context, where word embeddings are created so that each word can be predicted
by the preceding and succeeding two words. The resulted embedding weights can
then be used for other tasks. This process is often called pretraining
embeddings.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1122)

"""
We will build a CBOW model where every input is the preceding and succeeding
two words and the target is the word in the middle
"""
# building word embedding model with 2 previous contextual words
CONTEXT_SIZE = 4
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
print(raw_text[:5])

grams = [([raw_text[i - 2], raw_text[i - 1], raw_text[i + 1],
        raw_text[i + 2]], raw_text[i])
        for i in range(2, (len(raw_text) - 2))]
print(grams[:2])
vocab = set(raw_text)  # set() creates unique element tuple from the list
print(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)
VOCAB_SIZE = len(vocab)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


# Initiate the models
losses = []
loss_function = nn.NLLLoss()
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in grams:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)

losses = np.array([x.numpy()[0] for x in losses])

print(losses)
plt.plot(losses)
plt.show()


def return_label(label_ix, label_to_ix):
    """Returns original label with the index"""
    for key, value in label_to_ix.items():
        if value == label_ix:
            label = key
    return label


for i, contents in enumerate(grams):
    context, target = contents
    if i % 3 == 0:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model(context_var)
        decision = log_probs.topk(1)[1].data.numpy()[0]
        output = return_label(decision, word_to_ix)
        print("""
Input: %s
Predicted: %s
Target: %s
        """ % (repr(context), output, target))

"""
The model achieved a high level of precision with just 100 iterations of
data. This embedding layer's weights can then use used in a more advanced
language modeling system.
"""
