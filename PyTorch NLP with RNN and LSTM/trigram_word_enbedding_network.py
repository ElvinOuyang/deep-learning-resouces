"""
This script deals with trigram word embedding in a NLP context, where words are
being represented by latent features in the matrix to simulate semantic
similarities. The presumption of this method is that words of close meanings
tend to appear near each other.
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
We will build a 3-gram model where every input is the previous two words,
and the target is the third word
"""
# building word embedding model with 2 previous contextual words
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.
""".split()
print(test_sentence[:5])


# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
# representing the word, and its two preceding words
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
print(trigrams[:2])
vocab = set(test_sentence)  # set() creates unique element tuple from the list
print(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)

# let's quickly look at how the nn.Embedding() layer works
emd = nn.Embedding(len(vocab), 10)
context, target = trigrams[0]
context_idxs = [word_to_ix[w] for w in context]
context_var = autograd.Variable(torch.LongTensor(context_idxs))
embeddings = emd(context_var)
print(embeddings.size())
"""
The embeddings layer will return X * Y tensor, where
X: input vector length (here we have 2 inputs each time)
Y: embedding dimension
"""


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # nn.Embedding() layer takes in DTM, outputs continuous variables
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        # the output should be log_probs for each word in the dictionary
        # which is why we have vocab_size here

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        # embeds of shape 2*10 should be flatten to 1*20 since nn.Linear
        # only takes in one line
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

# Initiate the models
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = torch.Tensor([0])  # initiate total_loss for recording
    for context, target in trigrams:
        # generate input vector
        context_idxs = [word_to_ix[w] for w in context]
        # put into autograd.Variable for back prop
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        # calibrate model gradients
        model.zero_grad()
        # forward propagate model
        log_probs = model(context_var)
        # calculate loss
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
        # back propagate using ouput and error
        loss.backward()
        # update model
        optimizer.step()
        # record loss history
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


for i, contents in enumerate(trigrams):
    context, target = contents
    if i % 5 == 0:
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
data. We can use this methodology to train an embedding on a larger corpus
"""
