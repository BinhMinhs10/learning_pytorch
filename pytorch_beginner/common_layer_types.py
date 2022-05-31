"""
torch.nn.Module encapsulate behaviors pytorch model and their components

"""
import torch


# ====================================
#  Simple two linear layers and an activation
# ====================================
class TinyModel(torch.nn.Module):
    def __init__(self):
        """
        Defines the layers and other components of a model
        """
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        """
        where computation
        :param x:
        :return:
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


tinymodel = TinyModel()
print(tinymodel)

# print('\nModel params:')
# for param in tinymodel.parameters():
#     print(param)

print('\nLayer tinymodel.linear2 params:')
for param in tinymodel.linear2.parameters():
    print(param)

# ====================================
#  Linear Layers (linear or fully connected)
# ====================================
print("=" * 15 + "Linear layers" + "=" * 15)
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
y = lin(x)
print(f"Input: {x}, output: {y}")
print(lin.weight)  # Check weights of layer

# ====================================
#  Convolutional Layers
# ====================================
print("=" * 15 + "Convolutional layers" + "=" * 15)
# Input image channel (black & white), 6 output channel, 5x5 square conv
conv1 = torch.nn.Conv2d(1, 6, 5)  # (Input channel, output features, kernel size)

# ====================================
#  Recurrent Layers
# ====================================
print("=" * 15 + "Recurrent layers" + "=" * 15)


class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores

# ====================================
#  Max pooling Layers
# ====================================
print("=" * 15 + "Max pooling layers" + "=" * 15)
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))

# ====================================
#  Normalization Layers
# ====================================
print("=" * 15 + "Normalization layers" + "=" * 15)
