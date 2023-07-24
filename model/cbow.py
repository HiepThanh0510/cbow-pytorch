import torch.nn as nn 


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear_1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        
        self.linear_2 = nn.Linear(128, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        

    def forward(self, inputs):
        embedding_context = self.embeddings(inputs)
        central_word = sum(embedding_context)
        
        out = central_word.view(1, -1) # shape: (1, 100)
        out = self.linear_1(out) # shape: (1, 128)
        out = self.relu(out) # shape: (1, 128)
        
        out = self.linear_2(out) # shape: (1, 87)
        out = self.logsoftmax(out) # shape: (1, 87)

        return out