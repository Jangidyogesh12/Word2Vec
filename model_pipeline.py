import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        # Flatten the embedded tensor properly
        embedded = embedded.view(1, -1)
        out = torch.relu(self.linear1(embedded))
        out = self.linear2(out)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs
