from utils import tokeniser, generate_training_samples, unique_word_vector
from model_pipeline import CBOW
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm 

path = r'D:\NLP\transformer\english.txt'
context_data, target_data,vocab_size= generate_training_samples(path)

learning_rate = 0.01
num_epochs = 100

# Create a CBOW model
model = CBOW(vocab_size, embedding_dim = 300, context_size = 2)

# Define loss and optimizer
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss for log probabilities
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for context_index, target_index in tqdm(zip(context_data, target_data), desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(context_data)):
        optimizer.zero_grad()
        target = torch.tensor([target_index])
        log_probs = model(context_index)
        loss = criterion(log_probs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(context_data)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

torch.save(model,r'D:\NLP\transformer\model.pth')



