from utils import tokeniser, generate_training_samples, unique_word_vector
import torch
from model_pipeline import CBOW
import torch.nn.functional as F

#Loading the Model
path = r'D:\NLP\transformer\english.txt'

word_to_id, id_to_word = unique_word_vector(path)

test = 'mandu fetched bananas from'
def convert(test):
    lst = []
    for i in test.split(' '):
        lst.append(word_to_id[i])
    return torch.tensor(lst)

model = torch.load(r'D:\NLP\transformer\model.pth')
model.eval()
output = model(convert(test))


# Assuming "output" is your log-softmax tensor
probabilities = torch.exp(output)

predicted_class = torch.argmax(probabilities, dim=1)
print(id_to_word[predicted_class.item()])