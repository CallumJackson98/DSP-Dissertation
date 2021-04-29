import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
from string import punctuation
from update_variable_track import update
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



from model import NeuralNet

with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\new_dataset\\train_tweets.json', 'r', encoding="utf8") as f:
    intents = json.load(f)


all_words = []

#collect patterns and their tags
tags = []

#this will hold all patterns and tags
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


all_words = [stem(w) for w in all_words if w not in punctuation]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)  #1 hot

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    
    def __len__(self):
        return self.n_samples
    
    

# Hyperparameters
num_epochs = 30
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)



dataset = ChatDataset()
#try set num_workers to 0 and 1
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = NeuralNet(input_size, hidden_size, output_size).to(device)



# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

loss_track = []


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        #forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        #backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_track.append(round(loss.item(), 4))
    if (epoch + 1) % 1 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
    
    #print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')


print(f'final loss, loss={loss.item():.4f}')
        



data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

# add sizes of neural network???
run_info = [round(loss.item(), 4), learning_rate, num_epochs, batch_size, loss_track]

update(run_info)

print(f'training complete. file saved to {FILE}')







