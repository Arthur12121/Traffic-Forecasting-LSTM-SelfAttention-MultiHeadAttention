import torch
from torch.utils.data import Dataset , DataLoader
import math

class DataPreperFromTrafficToTensore(Dataset):

    def __init__(self):
        super().__init__()
#ITS FAKE DATA FOR EXAMPLE YOU CAN USE REAL DATA
        traffic =[
    120,125,130,128,135,
    140,145,150,148,155,
    200,220,210,190,180,
    160,165,170,168,175,
    180,185,190,188,195,
    250,270,260,240,230,
    220,225,230,228,235,
    240,245,250,248,255,
    300,320,310,290,280,
    260,265,270,268,275,
    280,285,290,288,295,
    350,370,360,340,330,
    310,315,320,318,325,
    330,335,340,338,345,
    400,420,410,390,380,
    360,365,370,368,375,
    380,385,390,388,395,
    450,470,460,440,430,
    410,415,420,418,425
                      ]
        
        self.x = []
        self.y = []

        windows = 5

        for i in range(len(traffic)-windows):
            
            seq = traffic[i:i+windows]
            target = traffic[i+windows]

            self.x.append(seq)
            self.y.append(target)

        self.x = torch.tensor(self.x , dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(self.y , dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index] , self.y[index]
    

class SelfAttention(torch.nn.Module):

    def __init__(self , hidden_stat):
        super().__init__()

        self.Query = torch.nn.Linear(hidden_stat , hidden_stat)

        self.Key = torch.nn.Linear(hidden_stat , hidden_stat)

        self.Value = torch.nn.Linear(hidden_stat , hidden_stat)
    
    def forward(self , x):

        Query = self.Query(x)

        Key = self.Key(x)

        Value = self.Value(x)

        score = Query @ Key.transpose(-2 , -1)

        score = score/math.sqrt(Key.size(-1))

        weights = torch.softmax(score , dim=-1)

        out = weights @ Value

        return out

class ModelNetworkNeroun(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=1 , hidden_size=64 , batch_first=True)
        
        self.attention = SelfAttention(hidden_stat=64)

        self.multihead = torch.nn.MultiheadAttention(embed_dim=64 , num_heads=4 , batch_first=True)

        self.linearfirst = torch.nn.Linear(64 , 32)

        self.relu = torch.nn.ReLU()

        self.linearsecound = torch.nn.Linear(32 , 1)

    def forward(self , x):

        output , (hidden , cell) = self.lstm(x)

        output = self.attention(output)

        output , _ = self.multihead(output , output , output)

        output = output[:,-1,:]


        output = self.linearfirst(output)

        output = self.relu(output)

        output = self.linearsecound(output)

        return output
    
model = ModelNetworkNeroun()

certation = torch.nn.MSELoss()

data_set = DataPreperFromTrafficToTensore()

optimazer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

data = DataLoader(
    dataset=data_set,
    shuffle=True
)

for i in range(20):
    for x_batch , y_batch in data:

        predaction = model(x_batch)

        loss = certation(predaction , y_batch)

        optimazer.zero_grad()

        loss.backward()

        optimazer.step()

        print(i , loss.item())

test = torch.tensor([
    [
        [175.0],
        [180.0],
        [185.0],
        [190.0],
        [188.0]
    ]
])

with torch.no_grad():

    prediction = model(test)

print(prediction)
