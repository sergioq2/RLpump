from flask import Flask, request, render_template
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


app = Flask("__name__")

q = ""

class QNetwork(nn.Module):
    def __init__(self, state_size,action_size, seed, fc1_unit=64, fc2_unit=64):
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

@app.route("/")
def loadPage():
	return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def pump():

    inputQuery1 = float(request.form['query1'])
    inputQuery2 = float(request.form['query2'])
    inputQuery3 = float(request.form['query3'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file = 'model.pth'

    loaded_model = QNetwork(state_size=3,action_size=4, seed=0, fc1_unit=64, fc2_unit=64)
    loaded_model.load_state_dict(torch.load(file))
    loaded_model.to(device)
    loaded_model.eval()
    data = [inputQuery1, inputQuery2, inputQuery3]

    s=torch.from_numpy(np.array(data)).float().to(device)
    loaded_model.eval()
    with(torch.no_grad()):
        out=loaded_model(s)
        acc=np.argmax(out.cpu().numpy())
    acc

    output1 = "Número de Bombas a tener encendidas: {}".format(acc)

    return render_template('home.html', output1=output1, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'])
    
app.run()