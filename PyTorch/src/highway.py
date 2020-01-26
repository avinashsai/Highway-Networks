import torch
import torch.nn as nn

class Conv2DHighwayLayer(nn.Module):
  def __init__(self,inchannels,outchannels,kernelsize,activation='relu',stride=1,bias=-1):
    super(Conv2DHighwayLayer,self).__init__()
    self.inchannels = inchannels
    self.outchannels = outchannels
    self.kernelsize = kernelsize
    if(activation=='selu'):
      self.activation = nn.SELU()
    elif(activation=='elu'):
      self.activation = nn.ELU()
    else:
      self.activation = nn.ReLU()
    self.stride = stride
    self.padding = (self.kernelsize - 1) // 2
    self.conv = nn.Conv2d(self.inchannels,self.outchannels,self.kernelsize,stride=self.stride,
      padding=self.padding)
    self.gate = nn.Conv2d(self.inchannels,self.outchannels,self.kernelsize,stride=self.stride,
      padding=self.padding)
    self.gateact = nn.Sigmoid()
    self.gate.bias.data.fill_(bias)
  
  def forward(self,x):
    H = self.activation(self.conv(x))
    T = self.gateact(self.gate(x))
    out = H * T + x * (1 - T)
    return out


class Conv1DHighwayLayer(nn.Module):
  def __init__(self,inchannels,outchannels,kernelsize,activation='relu',stride=1,bias=-1):
    super(Conv1DHighwayLayer,self).__init__()
    self.inchannels = inchannels
    self.outchannels = outchannels
    self.kernelsize = kernelsize
    if(activation=='selu'):
      self.activation = nn.SELU()
    elif(activation=='elu'):
      self.activation = nn.ELU()
    else:
      self.activation = nn.ReLU()
    self.stride = stride
    self.padding = (self.kernelsize - 1) // 2
    self.conv = nn.Conv1d(self.inchannels,self.outchannels,self.kernelsize,stride=self.stride,
      padding=self.padding)
    self.gate = nn.Conv1d(self.inchannels,self.outchannels,self.kernelsize,stride=self.stride,
      padding=self.padding)
    self.gateact = nn.Sigmoid()
    self.gate.bias.data.fill_(bias)
  
  def forward(self,x):
    H = self.activation(self.conv(x))
    T = self.gateact(self.gate(x))
    out = H * T + x * (1 - T)
    return out


class HighwayFC(nn.Module):
  def __init__(self,indim,outdim,activation='relu',bias=-1):
    super(HighwayFC,self).__init__()
    self.indim = indim
    self.outdim = outdim
    if(activation=='selu'):
      self.activation = nn.SELU()
    elif(activation=='elu'):
      self.activation = nn.ELU()
    else:
      self.activation = nn.ReLU()
    self.fc = nn.Linear(self.indim,self.outdim)
    self.gate = nn.Linear(self.indim,self.outdim)
    self.gateact = nn.Sigmoid()
    self.gate.bias.data.fill_(bias)

  def forward(self,x):
    H = self.activation(self.fc(x))
    T = self.gateact(self.gate(x))
    out = H * T + x * (1 - T)
    return out