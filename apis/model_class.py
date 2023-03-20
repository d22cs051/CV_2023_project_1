import torch.nn as net

class cnn(net.Module):
  def __init__(self):
    super().__init__()
    self.c1 = net.Conv2d(1,22,3)
    self.c2 = net.Conv2d(22,16,3)
    self.c3 = net.Conv2d(16,10,3)
    self.fc1 = net.Linear(10*22*22,512)
    self.o1 = net.Linear(512,10)
    self.relu = net.ReLU()

  def forward(self,ip):
    a = self.relu(self.c1(ip))
    a = self.relu(self.c2(a))
    a = self.relu(self.c3(a))
    ft = a.reshape(-1,10*22*22)
    a = self.relu(self.fc1(ft))
    a = self.o1(a)

    return ft,a


def make_model():
  obj = cnn()
  return obj