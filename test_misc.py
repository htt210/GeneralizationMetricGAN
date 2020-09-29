import torch
import torch.utils.data as udata
import torchvision.datasets as dsets


x = torch.arange(0, 10).view(10, 1)
xdata = udata.TensorDataset(x)
xloader = udata.DataLoader(xdata, batch_size=3, drop_last=True, shuffle=True)

xiter = iter(xloader)
for j in range(10):
    try:
        print(next(xiter))
    except:
        xiter = iter(xloader)
        print(next(xiter))
