import torch
import torch.utils.data as udata
import torchvision
from torchvision import datasets as dsets, transforms


# x = torch.arange(0, 10).view(10, 1)
# xdata = udata.TensorDataset(x)
# xloader = udata.DataLoader(xdata, batch_size=3, drop_last=True, shuffle=True)
#
# xiter = iter(xloader)
# for j in range(10):
#     try:
#         print(next(xiter))
#     except:
#         xiter = iter(xloader)
#         print(next(xiter))


size = 28
transformMnist = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

mnist = dsets.MNIST(root='~/github/data/mnist', train=True, transform=transformMnist, download=True)
mnist = udata.DataLoader(mnist, batch_size=64, drop_last=True, shuffle=True)


noise_weights = [0, 0.1, 0.5, 1., 2, 5]

for i, (x, y) in enumerate(mnist):
    for nw in noise_weights:
        noise = (torch.rand_like(x) * 2 - 1) * nw
        xnoise = x + noise
        torchvision.utils.save_image(noise, 'results/noise_%.1f.png' % nw, normalize=True, range=(-1, 1))
        torchvision.utils.save_image(xnoise, 'results/noisy_%.1f.png' % nw, normalize=True)  # , range=(-1.5, 1.5))
    # end for
    break
# end for

