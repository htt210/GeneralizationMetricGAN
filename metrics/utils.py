import torch
import torch.nn.functional as F
import torch.autograd as ag


INFINITY = 1e10


# ----------------------------------------------------------------------------------------------------------------------
def slerp(start, end, n_steps):
    """
    Return spherical interpolation from start to end
    :param start:
    :param end:
    :param n_steps:
    :return:
    """
    angle = ((start * end).sum(dim=1, keepdim=True) / start.norm(2, dim=1, keepdim=True)
             / end.norm(2, dim=1, keepdim=True)).acos()
    sinangle = torch.sin(angle)
    step = angle / n_steps
    inter_list = []
    for i in range(n_steps + 1):
        inter = torch.sin(angle - i * step) / sinangle * start + torch.sin(i * step) / sinangle * end
        inter_list.append(inter)
    return inter_list


def lerp(start, end, n_steps):
    """
    Return linear interpolation from `start` to `end`.
    :param start: B x C x W x H
    :param end:  B x C x W x H
    :param n_steps: number of steps
    :return: list of (n_steps+1) tensors of size B x C x W x H
    """
    step = (end - start) / n_steps
    inter_list = []
    for i in range(n_steps + 1):
        inter_list.append(start + i * step)

    return inter_list


def p_distance(x, y, p):
    batch_size = x.size(0)
    return (x - y).view(batch_size, -1).norm(p=p, dim=1)


def data_path_length(z_start, z_end, interpolation_method, n_steps, p,
                     G, D, C):
    z_list = interpolation_method(z_start, z_end, n_steps)
    dists = torch.zeros(z_start.size(0)).to(z_start.device)
    with torch.no_grad():
        xp = G(z_start)
        for i in range(1, n_steps + 1):
            xi = G(z_list[i])
            dists += p_distance(xp, xi, p=p)
            xp = xi

        # required for inter class distance
        start_labels = None
        end_labels = None
        if C is not None:
            start_labels = C(G(z_start)).argmax(dim=1, keepdim=True)
            end_labels = C(G(z_end)).argmax(dim=1, keepdim=True)

    # print(dists.size(), start_labels.size())
    return dists, start_labels, end_labels


# def extract_features(images, C, layer, batch_size, device):
#     C = C.to(device)
#     with torch.no_grad():
#         for bidx, batch in enumerate(images):
#             features = C.get_features(batch, layer_name=layer)
#

# ----------------------------------------------------------------------------------------------------------------------

