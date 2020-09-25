import torch
import tqdm
from .utils import *


def complexity_measure(G, C, p, noise_data, n_batch, batch_size, interpolation_method, n_step):
    max_jns = []
    l_starts = []
    l_ends = []
    lengths = []
    for mdli in tqdm(range(n_batch)):
        z_start_batch = noise_data.sample((batch_size,))
        z_end_batch = noise_data.sample((batch_size,))
        max_jn, length, l_start, l_end = max_jacobian_norm_batch(z_start=z_start_batch,
                                                                 z_end=z_end_batch,
                                                                 interpolation_method=interpolation_method,
                                                                 n_step=n_step,
                                                                 p=p, G=G, C=C)
        max_jns.append(max_jn)
        lengths.append(length)
        l_starts.append(l_start)
        l_ends.append(l_end)
    max_jns = torch.cat(max_jns, dim=0)
    lengths = torch.cat(lengths, dim=0)
    l_starts = torch.cat(l_starts, dim=0)
    l_ends = torch.cat(l_ends, dim=0)
    mean_max_jn = max_jns.mean().item()
    abs_max_jn = max_jns.max().item()
    mean_length = lengths.mean().item()
    class_jn, class_jn_mat = class_pair_distance(dists=max_jns, start_labels=l_starts, end_labels=l_ends,
                                                 nclasses=C.nclasses())
    class_dist, class_dist_mat = class_pair_distance(dists=lengths, start_labels=l_starts, end_labels=l_ends,
                                                     nclasses=C.nclasses())

    return abs_max_jn, mean_max_jn, class_jn, class_jn_mat, mean_length, class_dist, class_dist_mat


def max_jacobian_norm_batch(z_start, z_end, interpolation_method, n_step, p, G, C, use_label=False):
    """
        Compute the maximum of the norm of the Jacobian on the interpolization
        path from :math:`G(z(0))` to :math:`G(z(1))` with step size of :math:`1/n\_steps`
        and :math:`z(0) = z\_start`, :math:`z(1) = z\_end`.

        :param z_start: starting latent codes
        :param z_end: ending latent codes
        :param interpolation_method: lerp or slerp
        :param n_step: number of interpolation steps
        :param p: p-norm, default 2, the Frobenius norm
        :param G: generator
        :param C: classifier to classify end points
        :param use_label: use label in generator
        :return:
            max_jn: max Jacobian norm; label_start: label of G(z_start); label_end: label of G(z_end)
        """

    with torch.no_grad():
        batch_size = z_start.size(0)
        max_jn = torch.zeros(batch_size, device=z_start.device)
        label_start = max_jn.clone()
        label_end = max_jn.clone()
        length = max_jn.clone()

        z_list = interpolation_method(z_start, z_end, n_step)
        # step_size = 1. / n_steps
        x_start = G(z_list[0])
        x_p = x_start
        for i in range(1, n_step):
            x_c = G(z_list[i])
            diff = x_c - x_p
            # print(i, length.size(), diff.size(), z_list[i].size())
            diff = diff.view(batch_size, -1)  # flatten images
            length += diff.norm(p=p, dim=1)
            j = diff * n_step
            jn = j.norm(p, dim=1)
            max_jn[max_jn < jn] = jn[max_jn < jn]
            x_p = x_c

        if C is not None:
            label_start = C(x_start).argmax(dim=1)
            label_end = C(x_c).argmax(dim=1)

        # print(max_jn)
        # print(length)
        return max_jn, length, label_start, label_end


def class_pair_distance(dists, start_labels, end_labels, nclasses):
    """
    Compute average distance between the two classes in each pair. Return list of class pairs sorted by distance.
    The distance could be distance on the manifold or max singular value
    :param dists: precomputed sample pair distances
    :param start_labels:
    :param end_labels:
    :param nclasses:
    :return: class_len, len_mat
    """
    nlabelpair = nclasses * nclasses
    labelid = start_labels * nclasses + end_labels
    labelid.squeeze_()
    class_len = {}
    len_mat = torch.empty(nclasses, nclasses).fill_(INFINITY)
    for i in range(nlabelpair):
        class_len[i] = INFINITY
        di = dists[labelid == i]
        if di.nelement() > 0:
            class_len[i] = di.mean().item()
        r = i // nclasses
        c = i % nclasses
        len_mat[r, c] = class_len[i]
    class_len = {k: v for k, v in sorted(class_len.items(), key=lambda item: item[1])}
    return class_len, len_mat


# ----------------------------------------------------------------------------------------------------------------------
