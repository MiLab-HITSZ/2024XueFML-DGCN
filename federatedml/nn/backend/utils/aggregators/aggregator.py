import numpy as np


def aggregate_bn_data(bn_tensors, degrees=None):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
    client_nums = len(bn_tensors)
    layer_nums = len(bn_tensors[0]) // 2
    bn_data = []

    for i in range(layer_nums):
        mean_idx = i * 2
        mean_var_dim = len(bn_tensors[0][mean_idx])
        mean = np.zeros(mean_var_dim)

        for idx in range(client_nums):
            client_mean = bn_tensors[idx][mean_idx]
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            mean += client_mean * client_weight
        mean /= total_weight
        bn_data.append(mean)

        var_idx = mean_idx + 1
        var = np.zeros(mean_var_dim)
        for idx in range(client_nums):
            client_mean = bn_tensors[idx][mean_idx]
            client_var = bn_tensors[idx][var_idx]
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            var += (client_var + client_mean ** 2 - mean ** 2) * client_weight
        var /= total_weight
        bn_data.append(var)
    return bn_data


def aggregate_by_labels(tensors, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)

    for i in range(len(tensors)):
        for j, tensor in enumerate(tensors[i]):

            if j == len(tensors[i]) - 2 or j == len(tensors[i]) - 1:

                for k in range(len(tensor)):

                    if degrees_sum[k] == 0:
                        tensor[k] *= degrees[i][-1]
                        tensor[k] /= degrees_sum[-1]
                    else:
                        tensor[k] *= degrees[i][k]
                        tensor[k] /= degrees_sum[k]
                    if i != 0:
                        tensors[0][j][k] += tensor[k]
            else:
                tensor *= degrees[i][-1]
                tensor /= degrees_sum[-1]
                if i != 0:
                    tensors[0][j] += tensor

    return tensors[0]


def aggregate_whole_model(tensors, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
    for i in range(len(tensors)):
        client_weight = degrees[i] if degrees.ndim == 1 else degrees[i][-1]
        for j, tensor in enumerate(tensors[i]):
            tensor *= client_weight
            tensor /= total_weight
            if i != 0:
                tensors[0][j] += tensor
    return tensors[0]


def aggregate_relation_matrix(relation_matrices, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    client_nums = len(relation_matrices)
    relation_matrix = np.zeros_like(relation_matrices[0])
    for i in range(client_nums):
        relation_matrix += relation_matrices[i] * degrees[i][-1] / degrees_sum[-1]
    return relation_matrix
