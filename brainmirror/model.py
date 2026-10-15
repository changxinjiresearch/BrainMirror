"""Small fully inspectable 3D convolutional autoencoder.

Each graph node represents an exact (channel, z, y, x) tensor element.
Each graph edge corresponds to an actual Conv3d kernel coefficient.
This does not claim semantic alignment or causal influence by itself.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BrainAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(8, 8, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose3d(8, 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x, capture=False, ablate=None):
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 32, 32, 32):
            raise ValueError('Expected x in shape (batch,1,32,32,32)')
        acts = {'input': x} if capture else {}
        for name in ('conv1', 'conv2', 'conv3'):
            pre = getattr(self, name)(x)
            if ablate is not None and ablate.get('layer') == name:
                c, z, y, xx = (int(ablate[k]) for k in ('channel', 'z', 'y', 'x'))
                if not (0 <= c < pre.shape[1] and 0 <= z < pre.shape[2]
                        and 0 <= y < pre.shape[3] and 0 <= xx < pre.shape[4]):
                    raise ValueError('Ablation unit outside tensor dimensions')
                pre = pre.clone()
                pre[:, c, z, y, xx] = 0.0
            x = F.relu(pre)
            if capture:
                acts[name + '_pre'] = pre
                acts[name] = x
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        result = torch.sigmoid(self.deconv3(x))
        return (result, acts) if capture else result


LAYERS = {'conv1': {'channels': 4, 'side': 16},
          'conv2': {'channels': 8, 'side': 8},
          'conv3': {'channels': 8, 'side': 4}}


def incoming_edges(model, activations, layer, channel, z, y, x):
    """Enumerate **all** real kernel edges feeding a selected pre-ReLU unit.

    Returns exact weights, activations, multiplicative contributions.
    Tests verify sum(weight * input) + bias == measured pre-activation.
    """
    if layer not in LAYERS:
        raise ValueError('Unknown convolution layer')
    module = getattr(model, layer)
    source_name = {'conv1': 'input', 'conv2': 'conv1', 'conv3': 'conv2'}[layer]
    src = activations[source_name][0].detach().cpu()
    weights = module.weight.detach().cpu()
    bias = float(module.bias[channel].detach().cpu())
    out = []
    d, h, w = src.shape[1:]
    for ci in range(src.shape[0]):
        for kz in range(3):
            for ky in range(3):
                for kx in range(3):
                    sz = z * 2 - 1 + kz
                    sy = y * 2 - 1 + ky
                    sx = x * 2 - 1 + kx
                    # Zero-padding is not an actual source neuron; omit those edges.
                    if not (0 <= sz < d and 0 <= sy < h and 0 <= sx < w):
                        continue
                    activation = float(src[ci, sz, sy, sx])
                    weight = float(weights[channel, ci, kz, ky, kx])
                    out.append({'source': [ci, sz, sy, sx], 'activation': activation,
                                'weight': weight, 'contribution': activation * weight})
    pre = float(activations[layer + '_pre'][0, channel, z, y, x].detach().cpu())
    post = float(activations[layer][0, channel, z, y, x].detach().cpu())
    numerical_sum = float(sum(e['contribution'] for e in out) + bias)
    if abs(numerical_sum - pre) > 2e-5:
        raise AssertionError(f'Graph not faithful: computed {numerical_sum}, actual {pre}')
    return {'layer': layer, 'source_layer': source_name, 'unit': [channel, z, y, x],
            'bias': bias, 'preactivation': pre, 'activation': post,
            'verified_sum': numerical_sum, 'edges': out}
