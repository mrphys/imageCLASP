from torch import nn
import torch.nn.functional as F


def get_image_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _IMAGE_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err

  
def get_activation(name):
  """Get an activation object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested activation is unknown.
  """
  try:
    return _ACTIVATIONS[name]
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err


_IMAGE_LAYERS = {
    ('AveragePooling', 1): nn.AvgPool1d,
    ('AveragePooling', 2): nn.AvgPool2d,
    ('AveragePooling', 3): nn.AvgPool3d,
    ('Conv', 1): nn.Conv1d,
    ('Conv', 2): nn.Conv2d,
    ('Conv', 3): nn.Conv3d,
    ('ConvTranspose', 1): nn.ConvTranspose1d,
    ('ConvTranspose', 2): nn.ConvTranspose2d,
    ('ConvTranspose', 3): nn.ConvTranspose3d,
    ('MaxPool', 1): nn.MaxPool1d,
    ('MaxPool', 2): nn.MaxPool2d,
    ('MaxPool', 3): nn.MaxPool3d,
    ('Dropout', 1): nn.Dropout1d,
    ('Dropout', 2): nn.Dropout2d,
    ('Dropout', 3): nn.Dropout3d,
    ('ZeroPadding', 1): nn.ZeroPad1d,
    ('ZeroPadding', 2): nn.ZeroPad2d,
    ('ZeroPadding', 3): nn.ZeroPad3d,
    ('BatchNorm', 1): nn.BatchNorm1d,
    ('BatchNorm', 2): nn.BatchNorm2d,
    ('BatchNorm', 3): nn.BatchNorm3d,
    ('InstanceNorm', 1): nn.InstanceNorm1d,
    ('InstanceNorm', 2): nn.InstanceNorm2d,
    ('InstanceNorm', 3): nn.InstanceNorm3d
}

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "softmax": nn.Softmax
}