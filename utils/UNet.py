import torch
from torch import nn
import torch.nn.functional as F
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

class ImageConvBlock(nn.Module):
    """
    A convolutional block for image data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        depth (int, optional): The number of successive convolutional layers (default: 2)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:2),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        dropout_rate (float, optional): The spatial dropout rate to be applied to the final convolution output (default:None)

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self, 
                in_channels, 
                filters=32, 
                kernel_size=3,
                depth=1, 
                rank=3,
                activation='relu', 
                norm_type=None,
                dropout_rate=None):
        super().__init__() 

        conv = get_image_layer('Conv', rank)
        drop = get_image_layer('Dropout', rank)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=kernel_size//2)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_image_layer(norm_type, rank)(filters) if norm_type else nn.Identity()
            for _ in range(depth)
        ])
        self.drop = drop(p=dropout_rate) if dropout_rate else nn.Identity()

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature maps in image space [in_channels, ...] where the number of dims in ... corresponds to rank

        Returns:
            torch.Tensor: Output feature maps [out_channels, ...]
        """

        for conv, norm in zip(self.convs, self.norms):
            x = norm(self.act(conv(x)))
        return self.drop(x)
    

class ImageEncoder(nn.Module):
    """
    A CNN encoder for images. Structured like the encoder of a UNet.

    Args:
        in_channels (int): The number of channels in the input image.
        filters (List[int], optional): The number of convolutional filters in each encoder level (default: [16,32,64,128,256])
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        conv_blocks_per_level (int, optional): The number of successive convolutional blocks per encoder level (default: 1)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:3),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        pool_size (int or tuple, optional): Pooling kernel size. If int, expands to tuple of given length for rank dimensions (default: 2). If tuple, must have length equal to rank (e.g., (1,2,2) for 3D to avoid pooling in depth dimension).
        dropout_rate (float, optional): The spatial dropout rate to be applied to each residual block prior to residual connection (default:None)

    Returns:
        A `torch.nn.Module` object.
    """

    def __init__(self,
                in_channels, 
                filters=[16,32,64,128,256],
                kernel_size=3,
                conv_blocks_per_level=1,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
                pool_size=2,
                activation='relu',
                dropout_rate=None):
        super().__init__()
        
        # Handle pool_size as int or tuple
        if isinstance(pool_size, int):
            pool_size = (pool_size,) * rank
        
        n_levels = len(filters)
        self.conv_blocks = nn.ModuleList([
                ImageConvBlock(in_channels=in_channels if i==0 else filters[i-1], 
                        filters=filters[i], 
                        kernel_size=kernel_size, 
                        depth=conv_blocks_per_level,
                        rank=rank,
                        activation=activation,
                        norm_type=norm_type,
                        dropout_rate=dropout_rate)
            for i in range(n_levels)
        ])
        pool = get_image_layer(pool_type, rank)
        self.maxpools = nn.ModuleList([
            pool(pool_size) if i>0 else nn.Identity()
            for i in range(n_levels)
        ])

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): Input image [in_channels, ...]

        Returns:
            List[torch.Tensor]: Output feature maps from each level ordered from top to bottom [Tensor([filters[0], ...], ..., Tensor([filters[N], ...])
        """
        outputs = []
        for pool, conv in zip(self.maxpools, self.conv_blocks):
            x = conv(pool(x))
            outputs.append(x)
        return outputs



class ImageDecoder(nn.Module):
    """
    CNN decoder for images. Mirrors ImageEncoder like a UNet decoder.

    Args:
        filters (List[int]): Encoder filter sizes in top→bottom order.
        kernel_size (int): Convolution kernel size.
        conv_blocks_per_level (int): Number of conv blocks per level.
        rank (int): Spatial rank (2 or 3).
        upsample_type (str): "ConvTranspose" or "Upsample".
        upsample_size (int or tuple, optional): Upsampling kernel/scale size. If int, expands to tuple of given length for rank dimensions (default: 2). If tuple, must have length equal to rank (e.g., (1,2,2) to avoid upsampling in depth dimension).
        activation (str): Activation name.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout rate.
        skip (bool): Use skip connections.
    """

    def __init__(self,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 upsample_type="ConvTranspose",
                 upsample_size=2,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 skip=True):
        super().__init__()

        self.skip = skip
        n_levels = len(filters)

        # Handle upsample_size as int or tuple
        if isinstance(upsample_size, int):
            upsample_size = (upsample_size,) * rank

        rev_filters = filters[::-1]

        if upsample_type.lower() == 'upsample':
            self.ups = nn.ModuleList([
                nn.Upsample(scale_factor=upsample_size, mode='trilinear' if rank==3 else 'bilinear', align_corners=True)
                for _ in range(n_levels - 1)
            ])
        else:
            up_layer = get_image_layer(upsample_type, rank)
            self.ups = nn.ModuleList([
                up_layer(rev_filters[i], rev_filters[i+1], kernel_size=upsample_size, stride=upsample_size)
                for i in range(n_levels - 1)
            ])


        self.conv_blocks = nn.ModuleList([
            ImageConvBlock(
                in_channels=rev_filters[i+1] * (2 if skip else 1),
                filters=rev_filters[i+1],
                kernel_size=kernel_size,
                depth=conv_blocks_per_level,
                rank=rank,
                activation=activation,
                norm_type=norm_type,
                dropout_rate=dropout_rate
            )
            for i in range(n_levels - 1)
        ])


    def _match_size(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            if all(s >= t for s, t in zip(skip.shape[2:], x.shape[2:])):
                # Center-crop skip to x when skip is larger or equal in every spatial dim
                diff = [s - t for s, t in zip(skip.shape[2:], x.shape[2:])]
                slices = [slice(d // 2, d // 2 + t) for d, t in zip(diff, x.shape[2:])]
                skip = skip[(..., *slices)]
            else:
                # Fallback: resize skip to x shape if any dim is smaller
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear' if x.ndim == 5 else 'bilinear', align_corners=True)
        return skip


    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: List of tensors from encoder (top→bottom).

        Returns:
            Decoded tensor at highest resolution.
        """

        # Reverse the encoder outputs so we traverse from bottleneck to top
        rev_enc = encoder_outputs[::-1]

        # Start from bottleneck
        x = rev_enc[0]

        # Traverse decoder levels
        for i, (up, conv) in enumerate(zip(self.ups, self.conv_blocks)):

            x = up(x)

            if self.skip:
                skip_feat = rev_enc[i + 1]  # next encoder feature
                skip_feat = self._match_size(x, skip_feat)
                x = torch.cat([x, skip_feat], dim=1)

            x = conv(x)

        return x

    
class UNet(nn.Module):
    """
    UNet for 2D or 3D images.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        filters (List[int]): Encoder filter sizes.
        kernel_size (int): Conv kernel size.
        conv_blocks_per_level (int): Depth per level.
        rank (int): Spatial rank.
        activation (str): Activation function.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout.
        pool_size (int or tuple, optional): Pooling kernel size (default: 2). If tuple, must match rank.
        upsample_size (int or tuple, optional): Upsampling kernel size (default: 2). If tuple, must match rank.
        final_activation (str): Output activation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 pool_size=2,
                 upsample_size=2,
                 final_activation="linear"):

        super().__init__()

        self.encoder = ImageEncoder(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            pool_size=pool_size
        )

        self.decoder = ImageDecoder(
            filters=filters,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            upsample_size=upsample_size
        )

        conv = get_image_layer("Conv", rank)

        self.final_conv = conv(filters[0], out_channels, kernel_size=1)

        self.final_act = (
            get_activation(final_activation)()
            if final_activation.lower() != "linear"
            else nn.Identity()
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = self.decoder(enc_feats)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x


