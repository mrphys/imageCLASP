import torch
from torch import nn
import torch.nn.functional as F
from .layer_util import get_image_layer, get_activation



class ImageConvBlock(nn.Module):
    """
    A convolutional block for image data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        kernel_size (int or tuple, optional): The kernel(filter) size for the convolutional layers (default: 3).
            If a tuple is given (e.g. (1,3,3) for anisotropic data) padding is computed per-axis as k//2.
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

        # Support tuple kernel_size (e.g. (1,3,3) for anisotropic stages)
        if isinstance(kernel_size, (tuple, list)):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2

        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=padding)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_image_layer(norm_type, rank)(filters, affine=True) if norm_type else nn.Identity()
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
        kernel_size (int, optional): Uniform kernel size used when kernel_sizes is not provided (default: 3)
        kernel_sizes (List[int or tuple], optional): Per-stage kernel sizes. When provided overrides kernel_size.
            Supports tuples for anisotropic stages, e.g. [(1,3,3), (1,3,3), (3,3,3), ...].
        conv_blocks_per_level (int, optional): The number of successive convolutional blocks per encoder level (default: 1)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:3),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        pool_size (int or tuple, optional): Uniform pooling kernel size used when strides is not provided (default: 2).
        strides (List[tuple], optional): Per-stage pooling strides. When provided overrides pool_size.
            First element should be (1,1,1) or (1,1) to indicate no pooling at stage 0.
            e.g. [(1,1,1), (1,2,2), (1,2,2), (2,2,2), (1,2,2), (1,2,2), (1,2,2)]
        dropout_rate (float, optional): The spatial dropout rate to be applied to each residual block prior to residual connection (default:None)

    Returns:
        A `torch.nn.Module` object.
    """

    def __init__(self,
                in_channels,
                filters=[16,32,64,128,256],
                kernel_size=3,
                kernel_sizes=None,
                conv_blocks_per_level_encoder=1,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
                pool_size=2,
                strides=None,
                activation='relu',
                dropout_rate=None):
        super().__init__()

        n_levels = len(filters)

        # Resolve per-stage kernel sizes
        if kernel_sizes is not None:
            stage_kernels = kernel_sizes
        else:
            stage_kernels = [kernel_size] * n_levels

        self.conv_blocks = nn.ModuleList([
                ImageConvBlock(in_channels=in_channels if i==0 else filters[i-1],
                        filters=filters[i],
                        kernel_size=stage_kernels[i],
                        depth=conv_blocks_per_level_encoder,
                        rank=rank,
                        activation=activation,
                        norm_type=norm_type,
                        dropout_rate=dropout_rate)
            for i in range(n_levels)
        ])

        pool = get_image_layer(pool_type, rank)

        if strides is not None:
            # Per-stage pooling: use stride as both kernel_size and stride for non-overlapping pool
            self.maxpools = nn.ModuleList([
                pool(kernel_size=strides[i], stride=strides[i])
                if any(s > 1 for s in strides[i])
                else nn.Identity()
                for i in range(n_levels)
            ])
        else:
            # Uniform pool_size (original behaviour)
            if isinstance(pool_size, int):
                pool_size = (pool_size,) * rank
            self.maxpools = nn.ModuleList([
                pool(pool_size) if i > 0 else nn.Identity()
                for i in range(n_levels)
            ])

    def forward(self, x, verbose=False):
        """
        Args:
            x (torch.Tensor): Input image [in_channels, ...]

        Returns:
            List[torch.Tensor]: Output feature maps from each level ordered from top to bottom [Tensor([filters[0], ...], ..., Tensor([filters[N], ...])
        """
        outputs = []
        for i, (pool, conv) in enumerate(zip(self.maxpools, self.conv_blocks)):
            x = conv(pool(x))
            if verbose:
                print(f"  Encoder stage {i}: {tuple(x.shape)}")
            outputs.append(x)
        return outputs



class ImageDecoder(nn.Module):
    """
    CNN decoder for images. Mirrors ImageEncoder like a UNet decoder.

    Args:
        filters (List[int]): Encoder filter sizes in top→bottom order.
        kernel_size (int): Uniform convolution kernel size for decoder blocks (default: 3).
        conv_blocks_per_level (int): Number of conv blocks per level.
        rank (int): Spatial rank (2 or 3).
        upsample_type (str): "ConvTranspose" or "Upsample".
        upsample_size (int or tuple, optional): Uniform upsampling kernel/scale size used when strides is not provided.
        strides (List[tuple], optional): Per-stage strides from the encoder (full list including stage-0 identity).
            Decoder upsample strides are derived as strides[1:][::-1].
            e.g. encoder strides [(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2),(1,2,2)]
                 → decoder strides [(1,2,2),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2)]
        activation (str): Activation name.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout rate.
        skip (bool): Use skip connections.
    """

    def __init__(self,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 conv_blocks_per_level_decoder=1,
                 rank=3,
                 upsample_type="ConvTranspose",
                 upsample_size=2,
                 strides=None,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 skip=True):
        super().__init__()

        self.skip = skip
        n_levels = len(filters)

        rev_filters = filters[::-1]

        # Resolve per-level decoder upsample strides
        if strides is not None:
            # strides[0] is identity (no pool); strides[1:] are the actual pool strides.
            # Decoder traverses in reverse, so flip strides[1:].
            decoder_strides = list(reversed(strides[1:]))
        else:
            # Uniform upsample_size (original behaviour)
            if isinstance(upsample_size, int):
                upsample_size = (upsample_size,) * rank
            decoder_strides = [upsample_size] * (n_levels - 1)

        if upsample_type.lower() == 'upsample':
            # Upsample mode: use scale factors derived from decoder_strides
            self.ups = nn.ModuleList([
                nn.Upsample(scale_factor=decoder_strides[i],
                            mode='trilinear' if rank == 3 else 'bilinear',
                            align_corners=True)
                for i in range(n_levels - 1)
            ])
        else:
            up_layer = get_image_layer(upsample_type, rank)
            self.ups = nn.ModuleList([
                up_layer(rev_filters[i], rev_filters[i+1],
                         kernel_size=decoder_strides[i], stride=decoder_strides[i])
                for i in range(n_levels - 1)
            ])

        self.conv_blocks = nn.ModuleList([
            ImageConvBlock(
                in_channels=rev_filters[i+1] * (2 if skip else 1),
                filters=rev_filters[i+1],
                kernel_size=kernel_size,
                depth=conv_blocks_per_level_decoder,
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


    def forward(self, encoder_outputs, verbose=False):
        """
        Args:
            encoder_outputs: List of tensors from encoder (top→bottom).

        Returns:
            List[torch.Tensor]: Decoder outputs at each level, ordered low-res → high-res.
                The last element is the highest-resolution output.
        """

        # Reverse the encoder outputs so we traverse from bottleneck to top
        rev_enc = encoder_outputs[::-1]

        # Start from bottleneck
        x = rev_enc[0]

        outputs = []

        # Traverse decoder levels
        for i, (up, conv) in enumerate(zip(self.ups, self.conv_blocks)):

            x = up(x)

            if self.skip:
                skip_feat = rev_enc[i + 1]  # next encoder feature
                skip_feat = self._match_size(x, skip_feat)
                x = torch.cat([x, skip_feat], dim=1)

            x = conv(x)
            if verbose:
                print(f"  Decoder stage {i}: {tuple(x.shape)}")
            outputs.append(x)

        return outputs  # [lowres_output, ..., highres_output]


class UNet(nn.Module):
    """
    UNet for 2D or 3D images.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        filters (List[int]): Encoder filter sizes.
        kernel_size (int): Uniform conv kernel size used when kernel_sizes is not provided.
        kernel_sizes (List[int or tuple], optional): Per-stage encoder kernel sizes. Overrides kernel_size.
        conv_blocks_per_level_encoder (int): Depth per encoder level.
        conv_blocks_per_level_decoder (int): Depth per decoder level.
        rank (int): Spatial rank.
        activation (str): Activation function.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout.
        pool_size (int or tuple, optional): Uniform pooling size used when strides is not provided.
        strides (List[tuple], optional): Per-stage pooling/upsampling strides. Overrides pool_size/upsample_size.
            First element should be the identity stride (1,...,1) for stage 0.
        upsample_size (int or tuple, optional): Uniform upsampling size used when strides is not provided.
        final_activation (str): Output activation.
        deep_supervision (bool): If True, return a list of predictions at each decoder resolution
            (highest-res first) rather than a single output tensor. Default: False.
        num_ds_outputs (int, optional): Number of decoder levels to supervise when
            deep_supervision=True. Counted from the highest resolution downward.
            nnUNet convention: skip the 2 lowest-resolution levels, so
            num_ds_outputs = len(filters) - 3. Default: None (supervise all levels).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 kernel_sizes=None,
                 conv_blocks_per_level_encoder=1,
                 conv_blocks_per_level_decoder=1,
                 rank=3,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 pool_size=2,
                 strides=None,
                 upsample_size=2,
                 final_activation="linear",
                 deep_supervision=False,
                 num_ds_outputs=None):

        super().__init__()

        self.deep_supervision = deep_supervision

        self.encoder = ImageEncoder(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            kernel_sizes=kernel_sizes,
            conv_blocks_per_level_encoder=conv_blocks_per_level_encoder,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            pool_size=pool_size,
            strides=strides,
        )

        self.decoder = ImageDecoder(
            filters=filters,
            kernel_size=kernel_size,
            conv_blocks_per_level_decoder=conv_blocks_per_level_decoder,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            upsample_size=upsample_size,
            strides=strides,
        )

        conv = get_image_layer("Conv", rank)

        def _make_final_act():
            return (
                get_activation(final_activation)()
                if final_activation.lower() != "linear"
                else nn.Identity()
            )

        n_decoder_levels = len(filters) - 1
        # Limit supervised outputs: nnUNet skips the 2 lowest-res levels
        if deep_supervision:
            n_ds = n_decoder_levels if num_ds_outputs is None else min(num_ds_outputs, n_decoder_levels)
        self.n_ds = n_ds if deep_supervision else 0

        if deep_supervision:
            # One 1x1 conv + activation per supervised decoder level (highest-res first).
            # channels at level i (high-res first) = filters[i]
            self.ds_convs = nn.ModuleList([
                conv(filters[i], out_channels, kernel_size=1)
                for i in range(n_ds)
            ])
            self.ds_acts = nn.ModuleList([_make_final_act() for _ in range(n_ds)])
        else:
            # Standard single output
            self.final_conv = conv(filters[0], out_channels, kernel_size=1)
            self.final_act = _make_final_act()

    def forward(self, x, verbose=False):
        if verbose:
            print(f"[UNet] Input:     {tuple(x.shape)}")
        enc_feats = self.encoder(x, verbose=verbose)
        decoder_outputs = self.decoder(enc_feats, verbose=verbose)  # list: [lowres, ..., highres]

        if self.deep_supervision:
            # Reverse so index 0 = highest resolution, then take only supervised levels
            ds_list = list(reversed(decoder_outputs))[:self.n_ds]
            results = [act(conv(feat))
                       for act, conv, feat in zip(self.ds_acts, self.ds_convs, ds_list)]
            if verbose:
                for i, r in enumerate(results):
                    print(f"  DS output  {i}: {tuple(r.shape)}")
            return results
        else:
            x = decoder_outputs[-1]  # highest-resolution decoder output
            result = self.final_act(self.final_conv(x))
            if verbose:
                print(f"[UNet] Output:    {tuple(result.shape)}")
            return result
