# =============================================================================
# nnUNetTrainerTransUNet.py
#
# Note: This file is provided for transparency during peer review.
# Full documentation, reproducibility instructions, and pretrained weights
# will be finalized and released upon manuscript acceptance.
#
# This trainer implements OF-TransUNet, corresponding to the architecture
# described in:
#   "Efficient Transformer Integration in nnU-Net for Liver Tumor Segmentation:
#    An External Validation Study"
#
# Key implementation details documented here:
#   - Single Conv-Transformer block inserted at encoder stage 3 (x8 downsampling)
#   - Output-focused progressive unfreezing schedule:
#       Epochs   0-89 : Transformer fully frozen
#       Epoch   90    : Group 1 ["final"] released (output-side mapping)
#       Epoch  110    : Group 2 ["final","conv2","conv3"] released (FFN)
#       Epoch  130    : Group 3 ["qkv","proj"] released (attention projections)
#       Epoch  150+   : Group 4 ["all"] - full unfreezing
#   - SGD optimizer (momentum=0.99, Nesterov=True, initial LR=0.01)
#   - Transformer LR after unfreezing: initial_lr * 0.1 = 1e-3
#   - NaN/Inf loss and gradient detection with automatic LR reduction
#
# For questions during peer review, please contact the corresponding author
# listed in the manuscript.
# =============================================================================

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import os
import inspect
import numpy as np

# TransUNetEncoder is only required when use_custom_transformer=False
# (i.e., when feature map spatial dimensions are sufficiently large for patch-based ViT).
# In the reported experiments, feature map dimensions triggered automatic fallback to the
# custom Conv-Transformer path; TransUNetEncoder was not used in the reported results.
try:
    from nnunetv2.training.network.trans_unet_encoder import TransUNetEncoder
except ImportError:
    TransUNetEncoder = None

from torch.cuda.amp import autocast, GradScaler


class nnUNetTrainerTransUNet(nnUNetTrainer):
    """
    nnU-Net trainer with a single lightweight Conv-Transformer block inserted at encoder stage 3,
    combined with an output-focused progressive unfreezing schedule.

    Corresponds to the OF-TransUNet architecture described in:
      'Efficient Transformer Integration in nnU-Net for Liver Tumor Segmentation:
       An External Validation Study'
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Transformer configuration ---
        self.transformer_config = {
            'vit_name': 'R50-ViT-B_16',
            'vit_patches_size': 8,        # Reduced patch size to accommodate small feature maps
            'use_pretrained': False,
            'enable_3d': True,
            'insert_stage': 3             # Insert at encoder stage 3 (x8 downsampling)
        }

        # --- Training hyperparameters ---
        self.max_num_epochs = 150
        self.initial_lr = 1e-2            # Initial learning rate: 0.01 (SGD)
        self.print_to_log_file(f"Initial learning rate: {self.initial_lr}")

        # Save configuration name for architecture lookup
        self.configuration_name = configuration

        # Logging control: track which messages have been printed once
        self.logged_once = set()

        # Flag: use custom convolutional Transformer (default True for small feature maps)
        self.use_custom_transformer = True

        # --- Unfreezing schedule parameters ---
        # Corresponds to the output-focused progressive unfreezing described in the manuscript.
        # The Transformer is frozen during epochs 0-89 (base_training_epochs = 90).
        # At epoch 90, output-side components are released first (layer_groups[1] = ["final"]).
        self.freeze_transformer = True    # Transformer is frozen at initialization
        self.base_training_epochs = 90   # Freeze duration; unfreezing begins at epoch 90
        self.unfreeze_blocks = 1          # Number of groups to unfreeze per interval
        self.unfreeze_interval = 20       # Epochs between successive unfreezing steps

        # Layer group definitions for output-focused unfreezing.
        # Group 1 (output-side) is released first, progressing inward toward attention layers.
        # This directly corresponds to Table/Figure descriptions in the manuscript.
        self.unfreeze_layers = [1, 2, 3, 4]
        self.custom_unfreeze_order = False  # Set True to apply a one-shot custom release
        self.layer_groups = {
            1: ["final"],                       # Output-side mapping component (released first)
            2: ["final", "conv2", "conv3"],     # FFN components
            3: ["qkv", "proj"],                 # Attention projection components
            4: ["all"]                          # Full unfreezing
        }

        # --- Gradient clipping ---
        self.grad_clip_value = 12.0       # Gradient clipping norm threshold
        self.enable_grad_clipping = True

        # Learning rate factor applied to Transformer parameters after unfreezing
        self.unfreeze_lr_factor = 0.1     # Transformer LR = initial_lr * 0.1 = 1e-3

        # --- NaN/Inf handling ---
        # If loss or gradients contain NaN/Inf for max_nan_counter consecutive steps,
        # the learning rate is halved (lr_reduce_factor = 0.5).
        self.nan_counter = 0
        self.max_nan_counter = 3
        self.lr_reduce_factor = 0.5
        self.min_lr = 1e-6

        # Initialize gradient scaler for mixed-precision training
        if self.device.type == 'cuda':
            self.grad_scaler = GradScaler()

    def initialize(self):
        """
        Initialize the trainer: build the base nnU-Net, then insert the Conv-Transformer block.
        """
        super().initialize()
        self.print_to_log_file("\n======= TransUNet Initialization =======")

        # Retrieve architecture configuration from plans
        try:
            config = self.configuration_name
            self.print_to_log_file(f"Configuration name: {config}")

            arch_config = self.plans_manager.plans['configurations'][config]['architecture']
            self.print_to_log_file(f"Architecture config type: {type(arch_config)}")

            if isinstance(arch_config, dict):
                self.print_to_log_file(f"Architecture config keys: {list(arch_config.keys())}")
            else:
                raise ValueError(f"Invalid architecture config type: {type(arch_config)}")

        except Exception as e:
            self.print_to_log_file(f"Failed to retrieve architecture config: {str(e)}")
            # Fallback to default configuration
            arch_config = {
                'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                'arch_kwargs': {
                    'features_per_stage': [32, 64, 128, 256, 320, 320],
                    'n_stages': 6,
                    'conv_op': 'torch.nn.modules.conv.Conv3d'
                }
            }
            self.print_to_log_file("Using default architecture configuration.")

        # Extract network parameters and insert Transformer
        try:
            network_params = arch_config.get('arch_kwargs', arch_config)

            features = network_params.get('features_per_stage', [32, 64, 128, 256, 320, 320])
            self.print_to_log_file(f"Feature dimensions per stage: {features}")

            patch_size = self.configuration_manager.patch_size
            self.print_to_log_file(f"Patch size: {patch_size}")

            is_3d = len(patch_size) == 3
            self.print_to_log_file(f"Network dimensionality: {'3D' if is_3d else '2D'}")

            self.insert_transformer_encoder(features, patch_size, is_3d)

        except Exception as e:
            self.print_to_log_file(f"Transformer initialization failed: {str(e)}")
            raise

    def insert_transformer_encoder(self, features, patch_size, is_3d=True):
        """
        Insert the Conv-Transformer block at the specified encoder stage.
        For 2D pipelines, the block is applied directly to the feature map.
        For 3D pipelines, slices are processed independently in 2D and reassembled.
        """
        self.print_to_log_file("\n----- Inserting Transformer block -----")

        # Resolve insertion stage index
        insert_stage = self.transformer_config['insert_stage']
        if insert_stage < 0:
            insert_stage = len(features) + insert_stage
        insert_stage = max(0, min(insert_stage, len(features) - 1))
        self.print_to_log_file(f"Insertion stage: {insert_stage} / {len(features) - 1}")

        in_channels = features[insert_stage]
        self.print_to_log_file(f"Input channels at insertion stage: {in_channels}")

        # Estimate feature map spatial size at the insertion stage
        downsample_factor = 2 ** insert_stage if insert_stage > 0 else 1
        feature_map_size = [p // downsample_factor for p in patch_size]
        self.print_to_log_file(f"Estimated feature map size: {feature_map_size}")

        if is_3d:
            # For 3D data, use the two largest spatial dimensions for 2D slice processing
            h, w = sorted([feature_map_size[1], feature_map_size[2]], reverse=True)
            img_size = (h, w)
            self.print_to_log_file(f"3D mode: using spatial size {img_size} for slice-wise processing")
        else:
            img_size = feature_map_size
            self.print_to_log_file(f"2D mode: using spatial size {img_size}")

            vit_patch_size = self.transformer_config['vit_patches_size']
            min_dim = min(img_size)
            if min_dim < vit_patch_size * 2:
                self.print_to_log_file(
                    f"Warning: minimum spatial dimension {min_dim} < 2x patch size {vit_patch_size * 2}. "
                    f"Falling back to custom convolutional Transformer."
                )
                self.use_custom_transformer = True
            else:
                self.use_custom_transformer = False

        self.transformer_config['enable_3d'] = is_3d

        # Build and register the Transformer block
        try:
            if self.use_custom_transformer:
                self.trans_encoder = self.create_custom_transformer(in_channels, img_size).to(self.device)
                self.print_to_log_file(f"Custom Conv-Transformer loaded to device: {self.device}")
            else:
                if TransUNetEncoder is None:
                    raise ImportError(
                        "TransUNetEncoder is not available. "
                        "Please ensure trans_unet_encoder.py is present, "
                        "or set use_custom_transformer=True."
                    )
                encoder_params = inspect.signature(TransUNetEncoder.__init__).parameters
                self.print_to_log_file(f"TransUNetEncoder accepted parameters: {list(encoder_params.keys())}")

                valid_params = {
                    k: v for k, v in self.transformer_config.items()
                    if k in encoder_params and k not in ('insert_stage', 'enable_3d')
                }
                if valid_params.get('use_pretrained') and 'pretrained_model_path' in encoder_params:
                    valid_params['pretrained_model_path'] = None

                self.print_to_log_file(f"Using encoder parameters: {list(valid_params.keys())}")
                self.trans_encoder = TransUNetEncoder(
                    img_size=img_size,
                    in_channels=in_channels,
                    **valid_params
                ).to(self.device)
                self.print_to_log_file(f"TransUNetEncoder loaded to device: {self.device}")

            # Freeze Transformer at initialization
            self.freeze_transformer_layers()

        except Exception as e:
            self.print_to_log_file(f"Encoder initialization failed: {str(e)}")
            raise

        self.modify_network_forward(insert_stage)

    def freeze_transformer_layers(self):
        """Freeze all Transformer parameters (called at initialization)."""
        if self.freeze_transformer:
            for param in self.trans_encoder.parameters():
                param.requires_grad = False
            self.print_to_log_file("Transformer parameters frozen.")

    def unfreeze_transformer_blocks(self, current_epoch):
        """
        Progressively unfreeze Transformer parameter groups based on the current epoch.

        Output-focused schedule (manuscript Table/Supplementary Note S2):
          - Epoch  0-89: fully frozen
          - Epoch 90:    Group 1 released (["final"] - output-side mapping)
          - Epoch 110:   Group 2 released (FFN components added)
          - Epoch 130:   Group 3 released (attention projections added)
          - Epoch 150+:  Group 4 - full unfreezing
        """
        if not self.freeze_transformer:
            return  # Already fully unfrozen

        if current_epoch < self.base_training_epochs:
            return  # Still in the frozen base training phase

        epochs_after_base = current_epoch - self.base_training_epochs
        blocks_to_unfreeze = min(self.unfreeze_blocks, 1 + epochs_after_base // self.unfreeze_interval)

        if blocks_to_unfreeze <= 0:
            return

        if self.custom_unfreeze_order and self.unfreeze_layers:
            self._unfreeze_custom_layers()
            return

        if self.use_custom_transformer:
            self._unfreeze_custom_transformer_by_group(blocks_to_unfreeze)
        else:
            self._unfreeze_transformer_encoder_by_group(blocks_to_unfreeze)

    def _unfreeze_custom_layers(self):
        """
        Apply a one-shot custom unfreezing using the predefined layer_groups mapping.
        Used when custom_unfreeze_order is True.
        """
        self.print_to_log_file(f"Applying custom unfreeze order: {self.unfreeze_layers}")

        # Re-freeze all parameters first
        for param in self.trans_encoder.parameters():
            param.requires_grad = False

        for layer_id in self.unfreeze_layers:
            if layer_id not in self.layer_groups:
                self.print_to_log_file(f"Warning: unknown layer group ID {layer_id}, skipping.")
                continue

            layer_names = self.layer_groups[layer_id]

            if "all" in layer_names:
                for param in self.trans_encoder.parameters():
                    param.requires_grad = True
                self.print_to_log_file(f"Full unfreezing applied (group {layer_id}).")
                self.freeze_transformer = False
                return

            unfrozen_count = 0
            for name, param in self.trans_encoder.named_parameters():
                if any(ln in name for ln in layer_names):
                    param.requires_grad = True
                    unfrozen_count += 1
            self.print_to_log_file(
                f"Unfrozen group {layer_id} ({layer_names}): {unfrozen_count} parameter tensors."
            )

        trainable = sum(p.numel() for p in self.trans_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.trans_encoder.parameters())
        self.print_to_log_file(
            f"Transformer trainable parameters: {trainable}/{total} ({trainable / total * 100:.2f}%)"
        )

    def _unfreeze_custom_transformer_by_group(self, blocks_to_unfreeze):
        """
        Incrementally unfreeze the custom Conv-Transformer by group.

        Group 1: output-side mapping ('final')
        Group 2: FFN layers ('conv2', 'conv3') + output
        Group 3: attention projections ('qkv', 'proj')
        Group 4: full unfreezing
        """
        if blocks_to_unfreeze == 1:
            for name, param in self.trans_encoder.named_parameters():
                if 'final' in name:
                    param.requires_grad = True
            self.print_to_log_file("Unfrozen: output-side mapping layer (group 1).")
        elif blocks_to_unfreeze == 2:
            for name, param in self.trans_encoder.named_parameters():
                if any(k in name for k in ['final', 'conv2', 'conv3']):
                    param.requires_grad = True
            self.print_to_log_file("Unfrozen: FFN layers (group 2).")
        elif blocks_to_unfreeze == 3:
            for name, param in self.trans_encoder.named_parameters():
                if any(k in name for k in ['qkv', 'proj']):
                    param.requires_grad = True
            self.print_to_log_file("Unfrozen: attention projection layers (group 3).")
        elif blocks_to_unfreeze >= 4:
            for param in self.trans_encoder.parameters():
                param.requires_grad = True
            self.print_to_log_file("Full unfreezing applied (group 4).")
            self.freeze_transformer = False

    def set_unfreeze_layers(self, layer_ids):
        """
        Configure a one-shot custom unfreezing by specifying which layer groups to release.

        Args:
            layer_ids (list of int): group IDs to unfreeze.
                1 = output-side mapping ('final')
                2 = FFN components
                3 = attention projections
                4 = all parameters
        """
        self.unfreeze_layers = layer_ids
        self.custom_unfreeze_order = True
        self.print_to_log_file(f"Custom unfreeze layer groups set: {layer_ids}")
        for layer_id in layer_ids:
            if layer_id in self.layer_groups:
                self.print_to_log_file(f"  Group {layer_id}: {self.layer_groups[layer_id]}")
            else:
                self.print_to_log_file(f"  Warning: unknown group ID {layer_id}")

    def _unfreeze_transformer_encoder_by_group(self, blocks_to_unfreeze):
        """Incrementally unfreeze TransUNetEncoder layers by group."""
        if not (hasattr(self.trans_encoder, 'transformer') and
                hasattr(self.trans_encoder.transformer, 'encoder')):
            return

        encoder_layers = self.trans_encoder.transformer.encoder.layer

        if blocks_to_unfreeze == 1:
            if len(encoder_layers) > 0:
                for param in encoder_layers[-1].parameters():
                    param.requires_grad = True
                self.print_to_log_file("Unfrozen: last Transformer encoder layer (group 1).")
        elif blocks_to_unfreeze == 2:
            half = len(encoder_layers) // 2
            for i in range(half, len(encoder_layers)):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            self.print_to_log_file("Unfrozen: second half of Transformer encoder layers (group 2).")
        elif blocks_to_unfreeze == 3:
            for param in self.trans_encoder.transformer.encoder.parameters():
                param.requires_grad = True
            self.print_to_log_file("Unfrozen: all Transformer encoder layers (group 3).")
        elif blocks_to_unfreeze >= 4:
            for param in self.trans_encoder.parameters():
                param.requires_grad = True
            self.print_to_log_file("Full unfreezing applied (group 4).")
            self.freeze_transformer = False

    def create_custom_transformer(self, in_channels, img_size):
        """
        Build the lightweight custom Conv-Transformer block.

        Architecture (corresponds to manuscript Figure 1 / Methods):
          - Conv1 (3x3) + InstanceNorm + GELU: initial feature projection
          - QKV Conv (1x1): convolutional self-attention query/key/value projection
          - Proj Conv (1x1): attention output projection
          - FFN: Conv2 (1x1, expand x4) -> GELU -> Conv3 (1x1, compress)
          - Final Conv (3x3): output-side residual mapping back to in_channels
          - Global residual: output += input (skip connection)

        The 'final' layer is the first component released during output-focused unfreezing.
        """
        self.print_to_log_file(
            f"Building custom Conv-Transformer: in_channels={in_channels}, img_size={img_size}"
        )

        class CustomTransformer(torch.nn.Module):
            def __init__(self, in_channels, hidden_dim=320):
                super().__init__()
                # Initial feature projection
                self.conv1 = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
                self.norm1 = torch.nn.InstanceNorm2d(hidden_dim)
                self.act1 = torch.nn.GELU()

                # Convolutional self-attention (QKV projection)
                self.qkv = torch.nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1)
                self.proj = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

                # Feed-forward network (FFN)
                self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)
                self.act2 = torch.nn.GELU()
                self.conv3 = torch.nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)
                self.norm2 = torch.nn.InstanceNorm2d(hidden_dim)

                # Output-side residual mapping (first component released in unfreezing schedule)
                self.final = torch.nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

                self._init_weights()

            def _init_weights(self):
                """Kaiming initialization for conv layers; small std for attention projections."""
                for m in self.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            torch.nn.init.constant_(m.bias, 0)
                # Smaller initialization for attention projections to stabilize early training
                torch.nn.init.normal_(self.qkv.weight, std=0.01)
                torch.nn.init.normal_(self.proj.weight, std=0.01)

            def forward(self, x):
                # Initial feature extraction
                h = self.conv1(x)
                h = self.norm1(h)
                h = self.act1(h)

                residual = h

                # Convolutional self-attention
                B, C, H, W = h.shape
                qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
                attn = torch.softmax(attn, dim=-1)
                h = (attn @ v).reshape(B, C, H, W)
                h = self.proj(h)

                # First residual connection (post-attention)
                h = h + residual

                # FFN with second residual connection
                residual = h
                h = self.conv2(h)
                h = self.act2(h)
                h = self.conv3(h)
                h = h + residual
                h = self.norm2(h)

                # Output-side mapping + global skip connection to input
                out = self.final(h)
                out = out + x  # Global residual: preserves input feature map

                return out, None  # Second return value reserved for attention maps

        return CustomTransformer(in_channels)

    def modify_network_forward(self, insert_stage):
        """
        Wrap the nnU-Net encoder to inject the Conv-Transformer block at the specified stage.
        The Transformer is only active when freeze_transformer is False or
        current_epoch >= base_training_epochs.
        """
        self.print_to_log_file("\n----- Modifying network forward pass -----")

        network = self.network
        if hasattr(network, 'module'):
            network = network.module
            self.print_to_log_file("Detected DataParallel-wrapped network.")

        if not hasattr(network, 'encoder') or not hasattr(network.encoder, 'stages'):
            self.print_to_log_file(f"Network attributes: {dir(network)}")
            raise AttributeError(
                "Incompatible network structure: requires encoder.stages attribute."
            )

        original_encoder = network.encoder
        self.print_to_log_file(f"Original encoder type: {type(original_encoder)}")

        class TransformerEncoder(torch.nn.Module):
            def __init__(self_m, original_encoder, transformer, insert_at, trainer_ref):
                super().__init__()
                self_m.stages = original_encoder.stages
                self_m.transformer = transformer
                self_m.insert_at = insert_at
                self_m.trainer_ref = trainer_ref
                self_m.shape_logged = False

            def forward(self_m, x):
                skips = []
                for stage_idx, stage in enumerate(self_m.stages):
                    x = stage(x)

                    if stage_idx == self_m.insert_at:
                        shape_before = x.shape
                        # Activate Transformer once unfreezing begins
                        use_transformer = (
                            not self_m.trainer_ref.freeze_transformer or
                            self_m.trainer_ref.current_epoch >= self_m.trainer_ref.base_training_epochs
                        )

                        if use_transformer:
                            if len(x.shape) == 5:  # 3D input: [B, C, D, H, W]
                                B, C, D, H, W = x.shape
                                if not self_m.shape_logged:
                                    self_m.trainer_ref.print_to_log_file(
                                        f"Transformer input shape: {shape_before}"
                                    )
                                # Process each slice independently in 2D
                                x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
                                x_trans, _ = self_m.transformer(x_2d)
                                _, C_new, H_new, W_new = x_trans.shape
                                x = x_trans.view(B, D, C_new, H_new, W_new).permute(0, 2, 1, 3, 4).contiguous()
                                if not self_m.shape_logged:
                                    self_m.trainer_ref.print_to_log_file(
                                        f"Transformer output shape: {x.shape}"
                                    )
                                    self_m.shape_logged = True
                            else:  # 2D input: [B, C, H, W]
                                if not self_m.shape_logged:
                                    self_m.trainer_ref.print_to_log_file(
                                        f"Transformer input shape: {shape_before}"
                                    )
                                x, _ = self_m.transformer(x)
                                if not self_m.shape_logged:
                                    self_m.trainer_ref.print_to_log_file(
                                        f"Transformer output shape: {x.shape}"
                                    )
                                    self_m.shape_logged = True

                    skips.append(x)
                return skips

        network.encoder = TransformerEncoder(
            original_encoder, self.trans_encoder, insert_stage, self
        )
        self.print_to_log_file(
            f"Network forward pass modified: Transformer inserted at stage {insert_stage}."
        )
        self.print_to_log_file(
            f"Initial state: {'base nnU-Net (Transformer frozen)' if self.freeze_transformer else 'OF-TransUNet active'}"
        )

    def configure_optimizer(self):
        """
        Configure SGD optimizer with separate parameter groups for encoder, decoder,
        and (when active) Transformer components.

        Optimizer settings (consistent with manuscript Methods):
          - SGD with momentum=0.99, Nesterov=True
          - Initial LR: 0.01 for encoder/decoder; 0.001 for Transformer (unfreeze_lr_factor=0.1)
          - Poly LR scheduler
        """
        self.print_to_log_file("\n----- Configuring optimizer -----")

        try:
            network = self.network
            if hasattr(network, 'module'):
                network = network.module

            encoder_params = []
            decoder_params = []
            trans_params = []

            for name, param in network.named_parameters():
                if 'encoder' in name and 'transformer' not in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)

            for name, param in self.trans_encoder.named_parameters():
                if param.requires_grad:
                    trans_params.append(param)

            encoder_count = sum(p.numel() for p in encoder_params)
            decoder_count = sum(p.numel() for p in decoder_params)
            trans_count = sum(p.numel() for p in trans_params)
            self.print_to_log_file(
                f"Parameter counts — Encoder: {encoder_count / 1e6:.2f}M, "
                f"Decoder: {decoder_count / 1e6:.2f}M, "
                f"Transformer (trainable): {trans_count / 1e6:.2f}M"
            )

            param_groups = [
                {"params": encoder_params, "lr": self.initial_lr},
                {"params": decoder_params, "lr": self.initial_lr * 0.5},
            ]

            if len(trans_params) > 0:
                trans_lr = self.initial_lr * self.unfreeze_lr_factor
                param_groups.append({"params": trans_params, "lr": trans_lr})
                self.print_to_log_file(f"Transformer parameter group LR: {trans_lr:.6f}")

            # SGD optimizer: consistent with nnU-Net default and manuscript description
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.initial_lr,
                momentum=0.99,
                nesterov=True,
                weight_decay=3e-5
            )
            self.print_to_log_file(f"Optimizer: {type(optimizer).__name__} (momentum=0.99, Nesterov=True)")

            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.max_num_epochs)

            return optimizer, lr_scheduler

        except Exception as e:
            self.print_to_log_file(f"Optimizer configuration failed: {str(e)}")
            self.print_to_log_file("Falling back to default nnU-Net optimizer.")
            return super().configure_optimizer()

    def handle_nan_in_loss(self, loss_value):
        """
        Detect NaN/Inf in loss. If detected for max_nan_counter consecutive steps,
        halve the learning rate (lr_reduce_factor=0.5) down to min_lr.
        Returns True if NaN/Inf was detected (caller should skip the update step).
        """
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            self.nan_counter += 1
            self.print_to_log_file(
                f"Warning: NaN/Inf loss detected (consecutive count: {self.nan_counter})."
            )

            if self.nan_counter >= self.max_nan_counter:
                current_lrs = [g['lr'] for g in self.optimizer.param_groups]
                if min(current_lrs) > self.min_lr:
                    for g in self.optimizer.param_groups:
                        g['lr'] = max(g['lr'] * self.lr_reduce_factor, self.min_lr)
                    new_lrs = [g['lr'] for g in self.optimizer.param_groups]
                    self.print_to_log_file(
                        f"LR reduced after {self.nan_counter} consecutive NaN events: {new_lrs}"
                    )
                    self.nan_counter = 0
                    self.optimizer.zero_grad()
            return True
        else:
            self.nan_counter = 0
            return False

    def _check_and_fix_gradients(self):
        """
        Scan all network parameters for NaN/Inf or complex-valued gradients.
        Complex gradients are replaced with their real part; NaN/Inf gradients are zeroed.
        Returns True if any problematic gradient was found.
        """
        has_bad_grad = False
        for name, param in self.network.named_parameters():
            if param.grad is None:
                continue
            if torch.is_complex(param.grad):
                self.print_to_log_file(
                    f"Warning: complex-valued gradient in {name}. Taking real part."
                )
                param.grad = torch.real(param.grad)
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_bad_grad = True
                self.print_to_log_file(
                    f"Warning: NaN/Inf gradient in {name}. Zeroing gradient."
                )
                param.grad = torch.zeros_like(param.grad)
        return has_bad_grad

    def train_step(self, batch):
        """
        Single training step with:
          - Mixed-precision forward pass (CUDA only)
          - NaN/Inf loss detection and LR reduction
          - Gradient clipping (norm threshold: grad_clip_value)
          - Complex/NaN/Inf gradient detection and correction
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        if self.device.type == 'cuda':
            with autocast(enabled=True):
                output = self.network(data)
                l = self.loss(output, target)

            if self.handle_nan_in_loss(l):
                return {'loss': np.array([1.0])}

            self.grad_scaler.scale(l).backward()

            if self.enable_grad_clipping:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_value)

            has_bad_grad = self._check_and_fix_gradients()
            if has_bad_grad:
                self.nan_counter += 1
                if self.nan_counter >= self.max_nan_counter:
                    for g in self.optimizer.param_groups:
                        g['lr'] = max(g['lr'] * self.lr_reduce_factor, self.min_lr)
                    self.print_to_log_file(
                        f"LR reduced after {self.nan_counter} consecutive bad gradient steps: "
                        f"{[g['lr'] for g in self.optimizer.param_groups]}"
                    )
                    self.nan_counter = 0
            else:
                self.nan_counter = 0

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        else:
            output = self.network(data)
            l = self.loss(output, target)

            if self.handle_nan_in_loss(l):
                return {'loss': np.array([1.0])}

            l.backward()

            if self.enable_grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_value)

            has_bad_grad = self._check_and_fix_gradients()
            if has_bad_grad:
                self.nan_counter += 1
                if self.nan_counter >= self.max_nan_counter:
                    for g in self.optimizer.param_groups:
                        g['lr'] = max(g['lr'] * self.lr_reduce_factor, self.min_lr)
                    self.print_to_log_file(
                        f"LR reduced after {self.nan_counter} consecutive bad gradient steps."
                    )
                    self.nan_counter = 0
            else:
                self.nan_counter = 0

            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def run_iteration(self, data_dict, train=True):
        """Override run_iteration with additional CUDA and NaN error handling."""
        try:
            if train:
                return self.train_step(data_dict)
            else:
                return super().run_iteration(data_dict, train=False)
        except RuntimeError as e:
            err = str(e)
            if "CUDA out of memory" in err:
                self.print_to_log_file(f"CUDA OOM error: {err}")
                torch.cuda.empty_cache()
                return {'loss': np.array([1.0])}
            elif "value cannot be converted" in err or "nan" in err.lower():
                self.print_to_log_file(f"NaN-related runtime error: {err}")
                for name, param in self.network.named_parameters():
                    if torch.isnan(param).any():
                        self.print_to_log_file(f"NaN detected in parameter: {name}")
                return {'loss': np.array([1.0])}
            else:
                raise

    def on_epoch_end(self):
        """
        End-of-epoch hook:
          - Triggers progressive unfreezing at the appropriate epoch boundaries
          - Reconfigures optimizer when new parameters become trainable
          - Applies LR warmup for the first 5 epochs after unfreezing begins
          - Logs LR and parameter health every 50 epochs
        """
        current_epoch = self.current_epoch
        super().on_epoch_end()

        if self.custom_unfreeze_order:
            # One-shot custom unfreezing at the base epoch boundary
            if current_epoch == self.base_training_epochs:
                self.unfreeze_transformer_blocks(current_epoch)
                self.print_to_log_file(f"Epoch {current_epoch}: custom unfreeze order applied.")
                self.optimizer, self.lr_scheduler = self.configure_optimizer()
                if hasattr(self, 'grad_scaler') and self.device.type == 'cuda':
                    self.grad_scaler = GradScaler()
        else:
            self.unfreeze_transformer_blocks(current_epoch)
            if (current_epoch == self.base_training_epochs or
                    (current_epoch > self.base_training_epochs and
                     (current_epoch - self.base_training_epochs) % self.unfreeze_interval == 0 and
                     self.freeze_transformer)):
                self.print_to_log_file(
                    f"Epoch {current_epoch}: reconfiguring optimizer for newly unfrozen parameters."
                )
                self.optimizer, self.lr_scheduler = self.configure_optimizer()
                if hasattr(self, 'grad_scaler') and self.device.type == 'cuda':
                    self.grad_scaler = GradScaler()

        # Periodic health check every 50 epochs
        if current_epoch % 50 == 0:
            lrs = [g['lr'] for g in self.optimizer.param_groups]
            self.print_to_log_file(f"Epoch {current_epoch} — current LRs: {lrs}")
            has_nan = any(
                torch.isnan(p).any()
                for p in self.network.parameters()
            )
            if has_nan:
                self.print_to_log_file(f"Epoch {current_epoch} — Warning: NaN detected in network parameters!")
            else:
                self.print_to_log_file(f"Epoch {current_epoch} — All parameters healthy (no NaN).")

        # LR warmup for Transformer parameters during epochs [base, base+5)
        if self.base_training_epochs <= current_epoch < self.base_training_epochs + 5:
            self.warm_up_transformer_lr(current_epoch, self.base_training_epochs)

    def warm_up_transformer_lr(self, current_epoch, unfreeze_epoch, warmup_epochs=5):
        """
        Linear LR warmup for the Transformer parameter group over warmup_epochs.
        Target LR = initial_lr * unfreeze_lr_factor (e.g., 1e-3).
        Warmup period: [unfreeze_epoch, unfreeze_epoch + warmup_epochs).
        """
        if unfreeze_epoch <= current_epoch < unfreeze_epoch + warmup_epochs:
            warmup_factor = (current_epoch - unfreeze_epoch + 1) / warmup_epochs
            target_lr = self.initial_lr * self.unfreeze_lr_factor
            warmed_lr = target_lr * warmup_factor
            for i, g in enumerate(self.optimizer.param_groups):
                if i == 2:  # Transformer parameter group (third group)
                    g['lr'] = warmed_lr
            self.print_to_log_file(
                f"Epoch {current_epoch}: Transformer LR warmup — {warmed_lr:.6f} "
                f"(factor {warmup_factor:.2f} of target {target_lr:.6f})"
            )

    def on_train_end(self):
        """Save Transformer weights at the end of training, then call parent cleanup."""
        self.print_to_log_file("\n----- Training complete -----")
        try:
            trans_path = os.path.join(self.output_folder, 'transformer_final.pth')
            torch.save(self.trans_encoder.state_dict(), trans_path)
            self.print_to_log_file(f"Transformer weights saved to: {trans_path}")
        except Exception as e:
            self.print_to_log_file(f"Failed to save Transformer weights: {str(e)}")
        super().on_train_end()