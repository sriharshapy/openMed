#!/usr/bin/env python3
"""
Vision Transformer (ViT) Model for OpenMed - Inference Only
A Vision Transformer implementation with ImageNet pretrained weights for medical image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple
from einops import rearrange

from .base_model import BaseModel

class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding using conv2d
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Store attention weights for visualization (detached to avoid gradient issues)
        self.last_attention_weights = attn.detach().clone()
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class ViTModel(BaseModel):
    """
    Vision Transformer (ViT) model for medical image classification - Inference Only.
    
    Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    by Dosovitskiy et al.
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 pretrained: bool = True,
                 **kwargs):
        """
        Initialize Vision Transformer model for inference.
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of image patches
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
            pretrained: Whether to use pretrained weights (will try to load from timm)
        """
        super(ViTModel, self).__init__(num_classes=num_classes, **kwargs)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.pretrained = pretrained
        
        # Calculate number of patches
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Try to load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()
        
        # Set to evaluation mode by default
        self.eval()
        
        self.logger.info(f"ViT initialized: {embed_dim}D embeddings, {depth} layers, {num_heads} heads")
        self.logger.info(f"Patch size: {patch_size}x{patch_size}, Image size: {img_size}x{img_size}")
        self.logger.info(f"Number of patches: {self.n_patches}")
        
        # Print model info
        self.print_model_info()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _load_pretrained_weights(self):
        """Try to load pretrained weights from timm if available."""
        try:
            import timm
            # Try to load a similar ViT model from timm
            pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            pretrained_state = pretrained_model.state_dict()
            
            # Map the state dict keys to our model
            our_state = self.state_dict()
            loaded_keys = []
            
            # Map common keys
            key_mapping = {
                'patch_embed.proj.weight': 'patch_embed.projection.weight',
                'patch_embed.proj.bias': 'patch_embed.projection.bias',
                'cls_token': 'cls_token',
                'pos_embed': 'pos_embed',
                'norm.weight': 'norm.weight',
                'norm.bias': 'norm.bias'
            }
            
            # Load mapped keys
            for pretrained_key, our_key in key_mapping.items():
                if pretrained_key in pretrained_state and our_key in our_state:
                    if pretrained_state[pretrained_key].shape == our_state[our_key].shape:
                        our_state[our_key] = pretrained_state[pretrained_key]
                        loaded_keys.append(our_key)
            
            # Load transformer blocks
            for i in range(min(self.depth, len([k for k in pretrained_state.keys() if k.startswith('blocks.')])//12)):
                block_mappings = {
                    f'blocks.{i}.norm1.weight': f'blocks.{i}.norm1.weight',
                    f'blocks.{i}.norm1.bias': f'blocks.{i}.norm1.bias',
                    f'blocks.{i}.attn.qkv.weight': f'blocks.{i}.attn.qkv.weight',
                    f'blocks.{i}.attn.qkv.bias': f'blocks.{i}.attn.qkv.bias',
                    f'blocks.{i}.attn.proj.weight': f'blocks.{i}.attn.proj.weight',
                    f'blocks.{i}.attn.proj.bias': f'blocks.{i}.attn.proj.bias',
                    f'blocks.{i}.norm2.weight': f'blocks.{i}.norm2.weight',
                    f'blocks.{i}.norm2.bias': f'blocks.{i}.norm2.bias',
                    f'blocks.{i}.mlp.fc1.weight': f'blocks.{i}.mlp.0.weight',
                    f'blocks.{i}.mlp.fc1.bias': f'blocks.{i}.mlp.0.bias',
                    f'blocks.{i}.mlp.fc2.weight': f'blocks.{i}.mlp.3.weight',
                    f'blocks.{i}.mlp.fc2.bias': f'blocks.{i}.mlp.3.bias',
                }
                
                for pretrained_key, our_key in block_mappings.items():
                    if pretrained_key in pretrained_state and our_key in our_state:
                        if pretrained_state[pretrained_key].shape == our_state[our_key].shape:
                            our_state[our_key] = pretrained_state[pretrained_key]
                            loaded_keys.append(our_key)
            
            # Load the updated state dict
            self.load_state_dict(our_state)
            self.logger.info(f"Loaded pretrained weights for {len(loaded_keys)} parameters from timm")
            
        except Exception as e:
            self.logger.warning(f"Could not load pretrained weights: {e}")
            self.logger.info("Continuing with random initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use only the class token)
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        logits = self.head(cls_token_final)  # (batch_size, num_classes)
        
        return logits
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the model (everything except the classification head).
        
        Returns:
            Feature extractor module
        """
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.patch_embed = vit_model.patch_embed
                self.cls_token = vit_model.cls_token
                self.pos_embed = vit_model.pos_embed
                self.dropout = vit_model.dropout
                self.blocks = vit_model.blocks
                self.norm = vit_model.norm
            
            def forward(self, x):
                batch_size = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)
                
                # Add class token
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                
                # Add positional embedding
                x = x + self.pos_embed
                x = self.dropout(x)
                
                # Apply transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                # Final layer norm
                x = self.norm(x)
                
                # Return class token features
                return x[:, 0]  # (batch_size, embed_dim)
        
        return ViTFeatureExtractor(self)
    
    def get_classifier(self) -> nn.Module:
        """
        Get the classification head of the model.
        
        Returns:
            Classifier module
        """
        return self.head
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input without classification.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, embed_dim)
        """
        feature_extractor = self.get_feature_extractor()
        return feature_extractor(x)
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention weights from a specific transformer layer.
        
        Args:
            x: Input tensor
            layer_idx: Index of the transformer layer (-1 for last layer)
            
        Returns:
            Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        if layer_idx < 0:
            layer_idx = len(self.blocks) + layer_idx
        
        if layer_idx >= len(self.blocks) or layer_idx < 0:
            raise ValueError(f"Invalid layer index: {layer_idx}. Must be between 0 and {len(self.blocks)-1}")
        
        # Clear any existing stored attention weights
        for block in self.blocks:
            block.attn.last_attention_weights = None
        
        # Forward pass to generate attention weights
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)
        
        # Get the attention weights from the specified layer
        target_attention = self.blocks[layer_idx].attn.last_attention_weights
        
        if target_attention is None:
            raise RuntimeError(f"No attention weights captured from layer {layer_idx}. "
                             "Make sure the forward pass completed successfully.")
        
        return target_attention
    
    def get_patch_attention_map(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention map for patches (excluding class token).
        
        Args:
            x: Input tensor
            layer_idx: Index of the transformer layer
            
        Returns:
            Attention map tensor of shape (batch_size, num_heads, n_patches, n_patches)
        """
        attention_weights = self.get_attention_weights(x, layer_idx)
        # Remove class token attention (first row and column)
        patch_attention = attention_weights[:, :, 1:, 1:]  # (batch_size, num_heads, n_patches, n_patches)
        return patch_attention
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """
        Get ViT-specific model information.
        
        Returns:
            Dictionary with ViT-specific information
        """
        base_info = self.get_model_info()
        vit_info = {
            'architecture': 'Vision Transformer (ViT)',
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'n_patches': self.n_patches,
            'pretrained': self.pretrained,
            'input_size': f'{self.img_size}x{self.img_size}',
            'normalization': 'ImageNet normalization recommended'
        }
        return {**base_info, **vit_info} 