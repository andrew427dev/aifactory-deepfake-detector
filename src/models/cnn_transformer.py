import logging
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.preprocess.data_handler import get_dataloaders
from src.utils import get_device, run_epoch, save_model, set_seed
from tqdm import tqdm

from src.preprocess.data_handler import get_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DeepfakeCNNTransformer(nn.Module):
    def __init__(self, num_transformer_layers=6):
        super().__init__()
        # CNN Backbone (EfficientNet-B0)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            self.num_patches = features.shape[2] * features.shape[3]
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.feature_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=self.feature_dim,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1
            ) for _ in range(num_transformer_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(self.feature_dim)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Extract CNN features
        features = self.backbone(x)  # B, C, H, W
        
        # Reshape to sequence
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Add position embeddings
        features = features + self.pos_embed
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            features = layer(features)
        
        # Global average pooling
        features = features.mean(dim=1)
        
        # Normalize and classify
        features = self.norm(features)
        return self.head(features)

class DeepfakeTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        checkpoint_path: str | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.checkpoint_path = checkpoint_path
        
        # Different learning rates for different components
        backbone_params = []
        transformer_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif any(x in name for x in ['transformer', 'pos_embed']):
                transformer_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': transformer_params, 'lr': 5e-5},
            {'params': head_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
    
    def train_epoch(self, epoch):
        return run_epoch(
            self.model,
            self.train_loader,
            self.criterion,
            self.device,
            optimizer=self.optimizer,
            use_tqdm=True,
            desc=f'Epoch {epoch}',
        )

    def validate(self, loader):
        return run_epoch(
            self.model,
            loader,
            self.criterion,
            self.device,
        )
    
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, step=epoch)
    
    def log_plots(self, y_true, y_pred, phase='train'):
        # Confusion Matrix
        cm = confusion_matrix(y_true, np.array(y_pred) > 0.5)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'{phase} Confusion Matrix')
        mlflow.log_figure(plt.gcf(), f'{phase}_confusion_matrix.png')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{phase} ROC Curve')
        plt.legend()
        mlflow.log_figure(plt.gcf(), f'{phase}_roc_curve.png')
        plt.close()
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_preds, train_targets = self.train_epoch(epoch)
            val_loss, val_acc, val_preds, val_targets = self.validate(self.val_loader)
            
            # Log metrics
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Log plots every few epochs
            if epoch % 5 == 0:
                self.log_plots(train_targets, train_preds, 'train')
                self.log_plots(val_targets, val_preds, 'validation')
            
            # Model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(self.model, "best_model")
                if self.checkpoint_path:
                    save_model(self.model, self.checkpoint_path)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                       f'Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, '
                       f'Val Acc={val_acc:.4f}')
    
    def test(self):
        test_loss, test_acc, test_preds, test_targets = self.validate(self.test_loader)
        self.log_plots(test_targets, test_preds, 'test')
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        return test_loss, test_acc

def _build_config(config: SimpleNamespace | None = None) -> SimpleNamespace:
    defaults = {
        'data_dir': 'data/processed',
        'image_size': 224,
        'batch_size': 64,
        'num_epochs': 15,
        'experiment_name': 'deepfake_cnn_transformer',
        'tracking_uri': 'file:./mlruns',
        'seed': 42,
        'checkpoint_path': 'models/best_cnn_transformer.pt',
    }
    if config is None:
        config = SimpleNamespace()
    for key, value in defaults.items():
        if not hasattr(config, key) or getattr(config, key) is None:
            setattr(config, key, value)
    if not hasattr(config, 'device') or config.device is None:
        config.device = get_device()
    return config


def run_training(config: SimpleNamespace | None = None) -> None:
    config = _build_config(config)

    set_seed(config.seed)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_dir,
        config.image_size,
        config.batch_size,
        val_dir=getattr(config, 'val_data_dir', None),
        test_dir=getattr(config, 'test_data_dir', None),
    )

    with mlflow.start_run():
        mlflow.log_params({
            'model_type': 'cnn_transformer',
            'image_size': config.image_size,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'optimizer': 'AdamW',
            'backbone_lr': 1e-5,
            'transformer_lr': 5e-5,
            'head_lr': 1e-4,
            'weight_decay': 0.01,
        })

        model = DeepfakeCNNTransformer()
        trainer = DeepfakeTrainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            config.device,
            checkpoint_path=config.checkpoint_path,
        )

        trainer.train(config.num_epochs)

        test_loss, test_acc = trainer.test()
        logger.info('Test Loss: %.4f, Test Accuracy: %.4f', test_loss, test_acc)