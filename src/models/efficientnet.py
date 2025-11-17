# train_efficientnet.py
import logging
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.preprocess.data_handler import get_dataloaders
from src.utils import get_device, run_epoch, save_model, set_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained model (changed from b0 to b3)
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b3'
        )
        
        # Get the number of features from the backbone
        in_features = self.backbone._fc.in_features
        
        # Replace classifier with custom head
        self.backbone._fc = nn.Sequential(
            nn.BatchNorm1d(in_features),  # Added normalization
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1024),  # Increased from 512 to 1024 for B3
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        
        # Initialize the new layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.backbone._fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_optimizer(self):
        # Different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        # Separate backbone and classifier parameters
        for name, param in self.named_parameters():
            if '_fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5},  # Slightly lower LR for B3
            {'params': classifier_params, 'lr': 5e-5}  # Adjusted LR for larger model
        ], weight_decay=1e-5)
        
        return optimizer

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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        self.checkpoint_path = checkpoint_path

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
        'image_size': 300,
        'batch_size': 32,
        'num_epochs': 15,
        'experiment_name': 'deepfake_efficientnet',
        'tracking_uri': 'file:./mlruns',
        'seed': 42,
        'checkpoint_path': 'models/best_efficientnet.pt',
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
            'model_type': 'efficientnet-b3',
            'image_size': config.image_size,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'optimizer': 'Adam',
            'backbone_lr': 1e-5,
            'classifier_lr': 5e-5,
            'weight_decay': 1e-5,
        })

        model = DeepfakeEfficientNet()
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