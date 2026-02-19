"""
Training script for Multi-Aspect Sentiment Analysis
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import create_model
from models.losses import AspectSpecificLossManager
from utils.data_utils import create_dataloaders, DependencyParser, compute_class_weights
from utils.metrics import AspectSentimentEvaluator


class Trainer:
    """
    Trainer for multi-aspect sentiment analysis model
    """
    def __init__(self, config_path):
        """
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds for reproducibility
        self.set_seed(self.config['experiment']['seed'])
        
        # Setup device
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create save directory
        self.save_dir = Path(self.config['experiment']['save_dir']) / self.config['experiment']['name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {self.save_dir}")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config['model']['roberta_model']
        )
        
        # Initialize dependency parser (optional)
        self.dependency_parser = None
        if self.config['data'].get('use_dependency_parsing', False):
            print("Initializing dependency parser...")
            self.dependency_parser = DependencyParser(
                language=self.config['data'].get('language', 'vi')
            )
        
        # Create dataloaders
        print("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config, self.tokenizer, self.dependency_parser
        )
        
        # Create model
        print("Creating model...")
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Compute class weights and create loss manager
        print("Computing class weights...")
        aspect_class_counts = compute_class_weights(
            self.config['data']['train_path'],
            self.config['aspects']['names'],
            self.config['aspects']['label_map']
        )
        
        self.loss_manager = AspectSpecificLossManager(
            aspect_class_counts,
            self.config['training']
        )
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Create learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Mixed precision training
        self.use_amp = self.config['hardware'].get('mixed_precision', False)
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using automatic mixed precision training")
        
        # Initialize wandb (optional)
        if self.config['experiment'].get('use_wandb', False):
            wandb.init(
                project=self.config['experiment']['wandb_project'],
                name=self.config['experiment']['name'],
                config=self.config
            )      
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.save_dir, flush_secs=30)
        
        # Tracking variables
        self.global_step = 0
        self.best_val_metric = 0
        self.patience_counter = 0
        
        # Evaluator
        self.evaluator = AspectSentimentEvaluator(self.config['aspects']['names'])
    
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        aspect_losses = {aspect: [] for aspect in self.config['aspects']['names']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            aspect_ids = batch['aspect_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare edge indices if using GCN
            edge_indices = None
            if self.config['model'].get('use_dependency_gcn', False):
                edge_indices = [e.to(self.device) if e is not None else None 
                               for e in batch['edge_indices']]
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(
                        input_ids, attention_mask, aspect_ids, edge_indices
                    )
                    loss, loss_details = self.loss_manager.compute_loss(
                        predictions, labels, aspect_ids, 
                        self.config['aspects']['names']
                    )
            else:
                predictions = self.model(
                    input_ids, attention_mask, aspect_ids, edge_indices
                )
                loss, loss_details = self.loss_manager.compute_loss(
                    predictions, labels, aspect_ids, 
                    self.config['aspects']['names']
                )
            
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log
            if self.global_step % self.config['experiment']['log_every'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log to TensorBoard
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/avg_loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

                # NOTE: This mid-epoch validation is supplementary. Main early stopping check is after each epoch.
                if self.config['experiment'].get('use_wandb', False):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
            
            # Evaluate
            if self.global_step % self.config['experiment']['eval_every'] == 0:
                val_metrics = self.evaluate(self.val_loader, "Validation")
                self.model.train()  # Back to training mode
                
                # Check for improvement
                val_metric = val_metrics['overall']['macro_f1']
                if val_metric > self.best_val_metric:
                    self.best_val_metric = val_metric
                    self.patience_counter = 0
                    self.save_checkpoint(f'best_model.pt')
                    print(f"New best model! Val Macro-F1: {val_metric:.4f}")
                else:
                    self.patience_counter += 1
                
                # Log to TensorBoard
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('val/macro_f1', val_metric, self.global_step)
                    self.writer.add_scalar('val/accuracy', val_metrics['overall']['accuracy'], self.global_step)
                    self.writer.add_scalar('val/weighted_f1', val_metrics['overall']['weighted_f1'], self.global_step)
                    
                    for aspect, metrics in val_metrics['aspects'].items():
                        self.writer.add_scalar(f'val/aspect/{aspect}_macro_f1', metrics['macro_f1'], self.global_step)

                # Log to wandb
                if self.config['experiment'].get('use_wandb', False):
                    wandb.log({
                        'val/macro_f1': val_metric,
                        'val/accuracy': val_metrics['overall']['accuracy'],
                        'global_step': self.global_step
                    })
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, dataloader, split_name="Validation"):
        """Evaluate model on validation or test set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_aspects = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                aspect_ids = batch['aspect_ids'].to(self.device)
                labels = batch['labels']
                
                # Prepare edge indices if using GCN
                edge_indices = None
                if self.config['model'].get('use_dependency_gcn', False):
                    edge_indices = [e.to(self.device) if e is not None else None 
                                   for e in batch['edge_indices']]
                
                # Forward pass
                predictions = self.model(
                    input_ids, attention_mask, aspect_ids, edge_indices
                )
                
                # Get predicted classes
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                
                all_predictions.extend(pred_classes)
                all_labels.extend(labels.numpy())
                all_aspects.extend(batch['aspects'])
        
        # Compute metrics for each aspect
        aspect_metrics = {}
        for aspect in self.config['aspects']['names']:
            # Filter samples for this aspect
            aspect_mask = np.array([a == aspect for a in all_aspects])
            if aspect_mask.sum() == 0:
                continue
            
            y_true = np.array(all_labels)[aspect_mask]
            y_pred = np.array(all_predictions)[aspect_mask]
            
            metrics = self.evaluator.evaluate_aspect(y_true, y_pred, aspect)
            aspect_metrics[aspect] = metrics
        
        # Compute overall metrics
        overall_metrics = self.evaluator.evaluate_aspect(
            np.array(all_labels),
            np.array(all_predictions),
            'overall'
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"{split_name} Results")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"Overall Macro-F1: {overall_metrics['macro_f1']:.4f}")
        print(f"Overall Weighted-F1: {overall_metrics['weighted_f1']:.4f}")
        
        print(f"\nPer-Aspect Macro-F1:")
        for aspect in self.config['aspects']['names']:
            if aspect in aspect_metrics:
                print(f"  {aspect}: {aspect_metrics[aspect]['macro_f1']:.4f}")
        
        return {
            'overall': overall_metrics,
            'aspects': aspect_metrics
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, "Validation")
            
            # Update patience counter based on validation performance
            val_metric = val_metrics['overall']['macro_f1']
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"New best model! Val Macro-F1: {val_metric:.4f}")
            else:
                self.patience_counter += 1
            
            print(f"Patience: {self.patience_counter}/{self.config['training']['early_stopping_patience']}")
            
            # Check early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Load best model and evaluate on test set
        print(f"\n{'='*60}")
        print("Training completed! Evaluating best model on test set...")
        print(f"{'='*60}\n")
        
        best_model_path = self.save_dir / 'best_model.pt'
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
        
        test_metrics = self.evaluate(self.test_loader, "Test")
        
        # Save test results
        import json
        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_metrics = convert_to_serializable(test_metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\nTest results saved to {results_path}")
        
        # Close wandb
        if self.config['experiment'].get('use_wandb', False):
            wandb.finish()
        
        # Close TensorBoard writer
        if hasattr(self, 'writer'):
            self.writer.close()
        
        return test_metrics


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Aspect Sentiment Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
