"""
Data processing utilities for multi-aspect sentiment analysis
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import re
from collections import Counter


class CosmeticReviewDataset(Dataset):
    """
    Dataset for cosmetic product reviews with multi-aspect sentiment labels
    """
    def __init__(self, data_path, tokenizer, config, aspect_names, is_train=True):
        """
        Args:
            data_path: Path to CSV file
            tokenizer: RoBERTa tokenizer
            config: Configuration dictionary
            aspect_names: List of aspect names
            is_train: Whether this is training data
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.aspect_names = aspect_names
        self.is_train = is_train
        
        data_config = config['data']
        self.max_length = data_config['max_seq_length']
        self.text_column = data_config['text_column']
        
        # Label mapping
        self.label_map = config['aspects']['label_map']
        
        # Prepare data samples
        self.samples = self._prepare_samples()
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
        self._print_statistics()
    
    def _prepare_samples(self):
        """
        Convert dataframe to list of samples
        Each sample: (text, aspect, label)
        """
        samples = []
        
        for idx, row in self.data.iterrows():
            text = str(row[self.text_column]) if pd.notna(row[self.text_column]) else ""
            
            # Skip empty texts
            if not text.strip():
                continue
            
            # For each aspect
            for aspect in self.aspect_names:
                if pd.notna(row[aspect]):
                    label_str = str(row[aspect]).lower()
                    if label_str in self.label_map:
                        label = self.label_map[label_str]
                        samples.append({
                            'text': text,
                            'aspect': aspect,
                            'aspect_id': self.aspect_names.index(aspect),
                            'label': label,
                            'original_idx': idx
                        })
        
        return samples
    
    def _print_statistics(self):
        """Print dataset statistics"""
        aspect_counts = Counter([s['aspect'] for s in self.samples])
        label_counts = Counter([s['label'] for s in self.samples])
        
        print(f"\nAspect distribution:")
        for aspect, count in sorted(aspect_counts.items()):
            print(f"  {aspect}: {count}")
        
        print(f"\nLabel distribution:")
        label_names = {v: k for k, v in self.label_map.items()}
        for label_id, count in sorted(label_counts.items()):
            print(f"  {label_names[label_id]}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - input_ids: (max_length,)
                - attention_mask: (max_length,)
                - aspect_id: int
                - label: int
                - text: str (for explainability)
        """
        sample = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'aspect_id': torch.tensor(sample['aspect_id'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'text': sample['text'],
            'aspect': sample['aspect'],
            'review_id': sample['original_idx']
        }


class DependencyParsingDataset(CosmeticReviewDataset):
    """
    Extended dataset that includes dependency parsing information
    """
    def __init__(self, data_path, tokenizer, config, aspect_names, 
                 dependency_parser=None, is_train=True):
        """
        Args:
            dependency_parser: DependencyParser instance (optional)
        """
        super().__init__(data_path, tokenizer, config, aspect_names, is_train)
        self.dependency_parser = dependency_parser
        
        # Pre-compute dependency trees if parser is provided
        if self.dependency_parser is not None:
            print("Pre-computing dependency trees...")
            self.dependency_trees = self._compute_dependency_trees()
        else:
            self.dependency_trees = None
    
    def _compute_dependency_trees(self):
        """
        Pre-compute dependency trees for all unique texts
        Returns dictionary mapping text to (tokens, edge_index, edge_types)
        """
        unique_texts = list(set([s['text'] for s in self.samples]))
        dependency_trees = {}
        
        from tqdm import tqdm
        for text in tqdm(unique_texts, desc="Parsing dependencies"):
            try:
                tokens, edge_index, edge_types = self.dependency_parser.parse(text)
                dependency_trees[text] = {
                    'tokens': tokens,
                    'edge_index': edge_index,
                    'edge_types': edge_types
                }
            except Exception as e:
                print(f"Error parsing text: {e}")
                dependency_trees[text] = {
                    'tokens': [],
                    'edge_index': torch.zeros((2, 0), dtype=torch.long),
                    'edge_types': []
                }
        
        return dependency_trees
    
    def __getitem__(self, idx):
        """
        Returns sample with dependency information
        """
        item = super().__getitem__(idx)
        
        # Add dependency tree if available
        if self.dependency_trees is not None:
            text = item['text']
            if text in self.dependency_trees:
                dep_info = self.dependency_trees[text]
                edge_index = dep_info['edge_index']
                
                # CRITICAL FIX: Prune edges that are out of bounds
                # RoBERTa truncates inputs to self.max_length
                # SpaCy parses full text, so indices can exceed max_length
                if edge_index.size(1) > 0:
                    mask = (edge_index[0] < self.max_length) & (edge_index[1] < self.max_length)
                    edge_index = edge_index[:, mask]
                
                item['edge_index'] = edge_index
                item['tokens'] = dep_info['tokens']
                item['edge_types'] = dep_info['edge_types']
            else:
                item['edge_index'] = torch.zeros((2, 0), dtype=torch.long)
                item['tokens'] = []
                item['edge_types'] = []
        
        return item


def collate_fn_with_dependencies(batch):
    """
    Custom collate function for batches with dependency trees
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    aspect_ids = torch.stack([item['aspect_id'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    review_ids = [item['review_id'] for item in batch]
    
    # Collect texts and aspects (for explainability)
    texts = [item['text'] for item in batch]
    aspects = [item['aspect'] for item in batch]
    
    # Collect dependency information
    edge_indices = []
    tokens = []
    edge_types = []
    
    for item in batch:
        if 'edge_index' in item:
            edge_indices.append(item['edge_index'])
            tokens.append(item.get('tokens', []))
            edge_types.append(item.get('edge_types', []))
        else:
            edge_indices.append(None)
            tokens.append([])
            edge_types.append([])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'aspect_ids': aspect_ids,
        'labels': labels,
        'review_ids': review_ids,
        'edge_indices': edge_indices,
        'texts': texts,
        'aspects': aspects,
        'tokens': tokens,
        'edge_types': edge_types
    }


def create_dataloaders(config, tokenizer, dependency_parser=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
        tokenizer: RoBERTa tokenizer
        dependency_parser: Optional dependency parser
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config['data']
    training_config = config['training']
    aspect_names = config['aspects']['names']
    
    # Determine dataset class
    if data_config.get('use_dependency_parsing', False) and dependency_parser is not None:
        dataset_class = DependencyParsingDataset
        print("Using DependencyParsingDataset")
    else:
        dataset_class = CosmeticReviewDataset
        print("Using CosmeticReviewDataset (no dependency parsing)")
    
    # Build kwargs — only pass dependency_parser for DependencyParsingDataset
    extra_kwargs = {}
    if dataset_class is DependencyParsingDataset:
        extra_kwargs['dependency_parser'] = dependency_parser

    # Create datasets
    train_dataset = dataset_class(
        data_path=data_config['train_path'],
        tokenizer=tokenizer,
        config=config,
        aspect_names=aspect_names,
        is_train=True,
        **extra_kwargs
    )

    val_dataset = dataset_class(
        data_path=data_config['val_path'],
        tokenizer=tokenizer,
        config=config,
        aspect_names=aspect_names,
        is_train=False,
        **extra_kwargs
    )

    test_dataset = dataset_class(
        data_path=data_config['test_path'],
        tokenizer=tokenizer,
        config=config,
        aspect_names=aspect_names,
        is_train=False,
        **extra_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn_with_dependencies
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn_with_dependencies
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn_with_dependencies
    )
    
    return train_loader, val_loader, test_loader


class DependencyParser:
    """
    Wrapper for dependency parsing using spaCy
    """
    def __init__(self, language='en', model_name='en_core_web_sm'):
        """
        Args:
            language: Language code 
            model_name: spaCy model name
        """
        import spacy
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
        except:
            print(f"Model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model_name])
            self.nlp = spacy.load(model_name)
    
    def parse(self, text):
        """
        Parse text and extract dependency tree
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of tokens
            edge_index: (2, num_edges) tensor of dependency edges
            edge_types: List of dependency relation types
        """
        doc = self.nlp(text)
        
        tokens = [token.text for token in doc]
        edges = []
        edge_types = []
        
        for token in doc:
            if token.head != token:  # Not root
                # Add directed edge from head to dependent
                edges.append([token.head.i, token.i])
                edge_types.append(token.dep_)
        
        if len(edges) == 0:
            # Handle single-token sentences or parsing failures
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return tokens, edge_index, edge_types


def compute_class_weights(data_path, aspect_names, label_map):
    """
    Compute class weights for handling imbalance
    
    Returns:
        Dictionary mapping aspect names to class counts [neg, neu, pos]
    """
    df = pd.read_csv(data_path)
    aspect_class_counts = {}
    
    for aspect in aspect_names:
        counts = [0, 0, 0]  # neg, neu, pos
        
        for label_str, label_id in label_map.items():
            count = (df[aspect] == label_str).sum()
            counts[label_id] = count
        
        aspect_class_counts[aspect] = counts
    
    return aspect_class_counts


if __name__ == "__main__":
    # Test data loading
    import yaml
    from transformers import RobertaTokenizer
    
    print("Testing data loading...")
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data paths for testing
    config['data']['train_path'] = '../../train.csv'
    config['data']['val_path'] = '../../val.csv'
    config['data']['test_path'] = '../../test.csv'
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['roberta_model'])
    
    # Test without dependency parsing
    print("\n=== Testing without dependency parsing ===")
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Aspect IDs shape: {batch['aspect_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Number of texts: {len(batch['texts'])}")
    
    # Test with dependency parsing
    print("\n=== Testing with dependency parsing ===")
    config['data']['use_dependency_parsing'] = True
    parser = DependencyParser()
    
    # Test parser on a sample
    sample_text = "The lipstick color is beautiful but smells bad"
    tokens, edge_index, edge_types = parser.parse(sample_text)
    print(f"\nSample parsing:")
    print(f"Tokens: {tokens}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge types: {edge_types}")
    
    print("\nData loading test passed!")
