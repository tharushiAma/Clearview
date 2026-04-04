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
from tqdm import tqdm


class CosmeticReviewDataset(Dataset):
    """
    Dataset for cosmetic product reviews with multi-aspect sentiment labels
    """
    def __init__(self, data_path, tokenizer, config, aspect_names, is_train=True):
        split = "train" if is_train else "val/test"
        print(f"\n[Dataset] Loading {split} data from: {data_path}")

        self.data = pd.read_csv(data_path)
        print(f"[Dataset] Rows in CSV: {len(self.data)}")

        self.tokenizer    = tokenizer
        self.config       = config
        self.aspect_names = aspect_names
        self.is_train     = is_train

        data_config      = config['data']
        self.max_length  = data_config['max_seq_length']
        self.text_column = data_config['text_column']
        self.label_map   = config['aspects']['label_map']

        print(f"[Dataset] Text column: '{self.text_column}'  |  max_seq_length: {self.max_length}")
        print(f"[Dataset] Aspects ({len(aspect_names)}): {aspect_names}")
        print(f"[Dataset] Label map: {self.label_map}")
        print(f"[Dataset] Building (text, aspect, label) samples...")

        self.samples = self._prepare_samples()

        print(f"[Dataset] Total samples built: {len(self.samples)}  "
              f"(avg {len(self.samples)/len(self.data):.1f} samples per row)")
        self._print_statistics()

    def _prepare_samples(self):
        samples: list = []
        skipped_empty: int = 0

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data),
                             desc="  Expanding rows"):
            text = str(row[self.text_column]) if pd.notna(row[self.text_column]) else ""
            if not text.strip():
                skipped_empty += 1  # type: ignore
                continue

            for aspect in self.aspect_names:
                if pd.notna(row[aspect]):
                    label_str = str(row[aspect]).lower()
                    if label_str in self.label_map:
                        samples.append({
                            'text'        : text,
                            'aspect'      : aspect,
                            'aspect_id'   : self.aspect_names.index(aspect),
                            'label'       : self.label_map[label_str],
                            'original_idx': idx,
                        })

        if skipped_empty:
            print(f"  [Dataset] Skipped {skipped_empty} rows with empty text")
        return samples

    def _print_statistics(self):
        aspect_counts = Counter([s['aspect'] for s in self.samples])
        label_names   = {v: k for k, v in self.label_map.items()}
        label_counts  = Counter([s['label']  for s in self.samples])

        print(f"[Dataset] Aspect distribution:")
        for aspect in self.aspect_names:
            print(f"  {aspect:<16}: {aspect_counts.get(aspect, 0)}")

        print(f"[Dataset] Label distribution:")
        for label_id in sorted(label_counts):
            print(f"  {label_names[label_id]:<10}: {label_counts[label_id]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'    : encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'aspect_id'    : torch.tensor(sample['aspect_id'], dtype=torch.long),
            'label'        : torch.tensor(sample['label'],     dtype=torch.long),
            'text'         : sample['text'],
            'aspect'       : sample['aspect'],
            'review_id'    : sample['original_idx'],
        }


class DependencyParsingDataset(CosmeticReviewDataset):
    """
    Extended dataset that includes dependency parsing information
    """
    def __init__(self, data_path, tokenizer, config, aspect_names,
                 dependency_parser=None, is_train=True):
        super().__init__(data_path, tokenizer, config, aspect_names, is_train)
        self.dependency_parser = dependency_parser

        if self.dependency_parser is not None:
            unique_texts = len(set(s['text'] for s in self.samples))
            print(f"[DepDataset] Pre-computing spaCy parse trees for {unique_texts} unique texts...")
            self.dependency_trees = self._compute_dependency_trees()
            print(f"[DepDataset] Parse trees ready ({len(self.dependency_trees)} unique texts)")
        else:
            print(f"[DepDataset] No dependency parser provided — edge_index will be empty")
            self.dependency_trees = None

    def _compute_dependency_trees(self):
        unique_texts = list(set(s['text'] for s in self.samples))
        trees: dict = {}
        parse_errors: list = []

        for text in tqdm(unique_texts, desc="  Parsing dependency trees"):
            try:
                toks, ei, et = self.dependency_parser.parse(text)
                trees[text]  = {'tokens': toks, 'edge_index': ei, 'edge_types': et}
            except Exception as e:
                parse_errors.append(str(e))
                trees[text] = {
                    'tokens'    : [],
                    'edge_index': torch.zeros((2, 0), dtype=torch.long),
                    'edge_types': [],
                }

        if parse_errors:
            print(f"  [DepDataset] WARNING: {len(parse_errors)} parse errors (set to empty edge_index)")
        return trees

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.dependency_trees is not None:
            dep_info              = self.dependency_trees.get(item['text'], {})
            edge_index: torch.Tensor = dep_info.get(  # type: ignore[assignment]
                'edge_index', torch.zeros((2, 0), dtype=torch.long)
            )

            if edge_index.size(1) > 0:
                mask       = (edge_index[0] < self.max_length) & (edge_index[1] < self.max_length)
                edge_index = edge_index[:, mask]

            item['edge_index'] = edge_index
            item['tokens']     = dep_info.get('tokens', [])
            item['edge_types'] = dep_info.get('edge_types', [])
        return item


def collate_fn_with_dependencies(batch):
    """Custom collate function for batches with dependency trees"""
    edge_indices, tokens, edge_types = [], [], []
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
        'input_ids'    : torch.stack([item['input_ids']     for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'aspect_ids'   : torch.stack([item['aspect_id']    for item in batch]),
        'labels'       : torch.stack([item['label']         for item in batch]),
        'review_ids'   : [item['review_id'] for item in batch],
        'texts'        : [item['text']      for item in batch],
        'aspects'      : [item['aspect']    for item in batch],
        'edge_indices' : edge_indices,
        'tokens'       : tokens,
        'edge_types'   : edge_types,
    }


def create_dataloaders(config, tokenizer, dependency_parser=None):
    """Create train / val / test DataLoaders from config."""
    data_config  = config['data']
    hw_config    = config['hardware']
    aspect_names = config['aspects']['names']

    use_dep = data_config.get('use_dependency_parsing', False) and dependency_parser is not None
    DatasetClass = DependencyParsingDataset if use_dep else CosmeticReviewDataset
    extra = {'dependency_parser': dependency_parser} if use_dep else {}

    print(f"\n[DataLoaders] Dataset class: {DatasetClass.__name__}")
    print(f"[DataLoaders] Batch size: {config['training']['batch_size']}  |  "
          f"num_workers: {hw_config['num_workers']}")

    splits = [
        ('train', data_config['train_path'], True),
        ('val',   data_config['val_path'],   False),
        ('test',  data_config['test_path'],  False),
    ]
    loaders = []
    for name, path, shuffle in splits:
        print(f"\n[DataLoaders] Building {name} loader from: {path}")
        ds = DatasetClass(path, tokenizer, config, aspect_names, is_train=shuffle, **extra)
        loader = DataLoader(
            ds,
            batch_size=config['training']['batch_size'],
            shuffle=shuffle,
            num_workers=hw_config['num_workers'],
            pin_memory=hw_config['pin_memory'],
            collate_fn=collate_fn_with_dependencies,
        )
        print(f"[DataLoaders] {name}: {len(ds)} samples -> {len(loader)} batches")
        loaders.append(loader)

    return tuple(loaders)


class DependencyParser:
    """Wrapper for dependency parsing using spaCy"""
    def __init__(self, language='en', model_name='en_core_web_sm'):
        import spacy
        print(f"[DepParser] Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
            print(f"[DepParser] Model loaded OK")
        except OSError:
            print(f"[DepParser] Model not found — downloading {model_name}...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model_name])
            self.nlp = spacy.load(model_name)
            print(f"[DepParser] Model downloaded and loaded OK")

    def parse(self, text):
        """
        Returns:
            tokens:     list of token strings
            edge_index: (2, num_edges) LongTensor  [head → dependent]
            edge_types: list of dependency relation strings (e.g. 'nsubj', 'dobj')
        """
        doc        = self.nlp(text)
        tokens     = [token.text for token in doc]
        edges      = []
        edge_types = []

        for token in doc:
            if token.head != token:
                edges.append([token.head.i, token.i])
                edge_types.append(token.dep_)

        edge_index = (torch.tensor(edges, dtype=torch.long).t()
                      if edges else torch.zeros((2, 0), dtype=torch.long))
        return tokens, edge_index, edge_types


def compute_class_weights(data_path, aspect_names, label_map):
    """
    Returns {aspect: [neg_count, neu_count, pos_count]} for HybridLoss init.
    """
    print(f"\n[ClassWeights] Computing class counts from: {data_path}")
    df = pd.read_csv(data_path)
    aspect_class_counts = {}

    for aspect in tqdm(aspect_names, desc="  Counting per aspect"):
        counts = [0, 0, 0]
        for label_str, label_id in label_map.items():
            counts[label_id] = int((df[aspect].astype(str) == label_str).sum())  # type: ignore
        aspect_class_counts[aspect] = counts

    print(f"[ClassWeights] Done. Summary (neg / neu / pos):")
    for asp, c in aspect_class_counts.items():
        ratio = f"  pos:neg={c[2]}:{c[0]}" if c[0] > 0 else "  (no negatives)"
        print(f"  {asp:<16}: neg={c[0]:>5}  neu={c[1]:>5}  pos={c[2]:>5}{ratio}")

    return aspect_class_counts


if __name__ == "__main__":
    import yaml
    print("Testing data loading...")
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['roberta_model'])
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)
    batch = next(iter(train_loader))
    print(f"Batch input_ids: {batch['input_ids'].shape}")
    print("Data loading test passed!")
