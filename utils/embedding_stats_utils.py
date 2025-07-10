"""
Utility functions for working with embedding statistics JSON.
"""

import json
import torch
import os
from typing import Tuple, Optional, Dict


class EmbeddingStatsManager:
    """Manager class for embedding statistics."""
    
    def __init__(self, stats_file: str = 'Embeddings/embedding_stats.json'):
        """
        Initialize the stats manager.
        
        Args:
            stats_file: Path to the JSON file containing embedding statistics
        """
        self.stats_file = stats_file
        self.stats = None
        self._load_stats()
    
    def _load_stats(self):
        """Load statistics from JSON file."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            print(f"Warning: Stats file not found: {self.stats_file}")
            self.stats = {}
    
    def get_normalization_stats(self, 
                               dataset_name: str,
                               model_name: str,
                               embedding_type: str = 'patch',
                               percent_thru_model: int = 100,
                               device: Optional[str] = None) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """
        Get normalization statistics for a specific configuration.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'CLEVR')
            model_name: Name of the model (e.g., 'Llama', 'CLIP')
            embedding_type: Type of embedding ('patch' or 'cls')
            percent_thru_model: Percentage through model (e.g., 100)
            device: Device to load tensors to
            
        Returns:
            Tuple of (mean_embedding, train_norm) or (None, None) if not found
        """
        try:
            # Navigate the nested structure
            stats = self.stats[dataset_name][model_name][embedding_type][str(percent_thru_model)]
            
            # Extract mean embedding
            if 'mean_embedding' in stats:
                mean_embedding = torch.tensor(stats['mean_embedding'])
                if device:
                    mean_embedding = mean_embedding.to(device)
            else:
                mean_embedding = None
            
            # Extract training norm
            train_norm = stats.get('train_norm', None)
            
            return mean_embedding, train_norm
            
        except KeyError as e:
            print(f"Stats not found for: {dataset_name}/{model_name}/{embedding_type}/{percent_thru_model}")
            return None, None
    
    def get_embedding_info(self,
                          dataset_name: str,
                          model_name: str,
                          embedding_type: str = 'patch',
                          percent_thru_model: int = 100) -> Dict:
        """
        Get general information about embeddings (dimensions, count, etc).
        
        Returns:
            Dictionary with embedding information
        """
        try:
            stats = self.stats[dataset_name][model_name][embedding_type][str(percent_thru_model)]
            
            info = {
                'embedding_dim': stats.get('embedding_dim'),
                'num_embeddings': stats.get('num_embeddings'),
                'file_size_mb': stats.get('file_size_mb'),
                'filename': stats.get('filename'),
                'has_mean_embedding': 'mean_embedding' in stats,
                'has_train_norm': 'train_norm' in stats
            }
            
            return info
            
        except KeyError:
            return {}
    
    def list_available_configs(self) -> Dict:
        """
        List all available dataset/model/type/percent configurations.
        
        Returns:
            Nested dictionary of available configurations
        """
        configs = {}
        
        for dataset in self.stats:
            configs[dataset] = {}
            for model in self.stats[dataset]:
                configs[dataset][model] = {}
                for emb_type in self.stats[dataset][model]:
                    configs[dataset][model][emb_type] = list(self.stats[dataset][model][emb_type].keys())
        
        return configs
    
    def print_summary(self):
        """Print a summary of all available statistics."""
        print("=== Embedding Statistics Summary ===")
        
        for dataset in sorted(self.stats.keys()):
            print(f"\nDataset: {dataset}")
            for model in sorted(self.stats[dataset].keys()):
                print(f"  Model: {model}")
                for emb_type in sorted(self.stats[dataset][model].keys()):
                    print(f"    Type: {emb_type}")
                    percents = sorted(self.stats[dataset][model][emb_type].keys(), 
                                    key=lambda x: int(x))
                    print(f"      Percentages: {', '.join(percents)}")


def load_normalization_stats(dataset_name: str,
                           model_name: str,
                           embedding_type: str = 'patch',
                           percent_thru_model: int = 100,
                           device: Optional[str] = None,
                           stats_file: str = 'Embeddings/embedding_stats.json') -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """
    Convenience function to load normalization statistics.
    
    This is a standalone function that doesn't require creating a manager instance.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        embedding_type: Type of embedding ('patch' or 'cls')
        percent_thru_model: Percentage through model
        device: Device to load tensors to
        stats_file: Path to stats JSON file
        
    Returns:
        Tuple of (mean_embedding, train_norm)
    """
    manager = EmbeddingStatsManager(stats_file)
    return manager.get_normalization_stats(
        dataset_name, model_name, embedding_type, percent_thru_model, device
    )


# Example usage
if __name__ == "__main__":
    # Create manager
    manager = EmbeddingStatsManager()
    
    # Print summary
    manager.print_summary()
    
    # Get specific stats
    mean_emb, norm = manager.get_normalization_stats(
        'CLEVR', 'Llama', 'patch', 100, device='cuda'
    )
    
    if mean_emb is not None:
        print(f"\nLoaded CLEVR/Llama stats:")
        print(f"  Mean embedding shape: {mean_emb.shape}")
        print(f"  Train norm: {norm}")
    
    # Get embedding info
    info = manager.get_embedding_info('CLEVR', 'Llama', 'patch', 100)
    print(f"\nEmbedding info: {info}")