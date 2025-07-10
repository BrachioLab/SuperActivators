"""
Memory Management Utilities for Processing Chunked Embeddings

This module provides utilities for efficiently processing large embedding files that 
have been split into chunks to reduce memory usage.

Key Features:
- Auto-detection of chunked vs non-chunked embeddings
- Incremental loading and processing of chunks
- Memory-efficient aggregation of results
- Index mapping between global and chunk-local indices
- Unified interface for both chunked and non-chunked embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import gc
from typing import Dict, List, Tuple, Optional, Union, Iterator, Callable, Any
from contextlib import contextmanager
from tqdm import tqdm

class ChunkedEmbeddingLoader:
    """
    Loader for chunked embedding files that provides memory-efficient access.
    
    Supports both chunked files (split into multiple .pt files) and regular files.
    Provides iterator interface for processing chunks incrementally.
    """
    
    def __init__(self, embeddings_path: str, device: str = 'cpu'):
        """
        Initialize the chunked embedding loader.
        
        Args:
            embeddings_path: Path to embedding file (with or without chunks)
            device: Device to load embeddings on
        """
        self.embeddings_path = embeddings_path
        self.device = device
        self.is_chunked = False
        self.chunk_info = None
        self.total_samples = 0
        self.embedding_dim = 0
        self.chunks_dir = os.path.dirname(embeddings_path)
        
        # Check if embeddings are chunked
        self._detect_chunked_embeddings()
        
    def _detect_chunked_embeddings(self):
        """Detect if embeddings are split into chunks."""
        # Look for chunk info file
        base_name = os.path.splitext(self.embeddings_path)[0]
        chunk_info_file = f"{base_name}_chunks_info.json"
        
        if os.path.exists(chunk_info_file):
            # Load chunk information
            with open(chunk_info_file, 'r') as f:
                self.chunk_info = json.load(f)
            
            self.is_chunked = True
            self.total_samples = self.chunk_info['total_samples']
            self.embedding_dim = self.chunk_info['embedding_dim']
            
            print(f"   📦 Detected chunked embeddings: {self.chunk_info['num_chunks']} chunks")
            print(f"   📊 Total samples: {self.total_samples:,}, Embedding dim: {self.embedding_dim}")
            
        elif os.path.exists(self.embeddings_path):
            # Regular single file
            self.is_chunked = False
            print(f"   📄 Using single embedding file: {os.path.basename(self.embeddings_path)}")
            
        else:
            raise FileNotFoundError(f"Embedding file not found: {self.embeddings_path}")
    
    def global_to_chunk_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global embedding index to (chunk_number, local_index_in_chunk).
        
        Args:
            global_idx: Index in the full embedding tensor
            
        Returns:
            Tuple of (chunk_number, local_index_in_chunk)
            
        Raises:
            ValueError: If global_idx is out of bounds
            RuntimeError: If embeddings are not chunked
        """
        if not self.is_chunked:
            # For non-chunked embeddings, everything is in chunk 0
            return (0, global_idx)
        
        if global_idx < 0 or global_idx >= self.total_samples:
            raise ValueError(f"Global index {global_idx} out of bounds [0, {self.total_samples})")
        
        # Find which chunk contains this global index
        for chunk_num, chunk_data in enumerate(self.chunk_info['chunks']):
            start_idx = chunk_data['start_idx']
            end_idx = chunk_data['end_idx']
            
            if start_idx <= global_idx < end_idx:
                local_idx = global_idx - start_idx
                return (chunk_num, local_idx)
        
        raise RuntimeError(f"Could not find chunk for global index {global_idx}")
    
    def chunk_to_global_index(self, chunk_num: int, local_idx: int) -> int:
        """
        Convert (chunk_number, local_index_in_chunk) to global embedding index.
        
        Args:
            chunk_num: Chunk number
            local_idx: Index within the chunk
            
        Returns:
            Global index in the full embedding tensor
            
        Raises:
            ValueError: If chunk_num or local_idx is out of bounds
            RuntimeError: If embeddings are not chunked
        """
        if not self.is_chunked:
            # For non-chunked embeddings, local_idx is the global index
            return local_idx
        
        if chunk_num < 0 or chunk_num >= self.chunk_info['num_chunks']:
            raise ValueError(f"Chunk number {chunk_num} out of bounds [0, {self.chunk_info['num_chunks']})")
        
        chunk_data = self.chunk_info['chunks'][chunk_num]
        start_idx = chunk_data['start_idx']
        end_idx = chunk_data['end_idx']
        chunk_size = end_idx - start_idx
        
        if local_idx < 0 or local_idx >= chunk_size:
            raise ValueError(f"Local index {local_idx} out of bounds [0, {chunk_size}) for chunk {chunk_num}")
        
        return start_idx + local_idx
    
    def global_indices_to_chunk_map(self, global_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        """
        Convert a list of global indices to a mapping of chunk_num -> [(global_idx, local_idx), ...].
        
        This is useful for efficiently processing multiple indices by grouping them by chunk.
        
        Args:
            global_indices: List of global indices
            
        Returns:
            Dictionary mapping chunk numbers to lists of (global_idx, local_idx) tuples
        """
        chunk_map = {}
        
        for global_idx in global_indices:
            chunk_num, local_idx = self.global_to_chunk_index(global_idx)
            
            if chunk_num not in chunk_map:
                chunk_map[chunk_num] = []
            
            chunk_map[chunk_num].append((global_idx, local_idx))
        
        return chunk_map
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embeddings."""
        if self.is_chunked:
            return {
                'is_chunked': True,
                'num_chunks': self.chunk_info['num_chunks'],
                'total_samples': self.total_samples,
                'embedding_dim': self.embedding_dim,
                'chunk_info': self.chunk_info
            }
        else:
            # Load info from single file
            data = torch.load(self.embeddings_path, map_location='cpu')
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            info = {
                'is_chunked': False,
                'total_samples': embeddings.shape[0],
                'embedding_dim': embeddings.shape[1],
                'data_keys': list(data.keys())
            }
            
            # Clean up
            del data, embeddings
            gc.collect()
            
            return info
    
    def load_specific_embeddings(self, global_indices: List[int]) -> torch.Tensor:
        """
        Load specific embeddings by their global indices.
        Memory-efficient: only loads the necessary chunks.
        
        Args:
            global_indices: List of global indices to load
            
        Returns:
            Tensor containing the requested embeddings in the same order as global_indices
        """
        if not self.is_chunked:
            # For non-chunked embeddings, load all and index
            data = torch.load(self.embeddings_path, map_location=self.device)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            selected_embeddings = embeddings[global_indices]
            
            # Clean up
            del data, embeddings
            gc.collect()
            
            return selected_embeddings
        
        # For chunked embeddings, load only necessary chunks
        chunk_map = self.global_indices_to_chunk_map(global_indices)
        
        # Create result tensor
        result_embeddings = torch.empty(len(global_indices), self.embedding_dim, 
                                      device=self.device, dtype=torch.float32)
        
        # Create mapping from global_idx to result position
        global_to_result_pos = {global_idx: i for i, global_idx in enumerate(global_indices)}
        
        # Load embeddings from each required chunk
        for chunk_num, chunk_indices in chunk_map.items():
            chunk_path = os.path.join(self.chunks_dir, self.chunk_info['chunks'][chunk_num]['file'])
            chunk_data = torch.load(chunk_path, map_location=self.device)
            
            if 'normalized_embeddings' in chunk_data:
                chunk_embeddings = chunk_data['normalized_embeddings']
            else:
                chunk_embeddings = chunk_data['embeddings']
            
            # Extract required embeddings from this chunk
            for global_idx, local_idx in chunk_indices:
                result_pos = global_to_result_pos[global_idx]
                result_embeddings[result_pos] = chunk_embeddings[local_idx]
            
            # Clean up chunk
            del chunk_data, chunk_embeddings
            gc.collect()
        
        return result_embeddings
    
    def load_full_embeddings(self) -> torch.Tensor:
        """
        Load all embeddings into memory at once.
        WARNING: Use only for small files or when you have enough memory.
        """
        if not self.is_chunked:
            # Load single file
            data = torch.load(self.embeddings_path, map_location=self.device)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            return embeddings
        else:
            # Load and concatenate all chunks
            print(f"   ⚠️  Loading all chunks into memory ({self.chunk_info['num_chunks']} chunks)")
            
            all_embeddings = []
            for i in range(self.chunk_info['num_chunks']):
                chunk_path = os.path.join(self.chunks_dir, self.chunk_info['chunks'][i]['file'])
                chunk_data = torch.load(chunk_path, map_location=self.device)
                
                if 'normalized_embeddings' in chunk_data:
                    chunk_embeddings = chunk_data['normalized_embeddings']
                else:
                    chunk_embeddings = chunk_data['embeddings']
                
                all_embeddings.append(chunk_embeddings)
                
                # Clean up
                del chunk_data
                gc.collect()
            
            return torch.cat(all_embeddings, dim=0)
    
    def iter_chunks(self, chunk_size: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """
        Iterate over embedding chunks.
        
        Args:
            chunk_size: Size of chunks to yield (only used for non-chunked files)
            
        Yields:
            Tuple of (embeddings_chunk, start_idx, end_idx)
        """
        if self.is_chunked:
            # Iterate over existing chunks
            for i, chunk_info in enumerate(self.chunk_info['chunks']):
                chunk_path = os.path.join(self.chunks_dir, chunk_info['file'])
                chunk_data = torch.load(chunk_path, map_location=self.device)
                
                if 'normalized_embeddings' in chunk_data:
                    chunk_embeddings = chunk_data['normalized_embeddings']
                else:
                    chunk_embeddings = chunk_data['embeddings']
                
                start_idx = chunk_info['start_idx']
                end_idx = chunk_info['end_idx']
                
                yield chunk_embeddings, start_idx, end_idx
                
                # Clean up
                del chunk_data, chunk_embeddings
                gc.collect()
        else:
            # Split single file into chunks on-the-fly
            data = torch.load(self.embeddings_path, map_location='cpu')
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            if chunk_size is None:
                # Return entire file as single chunk
                chunk_embeddings = embeddings.to(self.device)
                yield chunk_embeddings, 0, embeddings.shape[0]
            else:
                # Split into chunks of specified size
                total_samples = embeddings.shape[0]
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_embeddings = embeddings[start_idx:end_idx].to(self.device)
                    yield chunk_embeddings, start_idx, end_idx
                    
                    # Clean up
                    del chunk_embeddings
                    gc.collect()
            
            # Clean up full embeddings
            del data, embeddings
            gc.collect()

def process_embeddings_chunked(
    embeddings_path: str,
    processing_func: Callable[[torch.Tensor, int, int], Any],
    aggregation_func: Callable[[List[Any]], Any],
    device: str = 'cuda',
    chunk_size: Optional[int] = None,
    show_progress: bool = True
) -> Any:
    """
    Process embeddings in chunks and aggregate results.
    
    Args:
        embeddings_path: Path to embedding file
        processing_func: Function to process each chunk (chunk, start_idx, end_idx) -> result
        aggregation_func: Function to aggregate all results from chunks
        device: Device to use for processing
        chunk_size: Chunk size for non-chunked files
        show_progress: Whether to show progress bar
        
    Returns:
        Aggregated result from all chunks
    """
    loader = ChunkedEmbeddingLoader(embeddings_path, device)
    results = []
    
    chunk_iter = loader.iter_chunks(chunk_size)
    if show_progress:
        if loader.is_chunked:
            chunk_iter = tqdm(chunk_iter, total=loader.chunk_info['num_chunks'], 
                            desc="Processing chunks")
        else:
            chunk_iter = tqdm(chunk_iter, desc="Processing chunks")
    
    for chunk_embeddings, start_idx, end_idx in chunk_iter:
        result = processing_func(chunk_embeddings, start_idx, end_idx)
        results.append(result)
        
        # Force memory cleanup
        del chunk_embeddings
        gc.collect()
        
        # Force GPU memory cleanup if using CUDA
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    return aggregation_func(results)

def compute_activations_chunked(
    embeddings_path: str,
    concept_vectors: Dict[str, torch.Tensor],
    sample_ranges: List[Tuple[int, int]],
    device: str = 'cuda',
    method: str = 'avg',
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute activations between embeddings and concept vectors using chunked processing.
    
    Args:
        embeddings_path: Path to embedding file
        concept_vectors: Dictionary mapping concept names to vectors
        sample_ranges: List of (start_idx, end_idx) for each sample
        device: Device for computation
        method: 'avg' for cosine similarity, 'linsep' for signed distances
        show_progress: Whether to show progress
        
    Returns:
        Dictionary mapping concept names to activation tensors
    """
    
    def process_chunk(chunk_embeddings, chunk_start_idx, chunk_end_idx):
        """Process a single chunk of embeddings."""
        chunk_activations = {}
        
        for concept_name, concept_vector in concept_vectors.items():
            concept_vector = concept_vector.to(device)
            
            if method == 'linsep':
                # Signed distances for linear separator
                activations = torch.matmul(chunk_embeddings, concept_vector)
            else:
                # Cosine similarities for average concepts
                activations = F.cosine_similarity(
                    chunk_embeddings,
                    concept_vector.unsqueeze(0).expand_as(chunk_embeddings),
                    dim=1
                )
            
            chunk_activations[concept_name] = activations.cpu()  # Move to CPU to save GPU memory
        
        return chunk_activations, chunk_start_idx, chunk_end_idx
    
    def aggregate_results(results):
        """Aggregate activation results from all chunks."""
        all_activations = {}
        
        # Initialize activation lists for each concept
        for concept_name in concept_vectors.keys():
            all_activations[concept_name] = []
        
        # Sort results by start index to maintain order
        results.sort(key=lambda x: x[1])
        
        # Concatenate activations for each concept
        for chunk_activations, _, _ in results:
            for concept_name, activations in chunk_activations.items():
                all_activations[concept_name].append(activations)
        
        # Concatenate tensors for each concept
        final_activations = {}
        for concept_name, activation_list in all_activations.items():
            final_activations[concept_name] = torch.cat(activation_list, dim=0).to(device)
        
        return final_activations
    
    return process_embeddings_chunked(
        embeddings_path=embeddings_path,
        processing_func=process_chunk,
        aggregation_func=aggregate_results,
        device=device,
        show_progress=show_progress
    )

def compute_hybrid_activations_chunked(
    embeddings_path: str,
    hybrid_concepts: Dict[str, List[torch.Tensor]],
    sample_ranges: List[Tuple[int, int]],
    device: str = 'cuda',
    method: str = 'avg',
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute activations between embeddings and per-sample hybrid concept vectors using chunked processing.
    
    Args:
        embeddings_path: Path to embedding file
        hybrid_concepts: Dictionary mapping concept names to lists of hybrid vectors (one per sample)
        sample_ranges: List of (start_idx, end_idx) for each sample
        device: Device for computation
        method: 'avg' for cosine similarity, 'linsep' for signed distances
        show_progress: Whether to show progress
        
    Returns:
        Dictionary mapping concept names to activation tensors
    """
    
    def process_chunk(chunk_embeddings, chunk_start_idx, chunk_end_idx):
        """Process a single chunk of embeddings, respecting sample boundaries."""
        # CRITICAL: We need to maintain the exact same order as the original
        # by processing samples in order and only including the parts that fall within this chunk
        
        chunk_activations = {}
        
        for concept_name, concept_hybrid_vectors in hybrid_concepts.items():
            concept_activation_parts = []
            
            # Process samples in order to maintain global indexing
            for sample_idx, (sample_start, sample_end) in enumerate(sample_ranges):
                # Check if sample overlaps with this chunk
                overlap_start = max(sample_start, chunk_start_idx)
                overlap_end = min(sample_end, chunk_end_idx)
                
                if overlap_start < overlap_end:
                    # Get embeddings for this sample within the chunk
                    # These indices are relative to the chunk
                    sample_chunk_start = overlap_start - chunk_start_idx
                    sample_chunk_end = overlap_end - chunk_start_idx
                    sample_embeddings = chunk_embeddings[sample_chunk_start:sample_chunk_end]
                    
                    # Get hybrid vector for this sample
                    sample_hybrid_vector = concept_hybrid_vectors[sample_idx].to(device)
                    
                    if method == 'linsep':
                        # Signed distances for linear separator
                        sample_activations = torch.matmul(sample_embeddings, sample_hybrid_vector)
                    else:
                        # Cosine similarities for average concepts
                        sample_activations = F.cosine_similarity(
                            sample_embeddings,
                            sample_hybrid_vector.unsqueeze(0).expand_as(sample_embeddings),
                            dim=1
                        )
                    
                    concept_activation_parts.append(sample_activations.cpu())
            
            # Concatenate all activation parts for this concept in this chunk
            # This maintains the global ordering because we processed samples in order
            if concept_activation_parts:
                chunk_activations[concept_name] = torch.cat(concept_activation_parts, dim=0)
            else:
                # No activations for this concept in this chunk
                chunk_activations[concept_name] = torch.empty(0, device='cpu')
        
        return chunk_activations, chunk_start_idx, chunk_end_idx
    
    def aggregate_results(results):
        """Aggregate activation results from all chunks."""
        all_activations = {}
        
        # Initialize activation lists for each concept
        for concept_name in hybrid_concepts.keys():
            all_activations[concept_name] = []
        
        # Sort results by start index to maintain order
        results.sort(key=lambda x: x[1])
        
        # Concatenate activations for each concept
        for chunk_activations, _, _ in results:
            for concept_name, activations in chunk_activations.items():
                if len(activations) > 0:  # Only add non-empty tensors
                    all_activations[concept_name].append(activations)
        
        # Concatenate tensors for each concept
        final_activations = {}
        for concept_name, activation_list in all_activations.items():
            if activation_list:
                final_activations[concept_name] = torch.cat(activation_list, dim=0).to(device)
            else:
                final_activations[concept_name] = torch.empty(0, device=device)
        
        return final_activations
    
    return process_embeddings_chunked(
        embeddings_path=embeddings_path,
        processing_func=process_chunk,
        aggregation_func=aggregate_results,
        device=device,
        show_progress=show_progress
    )

def load_embeddings_by_indices(embeddings_path: str, global_indices: List[int], device: str = 'cuda') -> torch.Tensor:
    """
    Convenience function to load specific embeddings by their global indices.
    
    Args:
        embeddings_path: Path to embedding file
        global_indices: List of global indices to load
        device: Device to load embeddings on
        
    Returns:
        Tensor containing the requested embeddings
    """
    loader = ChunkedEmbeddingLoader(embeddings_path, device)
    return loader.load_specific_embeddings(global_indices)

def convert_global_to_chunk_indices(embeddings_path: str, global_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Convenience function to convert global indices to chunk mapping.
    
    Args:
        embeddings_path: Path to embedding file
        global_indices: List of global indices
        
    Returns:
        Dictionary mapping chunk numbers to lists of (global_idx, local_idx) tuples
    """
    loader = ChunkedEmbeddingLoader(embeddings_path, 'cpu')
    return loader.global_indices_to_chunk_map(global_indices)

@contextmanager
def memory_efficient_context(device: str = 'cuda'):
    """
    Context manager for memory-efficient processing.
    Automatically cleans up memory at the end.
    """
    try:
        yield
    finally:
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

def estimate_memory_usage(embedding_shape: Tuple[int, int], dtype_size: int = 4) -> float:
    """
    Estimate memory usage in GB for embeddings.
    
    Args:
        embedding_shape: (num_samples, embedding_dim)
        dtype_size: Size of data type in bytes (4 for float32)
        
    Returns:
        Estimated memory usage in GB
    """
    total_elements = embedding_shape[0] * embedding_shape[1]
    total_bytes = total_elements * dtype_size
    return total_bytes / (1024 ** 3)

def get_optimal_chunk_size(embedding_shape: Tuple[int, int], max_memory_gb: float = 8.0, dtype_size: int = 4) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Args:
        embedding_shape: (num_samples, embedding_dim)
        max_memory_gb: Maximum memory to use in GB
        dtype_size: Size of data type in bytes
        
    Returns:
        Optimal chunk size (number of samples)
    """
    total_samples, embedding_dim = embedding_shape
    
    # Calculate how many samples fit in the memory limit
    max_bytes = max_memory_gb * (1024 ** 3)
    bytes_per_sample = embedding_dim * dtype_size
    
    chunk_size = int(max_bytes / bytes_per_sample)
    
    # Ensure chunk size is at least 1 and not more than total samples
    chunk_size = max(1, min(chunk_size, total_samples))
    
    return chunk_size

def check_chunked_embeddings_status(embeddings_path: str) -> Dict[str, Any]:
    """
    Check the status of chunked embeddings and provide diagnostics.
    
    Args:
        embeddings_path: Path to embedding file
        
    Returns:
        Dictionary with status information
    """
    base_name = os.path.splitext(embeddings_path)[0]
    chunk_info_file = f"{base_name}_chunks_info.json"
    
    status = {
        'original_file_exists': os.path.exists(embeddings_path),
        'chunk_info_exists': os.path.exists(chunk_info_file),
        'is_chunked': False,
        'all_chunks_exist': False,
        'missing_chunks': [],
        'chunk_count': 0,
        'total_size_gb': 0.0
    }
    
    if status['chunk_info_exists']:
        with open(chunk_info_file, 'r') as f:
            chunk_info = json.load(f)
        
        status['is_chunked'] = True
        status['chunk_count'] = chunk_info['num_chunks']
        
        # Check if all chunks exist
        chunks_dir = os.path.dirname(embeddings_path)
        missing_chunks = []
        total_size = 0
        
        for i, chunk_data in enumerate(chunk_info['chunks']):
            chunk_path = os.path.join(chunks_dir, chunk_data['file'])
            if os.path.exists(chunk_path):
                total_size += os.path.getsize(chunk_path)
            else:
                missing_chunks.append(i)
        
        status['missing_chunks'] = missing_chunks
        status['all_chunks_exist'] = len(missing_chunks) == 0
        status['total_size_gb'] = total_size / (1024 ** 3)
    
    elif status['original_file_exists']:
        # Get size of original file
        file_size = os.path.getsize(embeddings_path)
        status['total_size_gb'] = file_size / (1024 ** 3)
    
    return status

def print_chunked_embedding_summary(embeddings_path: str):
    """Print a summary of chunked embedding status."""
    status = check_chunked_embeddings_status(embeddings_path)
    
    print(f"📊 Embedding Status: {os.path.basename(embeddings_path)}")
    print(f"   Original file exists: {'✅' if status['original_file_exists'] else '❌'}")
    print(f"   Is chunked: {'✅' if status['is_chunked'] else '❌'}")
    
    if status['is_chunked']:
        print(f"   Chunk count: {status['chunk_count']}")
        print(f"   All chunks exist: {'✅' if status['all_chunks_exist'] else '❌'}")
        if status['missing_chunks']:
            print(f"   Missing chunks: {status['missing_chunks']}")
    
    print(f"   Total size: {status['total_size_gb']:.2f} GB")


class ChunkedActivationLoader:
    """
    Loader for chunked activation CSV files (cosine similarities or distances).
    Provides memory-efficient access to activation metrics without loading full files.
    """
    
    def __init__(self, dataset_name: str, acts_file: str, scratch_dir: str = ''):
        """
        Initialize the chunked activation loader.
        
        Args:
            dataset_name: Name of dataset (e.g., 'CLEVR')
            acts_file: Name of activation file (e.g., 'cosine_similarities_avg_concepts_Llama_patch_embeddings_percentthrumodel_100.csv')
            scratch_dir: Optional scratch directory prefix
        """
        self.dataset_name = dataset_name
        self.acts_file = acts_file
        self.scratch_dir = scratch_dir
        self.is_chunked = False
        self.chunk_files = []
        self.total_samples = 0
        self.columns = []
        
        # Detect folder type from filename
        if 'dists_' in acts_file or 'linsep' in acts_file:
            self.folder = "Distances"
        else:
            self.folder = "Cosine_Similarities"
        
        self.base_path = f"{scratch_dir}{self.folder}/{dataset_name}"
        self.full_file_path = f"{self.base_path}/{acts_file}"
        
        # Detect chunked files
        self._detect_chunked_files()
        
    def _detect_chunked_files(self):
        """Detect if activation files are split into chunks."""
        # Look for chunk files
        base_name = self.acts_file.replace('.csv', '')
        chunk_0_file = f"{self.base_path}/{base_name}_chunk_0.csv"
        
        if os.path.exists(chunk_0_file):
            self.is_chunked = True
            
            # Find all chunk files
            chunk_idx = 0
            while True:
                chunk_file = f"{self.base_path}/{base_name}_chunk_{chunk_idx}.csv"
                if not os.path.exists(chunk_file):
                    break
                self.chunk_files.append(chunk_file)
                chunk_idx += 1
            
            # Get total samples and columns from all chunks
            total_samples = 0
            for chunk_file in self.chunk_files:
                chunk_df = pd.read_csv(chunk_file, index_col=0, nrows=0)  # Just get columns
                if not self.columns:
                    self.columns = list(chunk_df.columns)
                
                # Count rows without loading data
                with open(chunk_file, 'r') as f:
                    chunk_samples = sum(1 for line in f) - 1  # Subtract header
                total_samples += chunk_samples
            
            self.total_samples = total_samples
            print(f"   📦 Detected chunked activation file: {len(self.chunk_files)} chunks")
            print(f"   📊 Total samples: {self.total_samples:,}, Concepts: {len(self.columns)}")
            
        elif os.path.exists(self.full_file_path):
            self.is_chunked = False
            print(f"   📄 Using single activation file: {os.path.basename(self.full_file_path)}")
            
            # Get info from single file
            df_info = pd.read_csv(self.full_file_path, index_col=0, nrows=0)
            self.columns = list(df_info.columns)
            
            # Count rows
            with open(self.full_file_path, 'r') as f:
                self.total_samples = sum(1 for line in f) - 1
            
        else:
            raise FileNotFoundError(f"Activation file not found: {self.full_file_path}")
    
    def load_chunk_range(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Load activation data for a specific range of indices.
        
        Args:
            start_idx: Starting global index
            end_idx: Ending global index (exclusive)
            
        Returns:
            DataFrame with activation data for the specified range
        """
        if not self.is_chunked:
            # Load from single file
            df = pd.read_csv(self.full_file_path, index_col=0, skiprows=range(1, start_idx + 1), nrows=end_idx - start_idx)
            return df
        
        # Load from chunked files
        chunks_to_load = []
        current_offset = 0
        
        for chunk_file in self.chunk_files:
            # Count rows in this chunk
            with open(chunk_file, 'r') as f:
                chunk_size = sum(1 for line in f) - 1
            
            chunk_start = current_offset
            chunk_end = current_offset + chunk_size
            
            # Check if this chunk overlaps with our desired range
            if chunk_start < end_idx and chunk_end > start_idx:
                # Calculate which rows to load from this chunk
                chunk_skip_start = max(0, start_idx - chunk_start)
                chunk_skip_end = max(0, chunk_end - end_idx)
                chunk_rows_to_load = chunk_size - chunk_skip_start - chunk_skip_end
                
                if chunk_rows_to_load > 0:
                    if chunk_skip_start > 0:
                        skiprows = list(range(1, chunk_skip_start + 1))
                    else:
                        skiprows = None
                    
                    chunk_df = pd.read_csv(chunk_file, index_col=0, skiprows=skiprows, nrows=chunk_rows_to_load)
                    chunks_to_load.append(chunk_df)
            
            current_offset = chunk_end
        
        if chunks_to_load:
            # Concatenate all loaded chunks
            result_df = pd.concat(chunks_to_load, ignore_index=True)
            # Clear chunks from memory
            del chunks_to_load
            gc.collect()
            return result_df
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=self.columns)
    
    def load_full_dataframe(self) -> pd.DataFrame:
        """
        Load the complete activation dataframe.
        WARNING: This may use a lot of memory for large files.
        """
        if not self.is_chunked:
            return pd.read_csv(self.full_file_path, index_col=0)
        
        # Load and concatenate all chunks
        chunks = []
        for chunk_file in self.chunk_files:
            chunk_df = pd.read_csv(chunk_file, index_col=0)
            chunks.append(chunk_df)
        
        full_df = pd.concat(chunks, ignore_index=True)
        # Clear chunks from memory
        del chunks
        gc.collect()
        return full_df
    
    def get_activation_info(self) -> Dict[str, Any]:
        """Get information about the activation data."""
        return {
            'total_samples': self.total_samples,
            'num_concepts': len(self.columns),
            'concept_names': self.columns,
            'is_chunked': self.is_chunked,
            'num_chunks': len(self.chunk_files) if self.is_chunked else 1,
            'file_type': 'distances' if 'dists' in self.acts_file else 'cosine_similarities'
        }