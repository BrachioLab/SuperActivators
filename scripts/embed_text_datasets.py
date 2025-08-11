import sys
import os
import torch

from transformers import MllamaForConditionalGeneration, AutoModel
from transformers import AutoProcessor, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.embedding_utils import compute_batch_embeddings
from utils.general_utils import load_text
from utils.gt_concept_segmentation_utils import compute_attention_masks, map_sentence_to_concept_gt


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAMES = ['Sarcasm', 'iSarcasm', 'GoEmotions']
scratch_dir='/scratch/cgoldberg/'


# Model configurations
MODEL_CONFIGS = [
    {
        'name': 'Llama',
        'model_id': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'model_class': MllamaForConditionalGeneration,
        'processor_class': AutoProcessor,
        'model_input_size': ('text', 'text')
    },
    {
        'name': 'Qwen',
        'model_id': 'Qwen/Qwen3-Embedding-4B',
        'model_class': AutoModel,
        'processor_class': AutoTokenizer,
        'model_input_size': ('text', 'text3')
    }
]

if __name__ == "__main__":
    for DATASET_NAME in DATASET_NAMES:
        print("Dataset:", DATASET_NAME)
        all_text, train_text, test_text, cal_text = load_text(DATASET_NAME)
        
        # Loop through both Llama and Mistral models
        for model_config in MODEL_CONFIGS:
            print(f"\nProcessing with {model_config['name']} model...")
            
            # Load model and processor
            if model_config['name'] == 'Qwen':
                # Load Qwen embedding model
                MODEL = model_config['model_class'].from_pretrained(
                    model_config['model_id'], 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                MODEL = model_config['model_class'].from_pretrained(
                    model_config['model_id'], 
                    torch_dtype=torch.float16, 
                    device_map="auto"
                )
            PROCESSOR = model_config['processor_class'].from_pretrained(model_config['model_id'])
            MODEL_NAME = model_config['name']
            
            # Compute list of list of tokens
            compute_attention_masks(all_text, PROCESSOR, DATASET_NAME, model_config['model_input_size'])
            
            # Compute gt
            map_sentence_to_concept_gt(DATASET_NAME, model_config['model_input_size'], one_indexed=True)
            
            # Compute both cls and patch embeddings in one pass
            print(f"Computing both CLS and patch embeddings for {MODEL_NAME}")
            compute_batch_embeddings(all_text, MODEL, PROCESSOR, DEVICE, 
                                    percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME, 
                                    model_input_size=model_config['model_input_size'],
                                    batch_size=32, scratch_dir=scratch_dir, model_name=MODEL_NAME)
            
            # Clear model from memory before loading next one
            del MODEL
            del PROCESSOR
            torch.cuda.empty_cache()