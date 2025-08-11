import sys
import os
import torch

from transformers import CLIPModel, AutoProcessor, MllamaForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import load_images
from utils.compute_concepts_utils import compute_avg_concept_vectors
from utils.activation_utils import compute_cosine_sims
from utils.embedding_utils import compute_batch_embeddings


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAMES = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
scratch_dir='/scratch/cgoldberg/' 


#for clip
clip_model_name = "openai/clip-vit-large-patch14"
CLIP_PROCESSOR = AutoProcessor.from_pretrained(clip_model_name)
CLIP_MODEL = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)
CLIP_MODEL.eval()
CLIP_INPUT_IMAGE_SIZE = (224, 224)

# #for llama
llama_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_MODEL = MllamaForConditionalGeneration.from_pretrained(llama_model_id, torch_dtype=torch.float16).to(DEVICE)
LLAMA_PROCESSOR = AutoProcessor.from_pretrained(llama_model_id)
LLAMA_MODEL.eval()
LLAMA_INPUT_IMAGE_SIZE = (560, 560)


if __name__ == "__main__":
    for DATASET_NAME in DATASET_NAMES:
        print("Dataset", DATASET_NAME)
        
        # Process CLIP model - compute both cls and patch embeddings in one pass
        print("\n=== Processing CLIP Model ===")
        clip_images, _, _ = load_images(dataset_name=DATASET_NAME, model_input_size=CLIP_INPUT_IMAGE_SIZE)
        
        print(f"Computing both CLS and patch embeddings for CLIP")
        compute_batch_embeddings(clip_images, CLIP_MODEL, CLIP_PROCESSOR, DEVICE, 
                                percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
                                model_input_size=CLIP_INPUT_IMAGE_SIZE,
                                batch_size=100, scratch_dir=scratch_dir)
        
        torch.cuda.empty_cache()            
        torch.cuda.ipc_collect()
        
        # Process Llama model - compute both cls and patch embeddings in one pass
        print("\n=== Processing Llama Model ===")
        llama_images, _, _ = load_images(dataset_name=DATASET_NAME, model_input_size=LLAMA_INPUT_IMAGE_SIZE)
        
        print(f"Computing both CLS and patch embeddings for Llama")
        compute_batch_embeddings(llama_images, LLAMA_MODEL, LLAMA_PROCESSOR, DEVICE, 
                                percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
                                model_input_size=LLAMA_INPUT_IMAGE_SIZE,
                                batch_size=16, scratch_dir=scratch_dir)  # Smaller batch for Llama
        
        torch.cuda.empty_cache()            
        torch.cuda.ipc_collect()