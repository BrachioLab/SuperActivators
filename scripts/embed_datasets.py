import sys
import os
import torch

from transformers import CLIPModel, AutoProcessor, MllamaForConditionalGeneration

sys.path.append(os.path.abspath("../utils"))
from general_utils import load_images
from compute_concepts_utils import get_clip_cls_embeddings, get_clip_patch_embeddings, compute_batch_embeddings, compute_avg_concept_vectors, compute_cosine_sims, get_llama_patch_embeddings, get_llama_cls_embeddings


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#for clip
# clip_model_name = "openai/clip-vit-large-patch14"
# CLIP_PROCESSOR = AutoProcessor.from_pretrained(clip_model_name)
# CLIP_MODEL = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)
# CLIP_MODEL.eval()
# CLIP_INPUT_IMAGE_SIZE = (224, 224)

#for llama
llama_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_MODEL = MllamaForConditionalGeneration.from_pretrained(llama_model_id, torch_dtype=torch.float16, device_map="auto")
LLAMA_PROCESSOR = AutoProcessor.from_pretrained(llama_model_id)
LLAMA_MODEL.eval()
LLAMA_INPUT_IMAGE_SIZE = (560, 560)

if __name__ == "__main__":
    for DATASET_NAME in ['Coco']:
        all_images, train_images, test_images = load_images(dataset_name=DATASET_NAME, model_input_size=LLAMA_INPUT_IMAGE_SIZE)
        
        ######## CLS
        ### CLIP
        # EMBEDDINGS_FILE = f'CLIP_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
        # embeds_dic = compute_batch_embeddings(all_images, get_clip_cls_embeddings, CLIP_MODEL, CLIP_PROCESSOR, DEVICE, 
        #                                 percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
        #                                 embeddings_file=EMBEDDINGS_FILE, batch_size=100)
        
        ### Llama
        # EMBEDDINGS_FILE = f'Llama_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
        # embeds_dic = compute_batch_embeddings(all_images, get_llama_cls_embeddings, LLAMA_MODEL, LLAMA_PROCESSOR, DEVICE, 
        #                                 percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
        #                                 embeddings_file=EMBEDDINGS_FILE, batch_size=2)
        
        
#         ######## Patch
#         ### CLIP
        # EMBEDDINGS_FILE = f'CLIP_patch_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
        # embeds_dic = compute_batch_embeddings(all_images, get_clip_patch_embeddings, CLIP_MODEL, CLIP_PROCESSOR, DEVICE, 
        #                                 percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
        #                                 embeddings_file=EMBEDDINGS_FILE, batch_size=100)
        
        ### Llama
        EMBEDDINGS_FILE = f'Llama_patch_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
        embeds_dic = compute_batch_embeddings(all_images, get_llama_patch_embeddings, LLAMA_MODEL, LLAMA_PROCESSOR, DEVICE, 
                                        percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
                                        embeddings_file=EMBEDDINGS_FILE, batch_size=2)