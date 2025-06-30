import sys
import os
import torch

from transformers import MllamaForConditionalGeneration
from transformers import AutoProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.compute_concepts_utils import compute_batch_embeddings, get_llama_text_patch_embeddings, get_llama_text_cls_embeddings
from utils.general_utils import load_text
from utils.gt_concept_segmentation_utils import compute_attention_masks, map_sentence_to_concept_gt


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAMES = ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']
SAMPLE_TYPES = ['patch', 'cls']
scratch_dir=''

DATASET_NAMES = ['iSarcasm']

#for simple text model
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
PROCESSOR = AutoProcessor.from_pretrained(model_id)
MODEL_NAME = 'Llama'
MODEL_INPUT_SIZE = ('text', 'text')

if __name__ == "__main__":
    for DATASET_NAME in DATASET_NAMES:
        print("Dataset", DATASET_NAME)
        all_text, train_text, test_text, cal_text = load_text(DATASET_NAME)
        
        #compute list of list of tokens
        compute_attention_masks(all_text, PROCESSOR, DATASET_NAME, MODEL_INPUT_SIZE)
        
        #compute gt
        map_sentence_to_concept_gt(DATASET_NAME, MODEL_INPUT_SIZE, one_indexed=True)
        
        #compute embeddings
        for SAMPLE_TYPE in SAMPLE_TYPES:
            print("SAMPLE TYPE", SAMPLE_TYPE)
            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            if SAMPLE_TYPE == 'patch':
                compute_batch_embeddings(all_text, get_llama_text_patch_embeddings, MODEL, PROCESSOR, 
                                          DEVICE, percent_thru_model=PERCENT_THRU_MODEL, 
                                          dataset_name=DATASET_NAME, model_input_size=MODEL_INPUT_SIZE,
                                          embeddings_file=EMBEDDINGS_FILE, batch_size=32)
            else:
                #cls
                compute_batch_embeddings(all_text, get_llama_text_cls_embeddings, MODEL, PROCESSOR, DEVICE, 
                                            percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME, 
                                            model_input_size=MODEL_INPUT_SIZE, embeddings_file=EMBEDDINGS_FILE, 
                                         batch_size=32)