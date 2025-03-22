import PIL
from vllm import LLM, SamplingParams

from .prompt_concepts import img2base64


def prompt_inversion(model: LLM, concept_name: str, input: PIL.Image.Image):
    # get bounding boxes for the concept
    prompt = f"Output the bounding box of where in the image the {concept_name} is located. Output just the bounding box in the format [x1, y1, x2, y2] where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner. If the {concept_name} is not present in the image, output 'None'."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img2base64(input)}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
    output = (
        model.chat(messages, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text
    )

    return output
