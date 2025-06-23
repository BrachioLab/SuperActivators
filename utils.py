"""General utils"""
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

def retrieve_image(img_idx, dataset_name, test_only=False):
    """
    Retrieves an image from the specified dataset based on the given index.

    Args:
        img_idx (int): Index of the image in the dataset's metadata.
        dataset_name (str): Name of the dataset (default is 'CLEVR').

    Returns:
        PIL.Image: The image corresponding to the specified index.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    if test_only:
        metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    image_path = os.path.join(f'../Data/{dataset_name}', metadata.iloc[img_idx]['image_path'])
    image = Image.open(image_path).convert("RGB")
    return image

def load_images(dataset_name='CLEVR'):
    """
    Load images from a dataset.

    Args:
        dataset_name (str): The name of the dataset. Defaults to 'CLEVR'.

    Returns:
        list: A list of PIL.Image objects.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    image_paths = metadata['image_path'].tolist()
    splits = metadata['split'].tolist()

    print("Loading images...")
    all_images, train_images, test_images = [], [], []
    for idx, info in tqdm(metadata.iterrows()):
        image_filename = info['image_path']
        image = Image.open(f'../Data/{dataset_name}/{image_filename}').convert("RGB")
        all_images.append(image)
        
        split = info['split']
        if split == "train":
            train_images.append(image)
        else:
            test_images.append(image)
    print(f"Loaded {len(all_images)} images.")

    return all_images, train_images, test_images


def retrieve_topn_images(dataset_name, top_n, start_idx=0, split='test'):
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    my_image_indices = metadata[metadata['split'] == split].index[start_idx:top_n+start_idx]
    return my_image_indices


def retrieve_present_concepts(sample_idx, dataset_name):
    # Define the path to the metadata file based on the dataset
    data_dir = f'../Data/{dataset_name}/'
    metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

    # Select the metadata for the image at img_idx
    img_metadata = metadata_df.iloc[sample_idx]

    # Extract the column names for categories and supercategories
    category_columns = [col for col in metadata_df.columns if col not in ['image_path']]

    # Initialize an empty list to store the present concepts
    present_concepts = []

    # Iterate through all the categories and supercategories
    for concept in category_columns:
        if img_metadata[concept] == 1:
            present_concepts.append(concept) 

    return present_concepts


def get_split_df(dataset_name):
    """
    Expands an image-level metadata DataFrame to a per-patch split DataFrame.

    Args:
        image_metadata_df (pd.DataFrame): DataFrame containing image-level metadata, including a "split" column.
        num_patches (int): Number of patches per image (e.g., 14x14 = 196 patches).

    Returns:
        pd.DataFrame: A new DataFrame where each patch has its own row and inherits the split from the image.
    """
    per_sample_metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    split_df = per_sample_metadata_df['split']
    
    return split_df


def compute_cossim_w_vector(vector, embeddings):
    """
    Computes the cosine similarity between a given vector and all embeddings in the embeddings tensor.

    Args:
        vector (torch.Tensor): A tensor of shape (D,) representing the random vector.
        embeddings (torch.Tensor): A tensor of shape (N, D) containing the embeddings, 
            where N is the number of embeddings, and D is the dimension of each embedding.

    Returns:
        torch.Tensor: A tensor of cosine similarities between the random vector and all embeddings.
    """
    vector = vector.to(embeddings.device)
    cosine_similarities = F.cosine_similarity(embeddings, vector.unsqueeze(0), dim=1)
    return cosine_similarities


###Visualizations###
def plot_image_with_attributes(image_index, dataset_name='CLEVR', save_image=False, test_only=True):
    """
    Plots an image with its associated attributes from a dataset.

    Args:
        image_index (int): The index of the image in the dataset.
        dataset_name (str): Name of the dataset to load the image from.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    # if test_only:
    #     metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 11)

    info = metadata.iloc[image_index]
    attributes = [attr for attr in info.index if ((attr not in ['image_path', 'class', 'split']) and (info.loc[attr] == 1))]

    image_path = info.loc['image_path']
    img = Image.open(f'../Data/{dataset_name}/{image_path}')

    # Create a new image with extra space at the bottom to accommodate the text
    text_height = 4  # Adjust the height of the text area
    new_img = Image.new('RGB', (img.width, img.height + text_height * 15), color=(255, 255, 255))  # Added extra space for multiple lines
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # Prepare the text string with attributes separated by commas
    attribute_text = ', '.join(attributes)

    # Wrap the text if it's too wide
    max_width = img.width  # Keep some padding from the edge
    words = attribute_text.split(', ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line}, {word}" if current_line else word
        test_bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = test_bbox[2] - test_bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line + ",")  # Add comma at the end of each line
            current_line = word
    lines.append(current_line)  # No comma at the end of the last line

    # Draw the text on the new image, below the original image
    y_offset = img.height + 5  # Start a little below the image
    for line in lines:
        draw.text((0, y_offset), line, font=font, fill="black")
        y_offset += text_height  # Move down for the next line of text

    # Show the image with the attributes underneath it
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

    # Save the new image with attributes underneath
    if save_image:
        output_image_path = f'../Figs/{dataset_name}/examples/example_{image_index}.jpg'
        new_img.save(output_image_path, dpi=(500, 500))
        

def plot_random_image_samples(dataset_name='CLEVR', num_samples=10, save_image=True):
    """
    Plots random sample images with their attributes from a given dataset.

    Args:
        dataset_name (str): Name of the dataset to load images from.
        num_samples (int): Number of sample images to plot.
        save_image (Boolean): Whether to save png file of image.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    total_samples = len(metadata)

    random_indices = np.random.choice(total_samples, num_samples, replace=False)

    for idx in random_indices:
        plot_image_with_attributes(idx, dataset_name, save_image=save_image)
        

