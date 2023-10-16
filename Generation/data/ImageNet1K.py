import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_image_paths_from_file(file_path):
    """
    Extracts the list of image paths from a text file. 
    Each line in the file is assumed to have the format '/path/to/image.jpeg number'
    
    Args:
    file_path (str): The path to the text file.

    Returns:
    List[str]: A list of image paths.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    
    image_paths, labels = [], []
    for line in lines:
        # if if_imagenette:
        #     class_name = line.split()[0].split('/')[1]
        #     if not class_name in _LABEL_MAP:
        #         continue
        image_paths.append(line.split()[0])
        labels.append(line.split()[-1])
        
    # image_paths = [line.split()[0] for line in lines]
    # labels = [line.split()[-1] for line in lines]
    # print(image_paths)
    return image_paths, labels

def mirror_directory_structure(img_paths, source_directory, dest_directory):
    """
    Creates a mirror of the directory structure of source_directory in dest_directory.
    It does this based on the unique class directories specified in img_paths.
    
    Args:
    img_paths (list): List of image paths with structure 'class_name/image_name.jpeg'.
    source_directory (str): The path of the source directory.
    dest_directory (str): The path of the destination directory.

    Returns:
    None
    """
    
    unique_class_names = set(path.split('/')[1] for path in img_paths)
    for class_name in unique_class_names:
        os.makedirs(os.path.join(dest_directory, class_name), exist_ok=True)
        
def create_ImageNetFolder(root_dir, out_dir):
    image_paths, labels = get_image_paths_from_file(os.path.join(root_dir,"file_list.txt")) 
    mirror_directory_structure(image_paths, root_dir, out_dir)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()