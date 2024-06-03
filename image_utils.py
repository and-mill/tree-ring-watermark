from PIL import Image
import os


def load_pil(filename: str, dir_name: str = "cache"):

    # Full path for loading the image
    full_path = os.path.join(dir_name, filename)
    
    # Load and return the image
    return Image.open(full_path)


def save_pil(image: Image, filename: str, dir_name: str = "cache"):
    
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Full path for saving the image
    full_path = os.path.join(dir_name, filename)
    
    image.save(full_path)
    
    # Load the image to verify
    loaded_image = load_pil(filename)
    
    # Verify by comparing the two images
    assert list(image.getdata()) == list(loaded_image.getdata()), "Image not saved correctly"
