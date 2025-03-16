import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def verify_celeba_dataset(root_dir="data/celeba", sample_size=10):
    """
    Verify the CelebA dataset integrity
    """
    print("Verifying CelebA dataset...")
    
    # 1. Check directory structure
    img_dir = os.path.join(root_dir, "img_align_celeba")
    attr_file = os.path.join(root_dir, "list_attr_celeba.txt")
    
    if not os.path.exists(img_dir):
        print(f"❌ Error: Image directory not found at {img_dir}")
        return False
    
    if not os.path.exists(attr_file):
        print(f"❌ Error: Attribute file not found at {attr_file}")
        return False
    
    print("✅ Directory structure verified")
    
    # 2. Check attributes file
    try:
        attr_df = pd.read_csv(attr_file, sep='\s+', skiprows=1)
        num_images = len(attr_df)
        num_attributes = len(attr_df.columns)
        print(f"✅ Attributes file verified: {num_images} images, {num_attributes} attributes")
    except Exception as e:
        print(f"❌ Error reading attributes file: {e}")
        return False
    
    # 3. Check image files
    image_files = os.listdir(img_dir)
    print(f"Found {len(image_files)} images")
    
    # Check a sample of images
    sample_files = sorted(image_files)[:sample_size]
    print(f"\nVerifying {sample_size} sample images...")
    
    for img_file in tqdm(sample_files):
        try:
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            img.verify()  # Verify image integrity
            
            # Check image size
            img = Image.open(img_path)
            if img.size != (178, 218):
                print(f"❌ Warning: Unexpected image size for {img_file}: {img.size}")
                continue
                
        except Exception as e:
            print(f"❌ Error with image {img_file}: {e}")
            return False
    
    print("\n✅ Sample images verified successfully")
    print("\nDataset Verification Summary:")
    print(f"- Total images found: {len(image_files)}")
    print(f"- Expected images: {num_images}")
    print(f"- Attributes file: {os.path.basename(attr_file)}")
    print(f"- Number of attributes: {num_attributes}")
    
    if len(image_files) != num_images:
        print(f"⚠️ Warning: Number of images ({len(image_files)}) doesn't match attributes file ({num_images})")
    else:
        print("✅ Number of images matches attributes file")
    
    return True

if __name__ == "__main__":
    verify_celeba_dataset() 