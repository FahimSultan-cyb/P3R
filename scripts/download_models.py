import os
import gdown

def download_models():
    os.makedirs("models", exist_ok=True)
    
    model_urls = {
        "models/p3r_headgate_model1.pth": "https://drive.google.com/file/d/YOUR_MODEL_ID1/view?usp=sharing",
        "models/symbolic_classifier1n.pth": "https://drive.google.com/file/d/YOUR_MODEL_ID2/view?usp=sharing"
    }
    
    print("Downloading pre-trained models...")
    
    for filepath, url in model_urls.items():
        if os.path.exists(filepath):
            print(f"Model already exists: {filepath}")
            continue
            
        print(f"Downloading {filepath}...")
        try:
            # Extract file ID from Google Drive URL
            file_id = url.split('/d/')[1].split('/')[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, filepath, quiet=False)
            print(f"✓ Downloaded: {filepath}")
        except Exception as e:
            print(f"✗ Failed to download {filepath}: {e}")
            print("Please manually download the model files to the models/ directory")
    
    print("\nModel download completed!")

if __name__ == "__main__":
    download_models()