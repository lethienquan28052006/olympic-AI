import torch
import torchvision.transforms as transforms
from PIL import Image
from scripts.model import MushroomNet

# Load model
model = MushroomNet(num_classes=4)
model.load_state_dict(torch.load("models/mushroom_model.pth"))
model.eval()

# Transform ảnh test
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Test ảnh từ thư mục test_data/
import os
test_dir = "test_data"
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0) 

    output = model(image)
    pred = torch.argmax(output, dim=1).item()
    
    print(f"Ảnh {img_name} được dự đoán là loại nấm: {pred}")
