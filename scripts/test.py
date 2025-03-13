import torch
import torchvision.transforms as transforms
from PIL import Image
from scripts.model import MushroomNet
import pandas as pd
import os

# Load model đã train
model = MushroomNet(num_classes=4)
model.load_state_dict(torch.load("models/mushroom_model.pth"))
model.eval()

# Transform ảnh test
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Thư mục test
test_dir = "test_data"

# Danh sách ảnh và kết quả dự đoán
test_images = sorted(os.listdir(test_dir))  # Sắp xếp để đảm bảo thứ tự ảnh
predictions = []

# Dự đoán từng ảnh
for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    
    image = Image.open(img_path).convert("RGB")  # Đảm bảo ảnh có 3 kênh
    image = transform(image).unsqueeze(0)  

    output = model(image)
    pred = torch.argmax(output, dim=1).item()
    predictions.append(pred)
    print(f"Ảnh {img_name} được dự đoán là loại nấm: {pred}")

# Tạo DataFrame để xuất ra CSV
df = pd.DataFrame({
    "id": [f"{i+1:03d}" for i in range(len(test_images))],  # Định dạng ID: 001, 002, 003...
    "type": predictions
})

# Xuất file CSV theo format Kaggle
df.to_csv("submission.csv", index=False)
print("✅ Đã lưu submission.csv thành công!")
