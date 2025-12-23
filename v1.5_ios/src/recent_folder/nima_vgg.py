import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# 🔹 NIMA 프로젝트 경로 추가 (model/model.py 접근용)
sys.path.append('/Users/hyunsoo/Try_Angle/Neural-IMage-Assessment')
from model.model import NIMA


# ---------------------------------------------------
# 1️⃣ Base model (VGG16)
# ---------------------------------------------------
base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model = NIMA(base_model)

# ---------------------------------------------------
# 2️⃣ 가중치 로드 (VGG16용)
# ---------------------------------------------------
weight_path = '/Users/hyunsoo/Try_Angle/Neural-IMage-Assessment/model/vgg16_aesthetic_model.pth'

state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)

# 일부 버전은 "params" 키로 감싸져 있음 → 자동 처리
if "params" in state_dict:
    state_dict = state_dict["params"]

model.load_state_dict(state_dict, strict=False)
model.eval()

# ---------------------------------------------------
# 3️⃣ 입력 이미지 전처리
# ---------------------------------------------------
img_path = '/Users/hyunsoo/Try_Angle/data/sample_images/wow.jpg'  # 원하는 이미지 경로
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)

# ---------------------------------------------------
# 4️⃣ 추론 (평균 미학 점수 계산)
# ---------------------------------------------------
with torch.no_grad():
    output = model(img_tensor)
    # NIMA(VGG16)는 Softmax 미포함 → 여기서 직접 적용
    probs = torch.softmax(output[0], dim=0).numpy()
    mean_score = sum((i + 1) * p for i, p in enumerate(probs))

# ---------------------------------------------------
# 5️⃣ 출력
# ---------------------------------------------------
print(f"📸 NIMA (VGG16) aesthetic mean score: {mean_score:.2f}")
print(f"Distribution (1~10): {probs.round(3)}")
