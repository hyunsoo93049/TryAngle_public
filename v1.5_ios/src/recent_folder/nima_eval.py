import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# 🔹 NIMA 프로젝트 경로 추가
sys.path.append('/Users/hyunsoo/Try_Angle/Neural-IMage-Assessment')

from model.model import NIMA

# ---------------------------------------------------
# 1️⃣ Base 모델 (Mobilenet)
# ---------------------------------------------------
base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model = NIMA(base_model)

# ---------------------------------------------------
# 2️⃣ 가중치 로드
# ---------------------------------------------------
state_dict = torch.load(
    '/Users/hyunsoo/Try_Angle/Neural-IMage-Assessment/model/NIMA-spaq-46a7fcb7.pth',
    map_location='cpu',
    weights_only=False   # 🔥 PyTorch 2.6 이상 필수
)
# 일부 버전은 state_dict가 {"params": ...}로 감싸져 있을 수 있음
if "params" in state_dict:
    state_dict = state_dict["params"]

model.load_state_dict(state_dict, strict=False)
model.eval()

# ---------------------------------------------------
# 3️⃣ 테스트할 이미지
# ---------------------------------------------------
img_path = '/Users/hyunsoo/Try_Angle/data/sample_images/nojot.png'  # 원하는 이미지로 변경
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
    probs = output[0].softmax(dim=0).numpy()
    mean_score = sum((i + 1) * p for i, p in enumerate(probs))

print(f"📸 NIMA aesthetic mean score: {mean_score:.2f}")
print(f"Distribution (1~10): {probs.round(3)}")
