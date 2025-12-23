import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# ---------------------------------------------------
# 1️⃣ 완전한 NIMA 모델 직접 구성
# ---------------------------------------------------
class NIMAAesthetic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # MobileNetV2 features만 사용
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        
        # NIMA classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.75),
            torch.nn.Linear(1280, 10),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)      # (batch, 1280, 7, 7)
        x = self.classifier(x)    # (batch, 10)
        return x

# 모델 생성
model = NIMAAesthetic()

# ---------------------------------------------------
# 2️⃣ 가중치 로드 및 검증
# ---------------------------------------------------
state_dict = torch.load(
    '/Users/hyunsoo/Try_Angle/Neural-IMage-Assessment/model/NIMA-spaq-46a7fcb7.pth',
    map_location='cpu',
    weights_only=False
)

# state_dict 구조 확인
print("=" * 60)
print("📦 원본 State Dict 키 목록:")
print("=" * 60)
if "params" in state_dict:
    state_dict = state_dict["params"]
    
for key in list(state_dict.keys())[:10]:  # 처음 10개만 출력
    print(f"  - {key}: {state_dict[key].shape}")
print(f"  ... (총 {len(state_dict)} 개의 키)")
print()

# 가중치 키 매칭
new_state_dict = {}
matched_keys = []
unmatched_keys = []

for key, value in state_dict.items():
    if key.startswith('base_model.'):
        new_key = key.replace('base_model.', '')
        new_state_dict[new_key] = value
        matched_keys.append(key)
    elif key.startswith('classifier.'):
        new_state_dict[key] = value
        matched_keys.append(key)
    else:
        unmatched_keys.append(key)

print("=" * 60)
print("🔄 가중치 로딩 결과:")
print("=" * 60)
print(f"✅ 매칭된 키: {len(matched_keys)}개")
print(f"❌ 매칭되지 않은 키: {len(unmatched_keys)}개")
if unmatched_keys:
    print("\n매칭되지 않은 키들:")
    for key in unmatched_keys[:5]:
        print(f"  - {key}")
print()

# 가중치 로드
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
print("=" * 60)
print("🔍 모델 로딩 상세 정보:")
print("=" * 60)
print(f"누락된 키: {len(missing_keys)}개")
if missing_keys:
    for key in missing_keys[:5]:
        print(f"  - {key}")
print(f"\n예상치 못한 키: {len(unexpected_keys)}개")
if unexpected_keys:
    for key in unexpected_keys[:5]:
        print(f"  - {key}")
print()

model.eval()

# ---------------------------------------------------
# 3️⃣ 여러 테스트 이미지로 검증
# ---------------------------------------------------
test_images = [
    '/Users/hyunsoo/Try_Angle/data/sample_images/wow.jpg',
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device('cpu')
model.to(device)

print("=" * 60)
print("📸 이미지별 미학 점수 분석:")
print("=" * 60)

for img_path in test_images:
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = output[0].cpu().numpy()
            
            # 통계 계산
            mean_score = sum((i + 1) * p for i, p in enumerate(probs))
            std_score = np.sqrt(sum(((i + 1) - mean_score) ** 2 * p for i, p in enumerate(probs)))
            
            print(f"\n📁 파일: {img_path.split('/')[-1]}")
            print(f"   평균 점수: {mean_score:.3f}")
            print(f"   표준편차: {std_score:.3f}")
            print(f"   분포 (1~10점):")
            
            # 막대 그래프 형태로 출력
            for i, p in enumerate(probs):
                bar = '█' * int(p * 50)  # 50자 기준
                print(f"      {i+1:2d}점: {p:.4f} {bar}")
            
            # 가장 높은 확률 3개
            top3_indices = np.argsort(probs)[-3:][::-1]
            print(f"   Top 3 점수: ", end="")
            print(", ".join([f"{idx+1}점({probs[idx]:.3f})" for idx in top3_indices]))
            
    except FileNotFoundError:
        print(f"\n❌ 파일을 찾을 수 없습니다: {img_path}")

print("\n" + "=" * 60)
print("💡 분석 포인트:")
print("=" * 60)
print("1. 모든 점수가 비슷하게 분산되어 있다면 → 가중치 로딩 실패")
print("2. 특정 점수에 집중되어 있다면 → 가중치는 로드되었지만 학습 데이터셋과 맞지 않음")
print("3. 표준편차가 매우 작다면 → 모델이 확신 없이 중간값 출력")
print("4. SPAQ 데이터셋은 스마트폰 사진 품질 평가용입니다")
print("   - 블러, 노이즈, 노출 등 기술적 품질에 민감")
print("   - 구도, 감성 등 예술적 요소는 덜 민감할 수 있음")