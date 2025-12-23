# Public Repository 보안 체크리스트

Public repository에 코드를 올리기 전 반드시 확인해야 할 항목들입니다.

---

## ✅ Push 전 필수 확인사항

### 1. 민감한 정보 검사

- [ ] API 키가 포함되지 않았는가?
- [ ] 비밀번호나 토큰이 없는가?
- [ ] .env 파일이 제외되었는가?
- [ ] 데이터베이스 접속 정보가 없는가?
- [ ] 개인 이메일이나 전화번호가 없는가?

### 2. 핵심 알고리즘 보호

- [ ] `v1.5_realtime` 폴더가 제외되었는가?
- [ ] `v1.5_ios_realtime` 폴더가 제외되었는가?
- [ ] `compare_final*.py` 파일들이 제외되었는가?
- [ ] 모델 변환 스크립트(`convert_*.py`)가 제외되었는가?

### 3. 모델 파일 확인

- [ ] `.pt` 파일이 없는가?
- [ ] `.onnx` 파일이 없는가?
- [ ] `.tar.gz` 모델 압축 파일이 없는가?
- [ ] 학습된 가중치 파일이 없는가?

### 4. 테스트 데이터

- [ ] 개인정보가 포함된 이미지가 없는가?
- [ ] 테스트용 개인 데이터가 제거되었는가?
- [ ] `data/ES`, `data/SH` 등 개인 폴더가 제외되었는가?

### 5. Git History

- [ ] 과거 커밋에 민감한 정보가 없는가?
- [ ] 새로운 Public repo로 시작하는가? (권장)

---

## 🔧 자동 검사 도구

### 1. 민감한 파일 검색

```bash
# Public repo에서 실행
cd ../Try_Angle_Public

# API 키 검색
grep -r "api_key\|API_KEY\|secret\|password" . --include="*.py" --include="*.js"

# 이메일 주소 검색
grep -r -E "\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b" . --include="*.py" --include="*.js"

# 핵심 알고리즘 파일 검색
find . -name "compare_final*.py" -o -name "convert_*.py"
```

### 2. 파일 크기 확인

```bash
# 큰 파일 찾기 (모델 파일 등)
find . -type f -size +10M

# 확장자별 파일 개수
find . -name "*.pt" -o -name "*.onnx" | wc -l
```

---

## 📋 단계별 가이드

### 초기 설정 (1회만)

```bash
# 1. Public repo 생성
python create_public_repo.py

# 2. 생성된 파일 확인
cd ../Try_Angle_Public
ls -la

# 3. 민감한 파일 수동 검사
grep -r "TODO\|FIXME\|secret\|password" .

# 4. Git 초기화
git init
git add .
git commit -m "Initial commit: Public version"

# 5. GitHub에 Push
git remote add origin https://github.com/hyunsoo93049/Try_Angle_Public.git
git push -u origin main
```

### 업데이트 시 (매번)

```bash
# 1. Private repo에서 변경사항 커밋
cd /c/try_angle
git add .
git commit -m "Update: [설명]"

# 2. Public으로 동기화
python sync_to_public.py

# 3. Public repo에서 확인
cd ../Try_Angle_Public
git status
git diff

# 4. 민감한 내용 검사
grep -r "api_key\|secret\|password" .

# 5. 문제없으면 Push
git add .
git commit -m "Update: [설명]"
git push
```

---

## ⚠️ 자주하는 실수

### 1. Git History에 민감한 정보 포함

**문제**: 현재 파일에는 없지만 과거 커밋에 민감한 정보가 있는 경우

**해결**:
```bash
# 특정 파일을 history에서 완전히 제거
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# 강제 푸시 (주의!)
git push origin --force --all
```

### 2. .gitignore 설정 누락

**문제**: .gitignore에 추가했지만 이미 추적 중인 파일

**해결**:
```bash
# 캐시에서 제거 (파일은 유지)
git rm --cached <file>

# 전체 캐시 재생성
git rm -r --cached .
git add .
git commit -m "Fix .gitignore"
```

### 3. 실수로 민감한 파일을 Push

**즉시 조치**:
```bash
# 1. Repository를 Private으로 변경 (GitHub 설정)

# 2. 해당 커밋 제거
git reset --hard HEAD~1
git push --force

# 3. 또는 파일만 제거
git rm <sensitive-file>
git commit --amend
git push --force
```

---

## 🔐 보안 베스트 프랙티스

### 1. 환경 변수 사용

**나쁜 예**:
```python
API_KEY = "sk-1234567890abcdef"  # ❌
```

**좋은 예**:
```python
import os
API_KEY = os.getenv("API_KEY")  # ✅
```

### 2. Config 파일 분리

```python
# config.py (Public)
DEFAULT_MODEL = "yolo11n"
CONFIDENCE_THRESHOLD = 0.5

# config_private.py (Private only)
API_KEYS = {
    "openai": "sk-...",
    "google": "AIza..."
}
```

### 3. 데모 모드 제공

```python
# Public repo에는 데모 모드만
if os.path.exists("models/proprietary"):
    from .proprietary_analyzer import AdvancedAnalyzer
else:
    from .demo_analyzer import DemoAnalyzer  # 간단한 데모 버전
```

---

## 📞 문제 발생 시

### 민감한 정보가 Public에 노출된 경우

1. **즉시 Repository를 Private으로 변경**
2. **해당 API 키/토큰 무효화 및 재발급**
3. **Git history에서 완전히 제거**
4. **보안팀에 보고 (필요시)**

### 핵심 알고리즘이 노출된 경우

1. **즉시 Repository를 Private으로 변경**
2. **해당 커밋 삭제 또는 force push로 덮어쓰기**
3. **새로운 Public repo 생성 고려**

---

## ✨ 추천 워크플로우

```
[Private Repo]
    ↓
개발 및 테스트
    ↓
핵심 알고리즘 확인
    ↓
[sync_to_public.py 실행]
    ↓
자동 필터링
    ↓
[Public Repo]
    ↓
수동 검사 (체크리스트)
    ↓
Git diff 확인
    ↓
Push to GitHub
```

---

## 📚 참고 자료

- [GitHub - 민감한 데이터 제거하기](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git - filter-branch](https://git-scm.com/docs/git-filter-branch)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) - Git history 정리 도구

---

**마지막 업데이트**: 2024-12-24
