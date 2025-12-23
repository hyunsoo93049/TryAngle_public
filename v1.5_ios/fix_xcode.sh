#!/bin/bash

echo "🔧 TryAngleApp Xcode 프로젝트 구조 정리 스크립트"
echo "================================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}📋 다음 단계를 Xcode에서 실행해주세요:${NC}\n"

echo -e "${GREEN}1단계: 중복 제거${NC}"
echo "   • Xcode 프로젝트 네비게이터에서:"
echo "   • 이름 없는 파란색 폴더 선택 → Delete → Remove Reference"
echo "   • ReferenceImages 폴더가 있다면 삭제 → Remove Reference"
echo ""

echo -e "${GREEN}2단계: 폴더 구조 재구성${NC}"
echo "   • TryAngleApp (맨 위 파란색 아이콘) 우클릭"
echo "   • 'Add Files to TryAngleApp...' 선택"
echo "   • 다음 폴더들을 하나씩 추가:"
echo "     - Extensions"
echo "     - Models"
echo "     - Services"
echo "     - Utils"
echo "     - Views"
echo "   • 각 폴더 추가 시 옵션:"
echo "     ❌ Copy items if needed (체크 해제)"
echo "     ✅ Create groups (노란색 폴더)"
echo "     ✅ Add to targets: TryAngleApp"
echo ""

echo -e "${GREEN}3단계: ReferenceImages 폴더 추가${NC}"
echo "   • TryAngleApp (맨 위) 우클릭"
echo "   • 'Add Files to TryAngleApp...' 선택"
echo "   • ReferenceImages 폴더 선택"
echo "   • 옵션:"
echo "     ❌ Copy items if needed (체크 해제)"
echo "     ✅ Create folder references (파란색 폴더) ← 중요!"
echo "     ✅ Add to targets: TryAngleApp"
echo ""

echo -e "${GREEN}4단계: Build Phases 확인${NC}"
echo "   • 프로젝트 파일 클릭 → TARGETS → TryAngleApp"
echo "   • Build Phases 탭 → Copy Bundle Resources"
echo "   • ReferenceImages가 있는지 확인"
echo "   • 없으면 + 버튼으로 추가"
echo ""

echo -e "${GREEN}5단계: Clean & Build${NC}"
echo "   • Product → Clean Build Folder (Shift+Cmd+K)"
echo "   • Product → Build (Cmd+B)"
echo ""

echo -e "${YELLOW}📂 현재 ReferenceImages 폴더 구조:${NC}"
echo "   ReferenceImages/"
ls -la /Users/hyunsoo/Try_Angle/ios/TryAngleApp/ReferenceImages/ 2>/dev/null | grep "^d" | awk '{print "   ├── " $NF "/"}'

echo ""
echo -e "${GREEN}✨ 위 단계를 완료하면 이미지가 정상적으로 로드됩니다!${NC}"

# Xcode 열기
echo ""
read -p "Xcode를 열까요? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    open -a Xcode /Users/hyunsoo/Try_Angle/ios/TryAngleApp/
fi