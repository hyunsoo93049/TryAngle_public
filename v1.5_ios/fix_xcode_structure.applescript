#!/usr/bin/osascript

-- Xcode 프로젝트 구조 자동 정리 스크립트
tell application "Xcode"
    activate

    -- 현재 열려있는 프로젝트 확인
    if (count of documents) = 0 then
        display dialog "Xcode에서 TryAngleApp 프로젝트를 먼저 열어주세요." buttons {"확인"} default button 1
        return
    end if

    display dialog "이 스크립트는 Xcode 프로젝트 구조를 정리합니다. 계속하시겠습니까?" buttons {"취소", "계속"} default button 2

    if button returned of result is "계속" then
        display notification "프로젝트 구조 정리 중..." with title "Xcode Helper"

        -- 수동 작업 안내
        display dialog "다음 단계를 따라주세요:

1. 프로젝트 네비게이터에서 이름 없는 파란색 폴더를 찾아 Delete → Remove Reference

2. TryAngleApp (최상위) 우클릭 → Add Files to TryAngleApp

3. ReferenceImages 폴더 선택 후:
   - Copy items if needed: 체크 해제
   - Create folder references: 선택 (파란색)
   - Add to targets: TryAngleApp 체크

4. Add 클릭

5. Product → Clean Build Folder (Shift+Cmd+K)" buttons {"완료"} default button 1
    end if
end tell