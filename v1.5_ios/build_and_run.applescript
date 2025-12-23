tell application "Xcode"
    activate
    delay 2

    -- 프로젝트 열기
    set myProject to "/Users/hyunsoo/Try_Angle/ios/TryAngleApp/TryAngleApp.xcodeproj"
    open myProject
    delay 3

    -- Clean Build Folder
    tell application "System Events"
        tell process "Xcode"
            -- Product 메뉴 클릭
            click menu item "Clean Build Folder" of menu "Product" of menu bar 1
            delay 1

            -- Clean 확인 다이얼로그가 나타나면 Clean 버튼 클릭
            try
                click button "Clean" of sheet 1 of window 1
            end try

            delay 5

            -- Build
            click menu item "Build" of menu "Product" of menu bar 1

            delay 10

            -- Run
            click menu item "Run" of menu "Product" of menu bar 1
        end tell
    end tell
end tell

display notification "빌드 및 실행이 시작되었습니다!" with title "TryAngleApp"