import SwiftUI

@main
struct TryAngleApp: App {
    init() {
        print("🎯🎯🎯 앱 시작! TryAngleApp init() 🎯🎯🎯")
        NSLog("🎯🎯🎯 NSLog: 앱 시작! TryAngleApp init() 🎯🎯🎯")

        // 파일로도 로그 저장
        let logMessage = "🎯 앱 시작 시각: \(Date())\n"
        if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let logFile = documentsPath.appendingPathComponent("app_log.txt")
            try? logMessage.write(to: logFile, atomically: true, encoding: .utf8)
            print("📝 로그 파일 위치: \(logFile.path)")
        }
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
    }
}
