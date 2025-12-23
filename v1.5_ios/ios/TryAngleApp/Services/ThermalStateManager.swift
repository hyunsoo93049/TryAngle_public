import Foundation
import UIKit
import Combine

// MARK: - 발열 및 배터리 관리자
class ThermalStateManager: ObservableObject {

    @Published var currentThermalState: ProcessInfo.ThermalState = .nominal
    @Published var isLowPowerMode: Bool = false
    @Published var batteryLevel: Float = 1.0
    @Published var recommendedAnalysisInterval: TimeInterval = 0.016  // 기본 60fps

    private var cancellables = Set<AnyCancellable>()

    init() {
        setupMonitoring()
        updateRecommendedInterval()
    }

    private func setupMonitoring() {
        // 🔥 발열 상태 모니터링
        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateThermalState()
            }
            .store(in: &cancellables)

        // 🔋 저전력 모드 모니터링
        NotificationCenter.default.publisher(for: .NSProcessInfoPowerStateDidChange)
            .sink { [weak self] _ in
                self?.updatePowerState()
            }
            .store(in: &cancellables)

        // 🔋 배터리 레벨 모니터링
        UIDevice.current.isBatteryMonitoringEnabled = true
        NotificationCenter.default.publisher(for: UIDevice.batteryLevelDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateBatteryLevel()
            }
            .store(in: &cancellables)

        // 초기값 설정
        updateThermalState()
        updatePowerState()
        updateBatteryLevel()
    }

    private func updateThermalState() {
        DispatchQueue.main.async {
            self.currentThermalState = ProcessInfo.processInfo.thermalState
            self.updateRecommendedInterval()
            self.logThermalState()
        }
    }

    private func updatePowerState() {
        DispatchQueue.main.async {
            self.isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
            self.updateRecommendedInterval()
            print("🔋 저전력 모드: \(self.isLowPowerMode ? "ON" : "OFF")")
        }
    }

    private func updateBatteryLevel() {
        DispatchQueue.main.async {
            self.batteryLevel = UIDevice.current.batteryLevel
            self.updateRecommendedInterval()
        }
    }

    // MARK: - 권장 분석 간격 계산
    private func updateRecommendedInterval() {
        let interval: TimeInterval

        switch currentThermalState {
        case .nominal:
            // 정상 온도: 최대 성능 (60fps)
            interval = 0.016

        case .fair:
            // 약간 높은 온도: 최대 성능 유지 (60fps)
            interval = 0.016

        case .serious:
            // 높은 온도: 약간 낮춤 (45fps)
            interval = 0.022

        case .critical:
            // 매우 높은 온도: 30fps
            interval = 0.033

        @unknown default:
            interval = 0.033
        }

        // 🔋 저전력 모드면 45fps로 제한
        if isLowPowerMode {
            recommendedAnalysisInterval = max(interval, 0.022)
        }
        // 🔋 배터리 20% 이하면 45fps로 제한
        else if batteryLevel > 0 && batteryLevel < 0.2 {
            recommendedAnalysisInterval = max(interval, 0.022)
        }
        else {
            recommendedAnalysisInterval = interval
        }
    }

    // MARK: - 발열 상태 로깅
    private func logThermalState() {
        let stateEmoji: String
        let stateName: String

        switch currentThermalState {
        case .nominal:
            stateEmoji = "❄️"
            stateName = "정상"
        case .fair:
            stateEmoji = "☁️"
            stateName = "약간 따뜻"
        case .serious:
            stateEmoji = "🔥"
            stateName = "뜨거움"
        case .critical:
            stateEmoji = "🚨"
            stateName = "매우 뜨거움"
        @unknown default:
            stateEmoji = "❓"
            stateName = "알 수 없음"
        }

        print("\(stateEmoji) 발열 상태: \(stateName) → 권장 간격: \(Int(recommendedAnalysisInterval * 1000))ms")
    }

    // MARK: - 분석 실행 가능 여부
    func shouldPerformAnalysis() -> Bool {
        // 🔥 최대 성능 모드: 모든 프레임 분석 (스킵 없음)
        // critical 상태에서도 interval로만 조절
        return true
    }

    // MARK: - CoreML 옵션 최적화
    func getCoreMLFlags() -> UInt32 {
        // 저전력 모드나 높은 발열 상태에서는 전력 효율 우선
        if isLowPowerMode || currentThermalState == .serious || currentThermalState == .critical {
            // CoreML 저전력 플래그 (정확도는 약간 낮지만 효율적)
            return 1  // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE
        }
        return 0  // 기본 (최고 성능)
    }
}
