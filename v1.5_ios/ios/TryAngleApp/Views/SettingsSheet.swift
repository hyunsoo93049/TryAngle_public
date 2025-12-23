import SwiftUI

struct SettingsSheet: View {
    @Binding var showGrid: Bool
    @Binding var showFPS: Bool
    @Binding var autoCapture: Bool
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                Section {
                    Toggle(isOn: $showGrid) {
                        HStack {
                            Image(systemName: "grid")
                                .foregroundColor(.blue)
                                .frame(width: 28)
                            Text("그리드 표시")
                        }
                    }

                    Toggle(isOn: $autoCapture) {
                        HStack {
                            Image(systemName: "camera.fill")
                                .foregroundColor(.blue)
                                .frame(width: 28)
                            Text("자동 촬영")
                        }
                    }
                } header: {
                    Text("촬영 옵션")
                }

                Section {
                    Toggle(isOn: $showFPS) {
                        HStack {
                            Image(systemName: "speedometer")
                                .foregroundColor(.blue)
                                .frame(width: 28)
                            Text("성능 정보 표시")
                        }
                    }
                } header: {
                    Text("디버그")
                }

                Section {
                    HStack {
                        Text("버전")
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("정보")
                }
            }
            .navigationTitle("설정")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("완료") {
                        dismiss()
                    }
                }
            }
        }
    }
}
