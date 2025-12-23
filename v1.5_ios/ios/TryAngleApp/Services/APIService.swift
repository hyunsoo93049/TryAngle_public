import Foundation
import UIKit

class APIService {
    // MARK: - Configuration
    // TODO: 실제 백엔드 서버 IP로 변경 필요
    // 같은 WiFi 네트워크에서 맥북의 IP 주소 사용
    // 예: "http://192.168.0.10:8000"
    private let baseURL = "http://192.168.137.138:8000"

    // MARK: - Singleton
    static let shared = APIService()
    private init() {}

    // MARK: - API Methods

    /// 실시간 프레임 분석
    func analyzeFrame(
        referenceImage: UIImage,
        currentFrame: UIImage
    ) async throws -> AnalysisResponse {

        let url = URL(string: "\(baseURL)/api/analyze/realtime")!

        // Multipart form data 생성
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        // Body 생성
        var body = Data()

        // 레퍼런스 이미지 추가
        if let refData = referenceImage.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"reference\"; filename=\"reference.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(refData)
            body.append("\r\n".data(using: .utf8)!)
        }

        // 현재 프레임 추가
        if let frameData = currentFrame.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"current_frame\"; filename=\"frame.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(frameData)
            body.append("\r\n".data(using: .utf8)!)
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        // 요청 전송
        let (data, response) = try await URLSession.shared.data(for: request)

        // 응답 확인
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.invalidResponse
        }

        // JSON 파싱
        let decoder = JSONDecoder()
        let result = try decoder.decode(AnalysisResponse.self, from: data)

        return result
    }

    /// 서버 상태 확인
    func checkServerStatus() async throws -> Bool {
        let url = URL(string: "\(baseURL)/")!
        let (_, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            return false
        }

        return httpResponse.statusCode == 200
    }
}

// MARK: - Errors

enum APIError: Error {
    case invalidResponse
    case networkError
    case encodingError

    var localizedDescription: String {
        switch self {
        case .invalidResponse:
            return "서버 응답이 올바르지 않습니다"
        case .networkError:
            return "네트워크 연결을 확인해주세요"
        case .encodingError:
            return "이미지 인코딩 실패"
        }
    }
}
