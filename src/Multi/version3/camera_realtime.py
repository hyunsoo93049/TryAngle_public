# ============================================================
# 📹 TryAngle - Realtime Camera Feedback System
# 실시간 카메라 촬영 가이드 (크로스 플랫폼 호환)
# ============================================================

import cv2
import numpy as np
import time
import sys
from typing import Dict, List, Optional
import tempfile
import threading
from queue import Queue, Empty
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont

# 프로젝트 루트 경로 자동 설정
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from analysis.image_analyzer import ImageAnalyzer
from analysis.image_comparator import ImageComparator

# Phase 3.3: Visual Guide Overlay
try:
    from utils.visual_guide import VisualGuideOverlay
    VISUAL_GUIDE_AVAILABLE = True
except ImportError:
    print("⚠️ Visual Guide Overlay not available")
    VISUAL_GUIDE_AVAILABLE = False


class Config:
    """설정 파일 로더"""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)

        # 경로를 절대 경로로 변환
        self.project_root = PROJECT_ROOT
        self._resolve_paths()

    def _resolve_paths(self):
        """상대 경로를 절대 경로로 변환"""
        for key, value in self.data['paths'].items():
            if key != 'default_reference':
                abs_path = (self.project_root / value).resolve()
                self.data['paths'][key] = abs_path

    def get(self, *keys, default=None):
        """중첩된 키 접근: config.get('camera', 'width')"""
        result = self.data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def get_path(self, *keys) -> Path:
        """경로 반환"""
        return Path(self.get(*keys))


class RealtimeCameraAnalyzer:
    """
    실시간 카메라 피드백 시스템
    - 레퍼런스 이미지와 실시간 비교
    - 화면에 피드백 오버레이 표시
    """

    def __init__(
        self,
        reference_path: Path,
        camera_index: Optional[int] = None,
        config: Optional[Config] = None
    ):
        """
        Args:
            reference_path: 레퍼런스 이미지 경로
            camera_index: 카메라 인덱스 (None이면 config에서 읽음)
            config: 설정 객체
        """

        # 설정 로드
        self.config = config if config else Config()

        # 레퍼런스 이미지 확인
        if not reference_path.exists():
            raise FileNotFoundError(f"레퍼런스 이미지를 찾을 수 없습니다: {reference_path}")

        self.reference_path = reference_path
        self.camera_index = camera_index if camera_index is not None else self.config.get('camera', 'default_index')

        # 카메라 설정
        self.frame_width = self.config.get('camera', 'width')
        self.frame_height = self.config.get('camera', 'height')
        self.analysis_interval = self.config.get('camera', 'analysis_interval')

        # 상태 변수
        self.last_analysis_time = 0
        self.current_feedback = []
        self.is_analyzing = False
        self.fps = 0
        self.analysis_count = 0

        # 비동기 분석을 위한 큐와 스레드
        self.analysis_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.analysis_thread = None
        self.stop_analysis = False

        # 레퍼런스 이미지 분석 (한 번만)
        print("\n" + "="*60)
        print("📸 레퍼런스 이미지 분석 중...")
        print("="*60)
        self.ref_analyzer = ImageAnalyzer(str(reference_path))
        self.ref_data = self.ref_analyzer.analyze()
        print("✅ 레퍼런스 분석 완료!")

        # 카메라 초기화
        self.cap = None

        # UI 설정 (config에서 로드)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.config.get('ui', 'font_scale')
        self.font_thickness = self.config.get('ui', 'font_thickness')
        self.line_height = self.config.get('ui', 'line_height')

        # 색상 (BGR) - config에서 로드
        colors = self.config.get('ui', 'colors')
        self.color_bg = tuple(colors['background'])
        self.color_text = tuple(colors['text'])
        self.color_priority_high = tuple(colors['priority_high'])
        self.color_priority_mid = tuple(colors['priority_mid'])
        self.color_priority_low = tuple(colors['priority_low'])
        self.color_info = tuple(colors['info'])
        self.color_success = tuple(colors['success'])

        # 텍스트 렌더링 (PIL 사용: 한글 지원)
        self.base_font_size = self.config.get('ui', 'base_font_size', default=36) or 36
        self._init_text_renderer()

        # 임계값 (config에서 로드)
        self.thresholds = self.config.get('thresholds')

        # Phase 3.3: 시각적 가이드 오버레이
        if VISUAL_GUIDE_AVAILABLE:
            self.visual_guide = VisualGuideOverlay()
            self.show_visual_guides = True  # 토글 가능
        else:
            self.visual_guide = None
            self.show_visual_guides = False

    def _init_camera(self) -> bool:
        """카메라 초기화"""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"❌ 카메라를 열 수 없습니다 (index={self.camera_index})")
            return False

        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # 실제 해상도 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ 카메라 초기화 완료: {actual_width}x{actual_height}")

        return True

    def _analyze_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        프레임 분석 및 피드백 생성

        Args:
            frame: OpenCV 이미지 (BGR)

        Returns:
            피드백 리스트
        """
        try:
            # 프레임을 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)

            # 사용자 이미지 분석
            user_analyzer = ImageAnalyzer(tmp_path)
            user_data = user_analyzer.analyze()

            # 비교
            feedback = self._generate_feedback(self.ref_data, user_data)

            # 임시 파일 삭제
            Path(tmp_path).unlink()

            self.analysis_count += 1

            return feedback

        except Exception as e:
            print(f"⚠️ 프레임 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _analysis_worker(self):
        """백그라운드 스레드에서 실행되는 분석 워커"""
        while not self.stop_analysis:
            try:
                frame = self.analysis_queue.get(timeout=0.1)
                feedback = self._analyze_frame(frame)

                try:
                    self.result_queue.put(feedback, block=False)
                except:
                    pass

            except Empty:
                continue
            except Exception as e:
                print(f"⚠️ 분석 워커 오류: {e}")

    def _start_analysis_thread(self):
        """분석 스레드 시작"""
        if self.analysis_thread is None or not self.analysis_thread.is_alive():
            self.stop_analysis = False
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()
            print("✅ 분석 스레드 시작")

    def _stop_analysis_thread(self):
        """분석 스레드 종료"""
        self.stop_analysis = True
        if self.analysis_thread is not None:
            self.analysis_thread.join(timeout=2.0)
            print("✅ 분석 스레드 종료")

    def _generate_feedback(self, ref_data: Dict, user_data: Dict) -> List[Dict]:
        """
        피드백 생성 (ImageComparator의 로직 간소화 버전)

        Returns:
            우선순위 순으로 정렬된 피드백 리스트
        """
        feedback = []

        # 1. 클러스터 비교 (정보성)
        ref_cluster = ref_data["cluster"]["cluster_id"]
        user_cluster = user_data["cluster"]["cluster_id"]

        if ref_cluster != user_cluster:
            feedback.append({
                "priority": 0,
                "category": "STYLE",
                "message": f"스타일: {ref_data['cluster']['cluster_label']} → {user_data['cluster']['cluster_label']}",
                "detail": "스타일이 다릅니다"
            })
        else:
            feedback.append({
                "priority": 0,
                "category": "STYLE",
                "message": f"스타일: {ref_data['cluster']['cluster_label']} ✓",
                "detail": "스타일이 일치합니다"
            })

        # 2. 거리 비교
        ref_depth = ref_data["depth"]["depth_mean"]
        user_depth = user_data["depth"]["depth_mean"]
        depth_ratio = user_depth / ref_depth if ref_depth > 0 else 1.0

        if abs(depth_ratio - 1.0) > self.thresholds['depth_ratio']:
            steps = round((depth_ratio - 1.0) * 3)
            direction = "뒤로" if steps > 0 else "앞으로"
            steps = abs(steps)

            feedback.append({
                "priority": 2.0,
                "category": "DISTANCE",
                "message": f"{steps}걸음 {direction}",
                "detail": f"거리 비율: {depth_ratio:.2f}"
            })

        # 3. 밝기 비교
        ref_brightness = ref_data["pixels"]["brightness"]
        user_brightness = user_data["pixels"]["brightness"]
        brightness_diff = user_brightness - ref_brightness

        if abs(brightness_diff) > self.thresholds['brightness_diff']:
            ev_adjustment = brightness_diff / 30
            direction = "어둡게" if brightness_diff > 0 else "밝게"

            feedback.append({
                "priority": 3.0,
                "category": "BRIGHTNESS",
                "message": f"EV {ev_adjustment:+.1f} ({direction})",
                "detail": f"밝기 차이: {brightness_diff:+.1f}"
            })

        # 4. 색감 비교
        ref_saturation = ref_data["pixels"]["saturation"]
        user_saturation = user_data["pixels"]["saturation"]
        saturation_diff = user_saturation - ref_saturation

        if abs(saturation_diff) > self.thresholds['saturation_diff']:
            direction = "채도 낮추기" if saturation_diff > 0 else "채도 높이기"

            feedback.append({
                "priority": 4.0,
                "category": "COLOR",
                "message": direction,
                "detail": f"채도 차이: {saturation_diff:+.2f}"
            })

        # 5. 구도 비교 (기울기)
        ref_tilt = ref_data["composition"]["tilt_angle"]
        user_tilt = user_data["composition"]["tilt_angle"]
        tilt_diff = user_tilt - ref_tilt

        if abs(tilt_diff) > self.thresholds['tilt_diff']:
            direction = "시계방향" if tilt_diff > 0 else "반시계방향"

            feedback.append({
                "priority": 5.0,
                "category": "COMPOSITION",
                "message": f"{abs(tilt_diff):.1f}° {direction} 회전",
                "detail": f"기울기 차이: {tilt_diff:+.1f}°"
            })

        # 우선순위 순으로 정렬
        feedback.sort(key=lambda x: x["priority"])

        return feedback

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """프레임에 피드백 오버레이 그리기 (Phase 3.3 통합)"""
        h, w = frame.shape[:2]

        # ==========================================
        # Phase 3.3: 시각적 가이드 오버레이
        # ==========================================
        if self.show_visual_guides and self.visual_guide is not None:
            # 1. 삼분할선 (Rule of Thirds)
            frame = self.visual_guide.draw_rule_of_thirds(frame, thickness=1)

            # 2. 수평선 가이드 (기울기 피드백이 있으면)
            tilt_feedback = [fb for fb in self.current_feedback if fb.get('category') == 'COMPOSITION']
            if tilt_feedback:
                # 현재 기울기 정보 추출 (간단히 0도로 가정, 실제로는 분석 데이터에서 가져와야 함)
                current_tilt = 0.0  # TODO: 실제 기울기 값으로 대체
                target_tilt = 0.0
                frame = self.visual_guide.draw_horizon_line(frame, current_tilt, target_tilt)

            # 3. 피드백 패널 (상단)
            feedback_messages = [fb['message'] for fb in self.current_feedback if fb.get('priority', 99) > 0]
            if feedback_messages:
                frame = self.visual_guide.draw_feedback_panel(
                    frame,
                    feedback_messages[:3],  # 최대 3개
                    position='top'
                )
                # 피드백 패널이 있으면 기존 텍스트 오버레이는 스킵
                # PIL 이미지로 변환 (한글 렌더링용)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                # 간단한 헤더만 하단에 표시
                self._put_text(
                    draw,
                    f"FPS: {self.fps:.1f} | Analysis: {self.analysis_count}",
                    (10, h - 40),
                    self.color_text,
                    scale=self.font_scale * 0.7,
                    thickness=max(1, self.font_thickness - 1)
                )

                # PIL → OpenCV 변환
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # ==========================================
        # 기존 텍스트 오버레이 (시각적 가이드 없을 때)
        # ==========================================
        overlay = frame.copy()

        # 반투명 배경
        overlay_height = 50 + len(self.current_feedback) * self.line_height
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), self.color_bg, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # PIL 이미지로 변환 (한글 렌더링용)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # 헤더
        self._put_text(
            draw,
            f"TryAngle - Realtime Guide | FPS: {self.fps:.1f} | Analysis: {self.analysis_count}",
            (10, 20),
            self.color_text,
            scale=self.font_scale,
            thickness=self.font_thickness
        )

        # 피드백 표시
        if self.current_feedback:
            info_messages = [fb for fb in self.current_feedback if fb["priority"] == 0]
            actionable_messages = [fb for fb in self.current_feedback if fb["priority"] > 0]

            y_offset = 60

            # 정보성 메시지
            if info_messages:
                for fb in info_messages:
                    self._put_text(
                        draw,
                        f"  {fb['message']}",
                        (10, y_offset),
                        self.color_info,
                        scale=self.font_scale * 0.9,
                        thickness=max(1, self.font_thickness - 1)
                    )
                    y_offset += self.line_height

            # 실행 가능한 메시지
            if actionable_messages:
                for i, fb in enumerate(actionable_messages, 1):
                    if fb["priority"] <= 2.0:
                        color = self.color_priority_high
                    elif fb["priority"] <= 4.0:
                        color = self.color_priority_mid
                    else:
                        color = self.color_priority_low

                    self._put_text(
                        draw,
                        f"  {i}. [{fb['category']}] {fb['message']}",
                        (10, y_offset),
                        color,
                        scale=self.font_scale,
                        thickness=self.font_thickness
                    )
                    y_offset += self.line_height
            else:
                self._put_text(
                    draw,
                    "  Perfect! No adjustments needed.",
                    (10, y_offset),
                    self.color_success,
                    scale=self.font_scale,
                    thickness=self.font_thickness
                )
        else:
            self._put_text(
                draw,
                "  Analyzing...",
                (10, 60),
                self.color_info,
                scale=self.font_scale,
                thickness=self.font_thickness
            )

        # PIL → OpenCV 변환
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 분석 중 표시
        if self.is_analyzing:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)

        return frame

    def _init_text_renderer(self):
        """한글 지원을 위한 폰트 로딩"""
        config_font = self.config.get('ui', 'font_path', default=None)
        font_candidates = []

        if config_font:
            font_candidates.append(config_font)

        # OS별 기본 폰트 후보
        if sys.platform == "darwin":
            font_candidates.extend([
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/Apple SD Gothic Neo.ttc"
            ])
        elif sys.platform.startswith("win"):
            font_candidates.extend([
                r"C:\Windows\Fonts\malgun.ttf",
                r"C:\Windows\Fonts\malgunsl.ttf",
                r"C:\Windows\Fonts\gulim.ttc"
            ])
        else:
            font_candidates.extend([
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            ])

        self._font_cache = {}
        self._font_path = None

        for path in font_candidates:
            if not path:
                continue
            font_path = Path(path).expanduser()
            if font_path.exists():
                self._font_path = str(font_path)
                break

        if self._font_path is None:
            print("⚠️ 한글 폰트를 찾지 못해 기본 폰트를 사용합니다. (config.yaml의 ui.font_path 설정 가능)")

    def _get_pil_font(self, scale: float = 1.0):
        """스케일에 맞는 PIL 폰트 캐시"""
        scale_key = round(scale, 2)
        if scale_key in self._font_cache:
            return self._font_cache[scale_key]

        font_size = max(12, int(self.base_font_size * scale))
        try:
            if self._font_path:
                font = ImageFont.truetype(self._font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        self._font_cache[scale_key] = font
        return font

    def _put_text(self, draw: ImageDraw.ImageDraw, text: str, position, color_bgr, scale: float = 1.0, thickness: Optional[int] = None):
        """PIL 기반 텍스트 출력 (OpenCV 컬러 → RGB 변환)"""
        font = self._get_pil_font(scale)
        rgb_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        # PIL stroke가 두꺼운 경우 번짐이 생기므로 살짝 줄여서 사용
        base_thickness = self.font_thickness if thickness is None else thickness
        stroke_width = max(0, base_thickness - 2)
        stroke_kwargs = {}
        if stroke_width > 0:
            stroke_kwargs = {
                "stroke_width": stroke_width,
                "stroke_fill": rgb_color
            }

        draw.text(
            position,
            text,
            font=font,
            fill=rgb_color,
            **stroke_kwargs
        )

    def run(self):
        """
        실시간 카메라 피드백 시스템 실행

        Controls:
            - 'q': 종료
            - 'r': 레퍼런스 이미지 재분석
            - 's': 현재 프레임 저장
            - 'g': 시각적 가이드 토글 (Phase 3.3)
            - SPACE: 분석 일시정지/재개
        """

        if not self._init_camera():
            return

        print("\n" + "="*60)
        print("📹 실시간 카메라 피드백 시작 (Phase 1-3 통합)")
        print("="*60)
        print("\n조작법:")
        print("  - 'q': 종료")
        print("  - 'r': 레퍼런스 재분석")
        print("  - 'g': 시각적 가이드 ON/OFF")
        print("  - 's': 현재 프레임 저장")
        print("  - SPACE: 분석 일시정지/재개")
        print("\n" + "="*60 + "\n")

        paused = False
        frame_count = 0
        start_time = time.time()

        # 분석 스레드 시작
        self._start_analysis_thread()

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다")
                    break

                frame_count += 1
                current_time = time.time()

                # FPS 계산
                elapsed = current_time - start_time
                if elapsed > 0:
                    self.fps = frame_count / elapsed

                # 주기적 분석
                if not paused and (current_time - self.last_analysis_time >= self.analysis_interval):
                    try:
                        self.analysis_queue.put(frame.copy(), block=False)
                        self.is_analyzing = True
                        self.last_analysis_time = current_time
                    except:
                        pass

                # 분석 결과 확인
                try:
                    feedback = self.result_queue.get(block=False)
                    self.current_feedback = feedback
                    self.is_analyzing = False
                except Empty:
                    pass

                # 오버레이 그리기
                display_frame = self._draw_overlay(frame)

                # 화면 표시
                cv2.imshow('TryAngle - Realtime Camera', display_frame)

                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 종료합니다")
                    break
                elif key == ord('r'):
                    print("\n🔄 레퍼런스 이미지 재분석...")
                    self.ref_analyzer = ImageAnalyzer(str(self.reference_path))
                    self.ref_data = self.ref_analyzer.analyze()
                    print("✅ 재분석 완료!")
                elif key == ord('s'):
                    save_path = Path(f"capture_{int(time.time())}.jpg")
                    cv2.imwrite(str(save_path), frame)
                    print(f"💾 프레임 저장: {save_path}")
                elif key == ord('g'):
                    # Phase 3.3: 시각적 가이드 토글
                    if self.visual_guide is not None:
                        self.show_visual_guides = not self.show_visual_guides
                        status = "ON" if self.show_visual_guides else "OFF"
                        print(f"👁️ 시각적 가이드: {status}")
                    else:
                        print("⚠️ 시각적 가이드를 사용할 수 없습니다")
                elif key == ord(' '):
                    paused = not paused
                    status = "일시정지" if paused else "재개"
                    print(f"⏸️ 분석 {status}")

        except KeyboardInterrupt:
            print("\n⚠️ 사용자 중단")

        finally:
            self._stop_analysis_thread()
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ 카메라 종료")


def main():
    """메인 함수"""

    # 설정 로드
    config = Config()

    # 레퍼런스 이미지 경로
    test_images_dir = config.get_path('paths', 'test_images_dir')
    default_ref = config.get('paths', 'default_reference')
    reference_path = test_images_dir / default_ref

    # 경로 확인
    if not reference_path.exists():
        print(f"❌ 레퍼런스 이미지를 찾을 수 없습니다: {reference_path}")
        print(f"\n📁 테스트 이미지 디렉토리: {test_images_dir}")

        if test_images_dir.exists():
            print("\n사용 가능한 이미지:")
            for f in test_images_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"  - {f.name}")
        return

    try:
        # 실시간 카메라 분석기 생성
        analyzer = RealtimeCameraAnalyzer(
            reference_path=reference_path,
            config=config
        )

        # 실행
        analyzer.run()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
