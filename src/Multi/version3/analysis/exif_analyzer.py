# ============================================================
# 📷 TryAngle - EXIF Analyzer
# 이미지 메타데이터(EXIF) 추출 및 카메라 설정 분석
# ============================================================

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Optional, Any
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent


class ExifAnalyzer:
    """
    EXIF 메타데이터 추출 및 분석

    주요 추출 항목:
    - ISO: 감도
    - F-Number: 조리개
    - ExposureTime: 셔터속도
    - FocalLength: 초점거리
    - WhiteBalance: 화이트밸런스
    - ExposureProgram: 촬영 모드
    - Flash: 플래시 사용
    - LensModel: 렌즈 모델
    """

    # EXIF 태그 매핑
    EXIF_TAGS = {
        'ISOSpeedRatings': 'iso',
        'ISO': 'iso',
        'FNumber': 'f_number',
        'ExposureTime': 'shutter_speed',
        'FocalLength': 'focal_length',
        'WhiteBalance': 'white_balance',
        'ExposureProgram': 'exposure_program',
        'ExposureBiasValue': 'exposure_compensation',
        'Flash': 'flash',
        'LensModel': 'lens_model',
        'Make': 'camera_make',
        'Model': 'camera_model',
        'DateTime': 'datetime',
        'DateTimeOriginal': 'datetime_original',
        'Orientation': 'orientation',
        'ExposureMode': 'exposure_mode',
        'MeteringMode': 'metering_mode',
        'SceneCaptureType': 'scene_type',
        'Contrast': 'contrast',
        'Saturation': 'saturation',
        'Sharpness': 'sharpness'
    }

    # 화이트밸런스 해석
    WHITE_BALANCE_MAP = {
        0: 'Auto',
        1: 'Manual',
        2: 'Auto (Tungsten)',
        3: 'Auto (Fluorescent)',
        4: 'Auto (Daylight)',
        5: 'Auto (Flash)',
        6: 'Auto (Cloudy)',
        7: 'Auto (Shade)'
    }

    # 노출 프로그램 해석
    EXPOSURE_PROGRAM_MAP = {
        0: 'Not Defined',
        1: 'Manual',
        2: 'Program AE',
        3: 'Aperture Priority',
        4: 'Shutter Priority',
        5: 'Creative (Slow)',
        6: 'Action (High-speed)',
        7: 'Portrait',
        8: 'Landscape'
    }

    def __init__(self, image_path: str):
        """
        Args:
            image_path: 이미지 파일 경로
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path
        self.exif_data = {}
        self.raw_exif = {}

        # EXIF 추출
        self._extract_exif()

    def _extract_exif(self):
        """EXIF 데이터 추출"""
        try:
            image = Image.open(self.image_path)
            exif = image.getexif()

            if exif is None or len(exif) == 0:
                print(f"  ⚠️ No EXIF data found in {os.path.basename(self.image_path)}")
                return

            # Raw EXIF 저장
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                self.raw_exif[tag_name] = value

            # 주요 EXIF 파싱
            self._parse_camera_settings()

        except Exception as e:
            print(f"  ⚠️ EXIF extraction failed: {e}")

    def _parse_camera_settings(self):
        """카메라 설정 파싱"""
        # ISO
        iso = self.raw_exif.get('ISOSpeedRatings') or self.raw_exif.get('ISO')
        if iso:
            self.exif_data['iso'] = int(iso) if isinstance(iso, (int, float)) else None

        # F-Number (조리개)
        f_number = self.raw_exif.get('FNumber')
        if f_number:
            if isinstance(f_number, tuple):
                self.exif_data['f_number'] = float(f_number[0]) / float(f_number[1])
            else:
                self.exif_data['f_number'] = float(f_number)

        # Shutter Speed (셔터속도)
        exposure_time = self.raw_exif.get('ExposureTime')
        if exposure_time:
            if isinstance(exposure_time, tuple):
                numerator, denominator = exposure_time
                self.exif_data['shutter_speed'] = float(numerator) / float(denominator)
                self.exif_data['shutter_speed_display'] = f"1/{int(denominator/numerator)}" if numerator < denominator else f"{numerator/denominator}"
            else:
                self.exif_data['shutter_speed'] = float(exposure_time)
                self.exif_data['shutter_speed_display'] = f"{exposure_time}s"

        # Focal Length (초점거리)
        focal_length = self.raw_exif.get('FocalLength')
        if focal_length:
            if isinstance(focal_length, tuple):
                self.exif_data['focal_length'] = float(focal_length[0]) / float(focal_length[1])
            else:
                self.exif_data['focal_length'] = float(focal_length)

        # White Balance
        wb = self.raw_exif.get('WhiteBalance')
        if wb is not None:
            self.exif_data['white_balance'] = self.WHITE_BALANCE_MAP.get(wb, f"Unknown ({wb})")

        # Exposure Program
        exp_program = self.raw_exif.get('ExposureProgram')
        if exp_program is not None:
            self.exif_data['exposure_program'] = self.EXPOSURE_PROGRAM_MAP.get(exp_program, f"Unknown ({exp_program})")

        # Exposure Compensation
        exp_comp = self.raw_exif.get('ExposureBiasValue')
        if exp_comp:
            if isinstance(exp_comp, tuple):
                self.exif_data['exposure_compensation'] = float(exp_comp[0]) / float(exp_comp[1])
            else:
                self.exif_data['exposure_compensation'] = float(exp_comp)

        # Flash
        flash = self.raw_exif.get('Flash')
        if flash is not None:
            self.exif_data['flash'] = bool(flash & 1)  # Bit 0: flash fired

        # Lens Model
        lens = self.raw_exif.get('LensModel')
        if lens:
            self.exif_data['lens_model'] = str(lens)

        # Camera Make & Model
        make = self.raw_exif.get('Make')
        model = self.raw_exif.get('Model')
        if make:
            self.exif_data['camera_make'] = str(make)
        if model:
            self.exif_data['camera_model'] = str(model)

        # DateTime
        dt = self.raw_exif.get('DateTimeOriginal') or self.raw_exif.get('DateTime')
        if dt:
            self.exif_data['datetime'] = str(dt)

        # 기타 설정
        for key, mapped_key in [
            ('Contrast', 'contrast'),
            ('Saturation', 'saturation'),
            ('Sharpness', 'sharpness'),
            ('MeteringMode', 'metering_mode'),
            ('ExposureMode', 'exposure_mode')
        ]:
            val = self.raw_exif.get(key)
            if val is not None:
                self.exif_data[mapped_key] = val

    def get_camera_settings(self) -> Dict[str, Any]:
        """
        주요 카메라 설정 반환

        Returns:
            {
                'iso': int,
                'f_number': float,
                'shutter_speed': float,
                'shutter_speed_display': str,
                'focal_length': float,
                'white_balance': str,
                'exposure_compensation': float,
                'flash': bool,
                ...
            }
        """
        return self.exif_data.copy()

    def get_shooting_info(self) -> Dict[str, str]:
        """
        사람이 읽기 쉬운 촬영 정보 반환

        Returns:
            {
                'camera': 'Canon EOS R5',
                'lens': 'RF 24-70mm F2.8 L IS USM',
                'settings': 'ISO 400, f/2.8, 1/200s, 50mm',
                'datetime': '2024-11-14 10:30:00'
            }
        """
        info = {}

        # 카메라
        if 'camera_make' in self.exif_data and 'camera_model' in self.exif_data:
            info['camera'] = f"{self.exif_data['camera_make']} {self.exif_data['camera_model']}"
        elif 'camera_model' in self.exif_data:
            info['camera'] = self.exif_data['camera_model']

        # 렌즈
        if 'lens_model' in self.exif_data:
            info['lens'] = self.exif_data['lens_model']

        # 설정
        settings_parts = []
        if 'iso' in self.exif_data:
            settings_parts.append(f"ISO {self.exif_data['iso']}")
        if 'f_number' in self.exif_data:
            settings_parts.append(f"f/{self.exif_data['f_number']:.1f}")
        if 'shutter_speed_display' in self.exif_data:
            settings_parts.append(self.exif_data['shutter_speed_display'])
        if 'focal_length' in self.exif_data:
            settings_parts.append(f"{self.exif_data['focal_length']:.0f}mm")

        if settings_parts:
            info['settings'] = ', '.join(settings_parts)

        # 날짜/시간
        if 'datetime' in self.exif_data:
            info['datetime'] = self.exif_data['datetime']

        # 추가 설정
        if 'white_balance' in self.exif_data:
            info['white_balance'] = self.exif_data['white_balance']
        if 'exposure_program' in self.exif_data:
            info['exposure_program'] = self.exif_data['exposure_program']
        if 'flash' in self.exif_data:
            info['flash'] = 'Used' if self.exif_data['flash'] else 'Not Used'

        return info

    def has_exif(self) -> bool:
        """EXIF 데이터가 있는지 확인"""
        return len(self.exif_data) > 0

    def __repr__(self):
        return f"<ExifAnalyzer: {len(self.exif_data)} fields>"


# ============================================================
# EXIF 비교 함수
# ============================================================

def compare_exif(ref_exif: Dict, user_exif: Dict) -> Dict:
    """
    레퍼런스 vs 사용자 EXIF 비교

    Returns:
        {
            'iso_diff': int,
            'f_number_diff': float,
            'shutter_speed_diff': float,
            'focal_length_diff': float,
            'white_balance_match': bool,
            'feedback': [...]
        }
    """
    comparison = {}
    feedback = []

    # ISO 비교
    if 'iso' in ref_exif and 'iso' in user_exif:
        iso_diff = user_exif['iso'] - ref_exif['iso']
        comparison['iso_diff'] = iso_diff

        if abs(iso_diff) > 100:
            if iso_diff > 0:
                feedback.append(f"ISO를 {abs(iso_diff)} 낮추세요 (현재 {user_exif['iso']} → 권장 {ref_exif['iso']})")
            else:
                feedback.append(f"ISO를 {abs(iso_diff)} 높이세요 (현재 {user_exif['iso']} → 권장 {ref_exif['iso']})")

    # 조리개 비교
    if 'f_number' in ref_exif and 'f_number' in user_exif:
        f_diff = user_exif['f_number'] - ref_exif['f_number']
        comparison['f_number_diff'] = f_diff

        if abs(f_diff) > 0.5:
            if f_diff > 0:
                feedback.append(f"조리개를 f/{ref_exif['f_number']:.1f}로 설정하세요 (현재 f/{user_exif['f_number']:.1f}, 더 열어야 함)")
            else:
                feedback.append(f"조리개를 f/{ref_exif['f_number']:.1f}로 설정하세요 (현재 f/{user_exif['f_number']:.1f}, 더 조여야 함)")

    # 셔터속도 비교
    if 'shutter_speed' in ref_exif and 'shutter_speed' in user_exif:
        ss_ratio = user_exif['shutter_speed'] / (ref_exif['shutter_speed'] + 1e-8)
        comparison['shutter_speed_ratio'] = ss_ratio

        if ss_ratio > 1.5 or ss_ratio < 0.67:
            ref_display = ref_exif.get('shutter_speed_display', f"{ref_exif['shutter_speed']}s")
            user_display = user_exif.get('shutter_speed_display', f"{user_exif['shutter_speed']}s")
            feedback.append(f"셔터속도를 {ref_display}로 설정하세요 (현재 {user_display})")

    # 초점거리 비교
    if 'focal_length' in ref_exif and 'focal_length' in user_exif:
        fl_diff = user_exif['focal_length'] - ref_exif['focal_length']
        comparison['focal_length_diff'] = fl_diff

        if abs(fl_diff) > 5:
            if fl_diff > 0:
                feedback.append(f"줌을 {abs(fl_diff):.0f}mm 줄이세요 (현재 {user_exif['focal_length']:.0f}mm → 권장 {ref_exif['focal_length']:.0f}mm)")
            else:
                feedback.append(f"줌을 {abs(fl_diff):.0f}mm 늘리세요 (현재 {user_exif['focal_length']:.0f}mm → 권장 {ref_exif['focal_length']:.0f}mm)")

    # 화이트밸런스 비교
    if 'white_balance' in ref_exif and 'white_balance' in user_exif:
        wb_match = ref_exif['white_balance'] == user_exif['white_balance']
        comparison['white_balance_match'] = wb_match

        if not wb_match:
            feedback.append(f"화이트밸런스를 '{ref_exif['white_balance']}'로 설정하세요 (현재 '{user_exif['white_balance']}')")

    # 노출 보정 비교
    if 'exposure_compensation' in ref_exif and 'exposure_compensation' in user_exif:
        ec_diff = user_exif['exposure_compensation'] - ref_exif['exposure_compensation']
        comparison['exposure_compensation_diff'] = ec_diff

        if abs(ec_diff) > 0.3:
            feedback.append(f"노출 보정을 {ref_exif['exposure_compensation']:+.1f} EV로 설정하세요 (현재 {user_exif['exposure_compensation']:+.1f} EV)")

    comparison['feedback'] = feedback
    comparison['has_differences'] = len(feedback) > 0

    return comparison


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"
    
    try:
        analyzer = ExifAnalyzer(str(test_img))

        print("\n" + "="*60)
        print("📷 EXIF ANALYSIS")
        print("="*60)

        if analyzer.has_exif():
            settings = analyzer.get_camera_settings()
            print("\n🔧 Camera Settings:")
            for key, value in settings.items():
                print(f"  {key}: {value}")

            print("\n📊 Shooting Info:")
            info = analyzer.get_shooting_info()
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("\n⚠️ No EXIF data found")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
