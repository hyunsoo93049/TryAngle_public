# Test pose analyzer
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "data").exists() and (PROJECT_ROOT / "src").exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

print("Testing PoseAnalyzer...")

try:
    from analysis.pose_analyzer import PoseAnalyzer
    print("✓ PoseAnalyzer imported successfully")

    # Test with image
    test_img = PROJECT_ROOT / "data" / "test_images" / "test1.jpg"

    print(f"\nAnalyzing: {test_img}")
    analyzer = PoseAnalyzer()
    print("✓ PoseAnalyzer initialized")

    result = analyzer.analyze(str(test_img))
    print("✓ Analysis completed")

    print(f"\nScenario: {result['scenario']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"YOLO keypoints: {len(result['yolo_keypoints'])}")

    if 'mediapipe_pose' in result and result['mediapipe_pose']:
        print(f"MediaPipe Pose: {result['mediapipe_pose']['count']} keypoints")

    if 'mediapipe_face' in result and result['mediapipe_face']:
        print(f"MediaPipe Face: {result['mediapipe_face']['count']} keypoints")

    if 'mediapipe_hands' in result and result['mediapipe_hands']:
        print(f"MediaPipe Hands: {result['mediapipe_hands']['hand_count']} hands")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
