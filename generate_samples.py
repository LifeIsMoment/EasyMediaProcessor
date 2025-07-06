"""
EasyMediaProcessor 샘플 파일 생성 스크립트
다양한 테스트용 이미지와 영상을 생성합니다.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path

def create_examples_folder():
    """examples 폴더 생성"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    return examples_dir

def generate_sample_images(examples_dir):
    """다양한 샘플 이미지 생성"""
    
    # 1. 컬러 그라디언트 이미지 (RGB 채널 분리 테스트용)
    print("🎨 컬러 그라디언트 이미지 생성 중...")
    width, height = 800, 600
    
    # RGB 채널별 그라디언트
    r_gradient = np.linspace(0, 255, width).reshape(1, -1).repeat(height, axis=0)
    g_gradient = np.linspace(0, 255, height).reshape(-1, 1).repeat(width, axis=1)
    b_gradient = np.full((height, width), 128)
    
    rgb_image = np.stack([r_gradient, g_gradient, b_gradient], axis=2).astype(np.uint8)
    cv2.imwrite(str(examples_dir / "sample_rgb_gradient.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    # 2. 기하학적 패턴 이미지 (필터 테스트용)
    print("📐 기하학적 패턴 이미지 생성 중...")
    pattern_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 원형 패턴
    center = (400, 300)
    for radius in range(50, 250, 30):
        color = (radius % 255, (radius * 2) % 255, (radius * 3) % 255)
        cv2.circle(pattern_img, center, radius, color, 15)
    
    # 직선 패턴
    for i in range(0, 800, 50):
        cv2.line(pattern_img, (i, 0), (i, 600), (255, 255, 0), 2)
    for i in range(0, 600, 50):
        cv2.line(pattern_img, (0, i), (800, i), (0, 255, 255), 2)
    
    cv2.imwrite(str(examples_dir / "sample_geometric_pattern.png"), pattern_img)
    
    # 3. 텍스트가 있는 이미지 (OCR/객체 감지 테스트용)
    print("📝 텍스트 이미지 생성 중...")
    text_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(text_img)
    
    # 기본 폰트 사용
    try:
        # Windows 시스템 폰트 시도
        font_large = ImageFont.truetype("arial.ttf", 48)
        font_medium = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        # 기본 폰트로 대체
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 다양한 크기와 색상의 텍스트
    draw.text((50, 50), "EasyMediaProcessor", font=font_large, fill='black')
    draw.text((50, 120), "AI Object Detection Test", font=font_medium, fill='blue')
    draw.text((50, 170), "Text Recognition Sample", font=font_medium, fill='red')
    draw.text((50, 220), "Multiple Language Support", font=font_small, fill='green')
    draw.text((50, 250), "한글 텍스트 테스트", font=font_small, fill='purple')
    draw.text((50, 280), "日本語テストテキスト", font=font_small, fill='orange')
    
    # 간단한 도형들 추가
    draw.rectangle([50, 320, 250, 420], outline='black', width=3)
    draw.ellipse([300, 320, 500, 420], outline='blue', width=3)
    draw.polygon([(550, 320), (650, 320), (600, 420)], outline='red', width=3)
    
    text_img.save(examples_dir / "sample_text_detection.png")
    
    # 4. 얼굴 시뮬레이션 이미지 (얼굴 감지 테스트용)
    print("👤 얼굴 시뮬레이션 이미지 생성 중...")
    face_img = Image.new('RGB', (400, 400), color='lightblue')
    draw = ImageDraw.Draw(face_img)
    
    # 간단한 얼굴 그리기
    # 머리 윤곽
    draw.ellipse([50, 50, 350, 320], fill='peachpuff', outline='black', width=2)
    
    # 눈
    draw.ellipse([100, 120, 140, 150], fill='white', outline='black', width=2)
    draw.ellipse([260, 120, 300, 150], fill='white', outline='black', width=2)
    draw.ellipse([110, 125, 130, 145], fill='black')  # 왼쪽 눈동자
    draw.ellipse([270, 125, 290, 145], fill='black')  # 오른쪽 눈동자
    
    # 코
    draw.polygon([(200, 160), (190, 200), (210, 200)], fill='pink', outline='black')
    
    # 입
    draw.arc([150, 220, 250, 280], 0, 180, fill='red', width=5)
    
    # 머리카락
    draw.ellipse([60, 40, 340, 150], fill='brown', outline='black', width=2)
    draw.ellipse([50, 50, 350, 120], fill='lightblue')  # 얼굴 부분 다시 그리기
    
    face_img.save(examples_dir / "sample_face_detection.png")
    
    # 5. 노이즈가 있는 이미지 (노이즈 제거 테스트용)
    print("📡 노이즈 이미지 생성 중...")
    clean_img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 깔끔한 패턴 생성
    for i in range(0, 400, 40):
        cv2.rectangle(clean_img, (i, i), (i+30, i+30), (0, 255, 0), -1)
        cv2.rectangle(clean_img, (i+10, i+10), (i+20, i+20), (255, 0, 0), -1)
    
    # 노이즈 추가
    noise = np.random.normal(0, 30, clean_img.shape).astype(np.int16)
    noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(examples_dir / "sample_clean_image.png"), clean_img)
    cv2.imwrite(str(examples_dir / "sample_noisy_image.png"), noisy_img)
    
    # 6. 다양한 해상도 이미지들
    print("📏 다양한 해상도 이미지 생성 중...")
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
        (320, 240, "QVGA")
    ]
    
    for width, height, name in resolutions:
        # 무지개 그라디언트
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for x in range(width):
            hue = int(180 * x / width)  # HSV의 Hue 값
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            img[:, x] = rgb_color
        
        cv2.imwrite(str(examples_dir / f"sample_resolution_{name}_{width}x{height}.jpg"), 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def generate_sample_videos(examples_dir):
    """샘플 비디오 생성"""
    print("🎬 샘플 비디오 생성 중...")
    
    # 1. 간단한 애니메이션 비디오 (10초)
    video_path = examples_dir / "sample_animation.mp4"
    
    # 비디오 설정
    fps = 30
    duration = 10  # 초
    width, height = 640, 480
    
    # VideoWriter 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_num in range(total_frames):
        # 배경 생성
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 시간에 따른 색상 변화
        hue = int(180 * frame_num / total_frames)
        bg_color = cv2.cvtColor(np.array([[[hue, 100, 100]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        frame[:, :] = bg_color
        
        # 움직이는 원
        circle_x = int(width * (frame_num / total_frames))
        circle_y = int(height / 2 + 100 * np.sin(2 * np.pi * frame_num / (fps * 2)))
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 255, 255), -1)
        
        # 회전하는 사각형
        center = (width // 2, height // 2)
        angle = 360 * frame_num / total_frames
        
        # 사각형 꼭짓점 계산
        size = 50
        points = np.array([
            [-size, -size], [size, -size], [size, size], [-size, size]
        ], dtype=np.float32)
        
        # 회전 변환
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_points = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1, 2)
        rotated_points += np.array([center[0], center[1]])
        
        cv2.fillPoly(frame, [rotated_points.astype(np.int32)], (0, 255, 255))
        
        # 프레임 번호 텍스트
        cv2.putText(frame, f"Frame: {frame_num+1}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 시간 표시
        time_text = f"Time: {frame_num/fps:.2f}s"
        cv2.putText(frame, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        # 진행률 표시
        if frame_num % (total_frames // 10) == 0:
            print(f"  비디오 생성 진행률: {frame_num/total_frames*100:.0f}%")
    
    out.release()
    print("✅ 샘플 애니메이션 비디오 생성 완료")
    
    # 2. 정적 테스트 비디오 (5초, 더 간단함)
    print("🎥 정적 테스트 비디오 생성 중...")
    static_video_path = examples_dir / "sample_static_test.mp4"
    
    out2 = cv2.VideoWriter(str(static_video_path), fourcc, fps, (320, 240))
    
    # 정적 프레임 생성
    static_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # 체스보드 패턴
    square_size = 40
    for i in range(0, 240, square_size):
        for j in range(0, 320, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                static_frame[i:i+square_size, j:j+square_size] = [255, 255, 255]
            else:
                static_frame[i:i+square_size, j:j+square_size] = [0, 0, 0]
    
    # 5초간 같은 프레임 반복
    for _ in range(fps * 5):
        out2.write(static_frame)
    
    out2.release()
    print("✅ 정적 테스트 비디오 생성 완료")

def generate_histogram_sample(examples_dir):
    """히스토그램 분석용 샘플 이미지 생성"""
    print("📊 히스토그램 분석용 이미지 생성 중...")
    
    # 다양한 밝기 분포를 가진 이미지들
    samples = [
        ("dark", lambda: np.random.gamma(0.5, 50, (400, 400, 3))),
        ("bright", lambda: np.random.gamma(2.0, 100, (400, 400, 3))),
        ("normal", lambda: np.random.normal(128, 50, (400, 400, 3))),
        ("bimodal", lambda: np.concatenate([
            np.random.normal(80, 30, (400, 200, 3)),
            np.random.normal(180, 30, (400, 200, 3))
        ], axis=1))
    ]
    
    for name, generator in samples:
        img = np.clip(generator(), 0, 255).astype(np.uint8)
        cv2.imwrite(str(examples_dir / f"sample_histogram_{name}.png"), img)

def create_readme_for_examples(examples_dir):
    """examples 폴더용 README 생성"""
    readme_content = """# Examples - 샘플 파일들

이 폴더에는 EasyMediaProcessor의 다양한 기능을 테스트할 수 있는 샘플 파일들이 포함되어 있습니다.

## 📸 이미지 파일들

### 기본 테스트 이미지
- `sample_rgb_gradient.jpg` - RGB 채널 분리 테스트용 그라디언트 이미지
- `sample_geometric_pattern.png` - 필터 및 엣지 감지 테스트용 기하학적 패턴
- `sample_text_detection.png` - OCR 및 텍스트 감지 테스트용 이미지
- `sample_face_detection.png` - 얼굴 감지 알고리즘 테스트용 이미지

### 이미지 처리 테스트
- `sample_clean_image.png` - 깔끔한 원본 이미지
- `sample_noisy_image.png` - 노이즈가 추가된 이미지 (노이즈 제거 테스트용)

### 해상도 테스트
- `sample_resolution_VGA_640x480.jpg` - VGA 해상도
- `sample_resolution_HD_1280x720.jpg` - HD 해상도  
- `sample_resolution_FHD_1920x1080.jpg` - Full HD 해상도
- `sample_resolution_QVGA_320x240.jpg` - QVGA 해상도

### 히스토그램 분석용
- `sample_histogram_dark.png` - 어두운 이미지
- `sample_histogram_bright.png` - 밝은 이미지
- `sample_histogram_normal.png` - 일반적인 밝기 분포
- `sample_histogram_bimodal.png` - 이중 분포 이미지

## 🎬 비디오 파일들

- `sample_animation.mp4` - 10초 애니메이션 비디오 (움직이는 도형, 색상 변화)
- `sample_static_test.mp4` - 5초 정적 비디오 (체스보드 패턴)

## 🎯 사용법

### 1. 기본 기능 테스트
1. EasyMediaProcessor 실행
2. 샘플 이미지를 드래그 앤 드롭
3. 크기 조정, 포맷 변환 등 테스트

### 2. AI 객체 감지 테스트
1. `sample_face_detection.png` 사용
2. 메뉴 → 도구 → AI 객체 감지
3. 얼굴 감지 결과 확인

### 3. 고급 처리 테스트
1. `sample_rgb_gradient.jpg` 사용
2. 메뉴 → 도구 → 고급 처리
3. 채널 분리, 색상 변환 테스트

### 4. 일괄 처리 테스트
1. 여러 샘플 파일 선택
2. 메뉴 → 도구 → 일괄 처리
3. 10개 이상 파일 동시 처리 테스트

### 5. 히스토그램 분석 테스트
1. 히스토그램 샘플 이미지들 사용
2. 고급 처리 → 분석 탭
3. 히스토그램 생성 및 분석

## 📝 참고사항

- 모든 샘플 파일은 Python 스크립트로 생성되었습니다
- 실제 사진이나 저작권이 있는 콘텐츠는 포함되지 않았습니다
- 테스트 목적으로만 사용해주세요
- 새로운 샘플이 필요하면 `generate_samples.py` 스크립트를 실행하세요

## 🔄 샘플 재생성

```bash
python generate_samples.py
```

이 명령어로 모든 샘플 파일을 다시 생성할 수 있습니다.
"""
    
    with open(examples_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    """메인 함수"""
    print("🚀 EasyMediaProcessor 샘플 파일 생성 시작")
    print("=" * 50)
    
    # examples 폴더 생성
    examples_dir = create_examples_folder()
    print(f"📁 Examples 폴더 생성: {examples_dir}")
    
    try:
        # 샘플 이미지들 생성
        generate_sample_images(examples_dir)
        
        # 히스토그램 분석용 이미지 생성
        generate_histogram_sample(examples_dir)
        
        # 샘플 비디오 생성
        generate_sample_videos(examples_dir)
        
        # README 파일 생성
        create_readme_for_examples(examples_dir)
        
        print("\n" + "=" * 50)
        print("✅ 모든 샘플 파일 생성 완료!")
        print(f"📂 생성된 파일들은 {examples_dir} 폴더에서 확인하세요.")
        
        # 생성된 파일 목록 출력
        print("\n📋 생성된 파일들:")
        for file_path in sorted(examples_dir.glob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"  📄 {file_path.name} ({size_str})")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()