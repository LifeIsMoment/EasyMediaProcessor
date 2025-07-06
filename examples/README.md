# Examples - 샘플 파일들

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
