# EasyMediaProcessor 설치 및 사용 가이드

## 📋 요구사항 명세서 달성 현황

### ✅ 완료된 기능들
- **시중에서 흔히 사용되는 영상 및 이미지 포맷 처리** ✅
- **컬러 채널 분리 및 도메인 변환** ✅ 
- **영상 크기 조정 및 자르기** ✅
- **딥러닝 모델 테스트 및 객체 감지** ✅
- **구글 드라이브와의 파일 업로드/다운로드** ✅
- **직관적인 사용자 인터페이스와 최소 10개 이상의 기능 버튼** ✅
- **최소 10개 파일의 일괄 처리** ✅

## 🚀 설치 방법

### 1. 기본 설치 (필수)
```bash
# 1. 저장소 클론 또는 파일 다운로드
git clone <repository_url>
cd EasyMediaProcessor

# 2. 가상환경 생성
python -m venv venv

# 3. 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. 기본 패키지 설치
pip install PySide6 opencv-python Pillow ffmpeg-python numpy
```

### 2. 전체 기능 설치 (권장)
```bash
# 모든 기능을 사용하려면
pip install -r requirements.txt
```

### 3. 선택적 설치
```bash
# Google Drive 연동만 원하는 경우
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# AI 객체 감지만 원하는 경우  
pip install tensorflow

# 고급 처리 기능만 원하는 경우
pip install matplotlib

# 일괄 처리 기능만 원하는 경우
pip install opencv-python pillow ffmpeg-python matplotlib
```

## 📁 프로젝트 구조

```
EasyMediaProcessor/
├── main.py                     # 메인 애플리케이션
├── google_drive_manager.py     # Google Drive 연동
├── ai_detection.py            # AI 객체 감지
├── advanced_processing.py     # 고급 처리 기능
├── batch_processing.py        # 일괄 처리
├── requirements.txt           # 의존성 패키지
├── setup.py                  # 패키지 설정
├── build.py                  # 빌드 스크립트
├── credentials.json          # Google Drive 인증 (사용자가 추가)
├── token.json               # Google Drive 토큰 (자동 생성)
└── dist/                    # 빌드된 실행파일
    └── EasyMediaProcessor.exe
```

## 🎯 사용법

### 기본 실행
```bash
python main.py
```

### exe 파일 생성
```bash
python build.py
```

## 🔧 주요 기능 사용법

### 1. 기본 미디어 처리
- 드래그 앤 드롭으로 파일 선택
- 크기 조정, 포맷 변환 설정
- "처리 시작" 버튼 클릭

### 2. Google Drive 연동
1. [Google Drive 설정 가이드](Google_Drive_Setup_Guide.md) 참조