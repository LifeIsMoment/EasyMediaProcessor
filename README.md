# EasyMediaProcessor

<div align="center">

![EasyMediaProcessor Logo](https://img.shields.io/badge/EasyMediaProcessor-v1.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PySide6](https://img.shields.io/badge/PySide6-GUI-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**전문적인 영상 및 이미지 처리 도구**

[다운로드](#-다운로드) • [설치 가이드](#-설치-가이드) • [기능](#-주요-기능) • [사용법](#-사용법) • [문서](#-문서)

</div>

## 📋 개요

EasyMediaProcessor는 소규모 영상 처리 회사를 위한 올인원 미디어 처리 솔루션입니다. 직관적인 사용자 인터페이스와 강력한 기능으로 복잡한 영상/이미지 처리 작업을 간소화합니다.

### ✨ 하이라이트
- 🎯 **10개 이상의 전문 기능** - 원클릭으로 복잡한 작업 수행
- 🤖 **AI 객체 감지** - 딥러닝 기반 얼굴 및 객체 인식
- ☁️ **Google Drive 연동** - 클라우드 파일 관리
- 📁 **일괄 처리** - 10개 이상 파일 동시 처리
- 🎨 **고급 이미지 처리** - 채널 분리, 색상 변환, 필터
- 🚀 **exe 배포** - Python 설치 없이 즉시 실행

## 🚀 다운로드

### 최신 릴리스 (v1.0.0)

**Windows 사용자 (권장)**
- [EasyMediaProcessor.exe](https://github.com/YourUsername/EasyMediaProcessor/releases/latest/download/EasyMediaProcessor.exe) (약 100MB)
- Python 설치 불필요, 즉시 실행 가능

**개발자/소스코드**
- [소스코드 ZIP](https://github.com/YourUsername/EasyMediaProcessor/archive/refs/heads/main.zip)
- Python 3.8+ 필요

## 🎯 주요 기능

### 📸 기본 미디어 처리
- **포맷 변환**: MP4 ↔ AVI ↔ MOV ↔ MKV | JPG ↔ PNG ↔ BMP
- **크기 조정**: 표준 해상도 (4K, FHD, HD) 또는 사용자 정의
- **비디오 자르기**: 시간 범위 또는 마우스 드래그로 정확한 편집
- **드래그 앤 드롭**: 직관적인 파일 선택

### 🤖 AI 기반 기능
- **얼굴 감지**: OpenCV Haar Cascade 알고리즘
- **객체 감지**: TensorFlow/Keras 모델 지원
- **엣지 감지**: Canny 알고리즘 기반 윤곽 추출
- **커스텀 모델**: 사용자 정의 AI 모델 로드 가능

### 🎨 고급 이미지 처리
- **채널 분리**: RGB, HSV, HSI 채널별 분석
- **색상 도메인 변환**: RGB ↔ HSV ↔ HSI ↔ LAB
- **이미지 향상**: 밝기, 대비, 채도, 선명도 조정
- **필터 적용**: 블러, 샤픈, 엣지, 엠보싱
- **노이즈 감소**: Bilateral, Gaussian, Median 필터
- **업스케일링**: 고품질 이미지 확대
- **히스토그램 분석**: 색상 분포 및 통계 분석

### ☁️ 클라우드 연동
- **Google Drive API**: 파일 업로드/다운로드
- **폴더 관리**: 클라우드 폴더 생성 및 탐색
- **진행률 표시**: 실시간 업로드/다운로드 상태
- **자동 인증**: OAuth 2.0 보안 인증

### 📁 일괄 처리
- **대용량 처리**: 최소 10개 이상 파일 동시 처리
- **진행률 모니터링**: 전체 및 파일별 진행률 실시간 표시
- **오류 처리**: 실패한 파일 건너뛰고 계속 진행
- **커스터마이징**: 파일명 접두사, 출력 구조 설정

## 🖥️ 시스템 요구사항

### 최소 사양
- **OS**: Windows 10/11 (64-bit)
- **CPU**: Intel i3 또는 동급
- **RAM**: 4GB
- **저장공간**: 200MB 여유 공간

### 권장 사양
- **OS**: Windows 11 (64-bit)
- **CPU**: Intel i5 또는 동급
- **RAM**: 8GB
- **저장공간**: 1GB 여유 공간
- **GPU**: 옵션 (AI 기능 가속용)

## 💻 설치 가이드

### 방법 1: exe 파일 실행 (일반 사용자)
1. [EasyMediaProcessor.exe](https://github.com/YourUsername/EasyMediaProcessor/releases/latest) 다운로드
2. 다운로드한 파일 더블클릭
3. Windows Defender 경고 시 "추가 정보" → "실행" 클릭
4. 즉시 사용 가능! 🎉

### 방법 2: 소스코드 설치 (개발자)
```bash
# 1. 저장소 클론
git clone https://github.com/YourUsername/EasyMediaProcessor.git
cd EasyMediaProcessor

# 2. 가상환경 생성
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python main.py
```

## 🎮 사용법

### 기본 사용법
1. **파일 선택**: 드래그 앤 드롭 또는 "파일 열기" 버튼
2. **옵션 설정**: 크기, 포맷, 품질 등 원하는 설정
3. **출력 폴더 선택**: 처리된 파일을 저장할 위치
4. **처리 시작**: 버튼 클릭으로 변환 시작

### 고급 기능
- **AI 객체 감지**: 메뉴 → 도구 → AI 객체 감지
- **채널 분리**: 메뉴 → 도구 → 고급 처리 → 채널 분리
- **일괄 처리**: 메뉴 → 도구 → 일괄 처리
- **Google Drive**: 메뉴 → 파일 → Google Drive 연동

### 단축키
- `Ctrl + O`: 파일 열기
- `Ctrl + D`: Google Drive 연동
- `Ctrl + I`: AI 객체 감지
- `Ctrl + A`: 고급 처리
- `Ctrl + B`: 일괄 처리
- `Ctrl + Q`: 프로그램 종료

## 📚 문서

- [Google Drive 설정 가이드](docs/Google_Drive_Setup_Guide.md)
- [설치 및 사용 가이드](docs/INSTALLATION_GUIDE.md)
- [API 문서](docs/API_DOCUMENTATION.md)
- [문제 해결](docs/TROUBLESHOOTING.md)

## 🔧 개발 정보

### 기술 스택
- **GUI Framework**: PySide6 (Qt6)
- **이미지 처리**: OpenCV, Pillow
- **비디오 처리**: FFmpeg-python
- **AI/ML**: TensorFlow, Keras
- **클라우드**: Google Drive API
- **그래프**: Matplotlib
- **패키징**: PyInstaller

### 프로젝트 구조
```
EasyMediaProcessor/
├── main.py                     # 메인 애플리케이션
├── google_drive_manager.py     # Google Drive 연동
├── ai_detection.py            # AI 객체 감지
├── advanced_processing.py     # 고급 처리 기능
├── batch_processing.py        # 일괄 처리
├── build.py                   # 빌드 스크립트
├── requirements.txt           # 의존성 패키지
├── setup.py                  # 패키지 설정
├── docs/                     # 문서
├── dist/                     # 빌드된 실행파일
└── README.md                 # 프로젝트 설명
```

## 🤝 기여하기

EasyMediaProcessor는 오픈소스 프로젝트입니다. 기여를 환영합니다!

### 기여 방법
1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 열기

### 개발 환경 설정
```bash
git clone https://github.com/YourUsername/EasyMediaProcessor.git
cd EasyMediaProcessor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발용 의존성
```

## 🐛 버그 리포트 & 기능 요청

- [Issues](https://github.com/YourUsername/EasyMediaProcessor/issues) 페이지에서 버그 리포트 및 기능 요청
- [Discussions](https://github.com/YourUsername/EasyMediaProcessor/discussions) 페이지에서 질문 및 토론

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👥 팀

- **개발자**: YourName ([@YourUsername](https://github.com/YourUsername))
- **이메일**: your.email@example.com

## 🙏 감사의 말

- [PySide6](https://doc.qt.io/qtforpython/) - GUI 프레임워크
- [OpenCV](https://opencv.org/) - 컴퓨터 비전
- [TensorFlow](https://tensorflow.org/) - 머신러닝
- [Google Drive API](https://developers.google.com/drive/) - 클라우드 연동

## 📊 릴리스 노트

### v1.0.0 (2024-01-XX)
- 🎉 초기 릴리스
- ✅ 기본 미디어 처리 기능
- ✅ AI 객체 감지
- ✅ Google Drive 연동
- ✅ 고급 이미지 처리
- ✅ 일괄 처리
- ✅ exe 파일 배포

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐**

[🔝 맨 위로](#easymediaprocessor)

</div>