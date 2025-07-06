# 기여 가이드

EasyMediaProcessor에 기여해주셔서 감사합니다! 

## 기여 방법

### 버그 리포트
1. [Issues](https://github.com/YourUsername/EasyMediaProcessor/issues) 페이지에서 새 이슈 생성
2. 버그 재현 단계 상세히 기술
3. 스크린샷 첨부 (가능한 경우)
4. 시스템 환경 정보 포함

### 기능 요청
1. [Issues](https://github.com/YourUsername/EasyMediaProcessor/issues) 페이지에서 기능 요청 이슈 생성
2. 기능의 필요성과 사용 사례 설명
3. 가능한 구현 방법 제안

### 코드 기여
1. Fork 프로젝트
2. 새 브랜치 생성: `git checkout -b feature/새기능명`
3. 코드 작성 및 테스트
4. 커밋: `git commit -m 'feat: 새 기능 추가'`
5. Push: `git push origin feature/새기능명`
6. Pull Request 생성

## 개발 환경 설정

```bash
git clone https://github.com/YourUsername/EasyMediaProcessor.git
cd EasyMediaProcessor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
