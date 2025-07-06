# Google Drive 연동 설정 가이드

## 1. Google Cloud Console에서 프로젝트 설정

### 1단계: Google Cloud Console 접속
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. Google 계정으로 로그인

### 2단계: 새 프로젝트 생성
1. 상단의 프로젝트 선택 드롭다운 클릭
2. "새 프로젝트" 클릭
3. 프로젝트 이름 입력 (예: "EasyMediaProcessor")
4. "만들기" 클릭

### 3단계: Google Drive API 활성화
1. 왼쪽 메뉴에서 "API 및 서비스" → "라이브러리" 클릭
2. "Google Drive API" 검색
3. "Google Drive API" 클릭
4. "사용" 버튼 클릭

## 2. OAuth 2.0 클라이언트 ID 생성

### 4단계: OAuth 동의 화면 구성
1. 왼쪽 메뉴에서 "API 및 서비스" → "OAuth 동의 화면" 클릭
2. "외부" 선택 후 "만들기" 클릭
3. 필수 정보 입력:
   - 앱 이름: "EasyMediaProcessor"
   - 사용자 지원 이메일: 본인 이메일
   - 개발자 연락처 정보: 본인 이메일
4. "저장 후 계속" 클릭
5. 범위 단계에서 "저장 후 계속" 클릭
6. 테스트 사용자에서 본인 이메일 추가 후 "저장 후 계속" 클릭

### 5단계: 클라이언트 ID 생성
1. 왼쪽 메뉴에서 "API 및 서비스" → "사용자 인증 정보" 클릭
2. "사용자 인증 정보 만들기" → "OAuth 클라이언트 ID" 클릭
3. 애플리케이션 유형: "데스크톱 애플리케이션" 선택
4. 이름: "EasyMediaProcessor Desktop" 입력
5. "만들기" 클릭

### 6단계: credentials.json 다운로드
1. 생성된 클라이언트 ID 옆의 다운로드 버튼 클릭
2. JSON 파일 다운로드
3. 파일 이름을 `credentials.json`으로 변경
4. **EasyMediaProcessor 프로그램 폴더**에 복사

## 3. 설치 및 설정

### 필수 패키지 설치
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 파일 구조
```
EasyMediaProcessor/
├── main.py
├── google_drive_manager.py
├── credentials.json          ← 여기에 복사!
├── token.json               ← 인증 후 자동 생성
└── requirements.txt
```

## 4. 사용법

### 첫 번째 인증
1. EasyMediaProcessor 실행
2. "파일" 메뉴 → "Google Drive 연동" 클릭
3. 웹 브라우저가 자동으로 열림
4. Google 계정으로 로그인
5. "EasyMediaProcessor가 Google Drive에 액세스하도록 허용하시겠습니까?" → "허용" 클릭
6. "인증이 완료되었습니다" 메시지 확인

### 기능 사용
- **파일 업로드**: "파일 업로드" 버튼으로 로컬 파일을 Google Drive에 업로드
- **파일 다운로드**: 목록에서 파일 선택 후 "파일 다운로드" 버튼 클릭
- **폴더 생성**: "폴더 생성" 버튼으로 새 폴더 생성
- **폴더 탐색**: 폴더를 더블클릭하여 내부로 이동

## 5. 문제 해결

### "credentials.json 파일이 없습니다" 오류
- credentials.json 파일이 프로그램과 같은 폴더에 있는지 확인
- 파일 이름이 정확한지 확인 (대소문자 구분)

### "인증 실패" 오류
- Google Cloud Console에서 OAuth 동의 화면이 올바르게 설정되었는지 확인
- 테스트 사용자에 본인 이메일이 추가되어 있는지 확인

### "API가 활성화되지 않음" 오류
- Google Drive API가 활성화되어 있는지 확인
- 올바른 프로젝트를 선택했는지 확인

### 토큰 만료 문제
- token.json 파일 삭제 후 다시 인증
- 브라우저 캐시 및 쿠키 삭제

## 6. 보안 주의사항

- `credentials.json` 파일을 다른 사람과 공유하지 마세요
- `token.json` 파일도 개인 정보이므로 보안에 주의하세요
- 프로덕션 환경에서는 OAuth 동의 화면을 "게시" 상태로 변경하세요

## 7. 제한사항

- Google Drive API는 무료로 하루 1억 요청까지 가능
- 파일 업로드/다운로드 속도는 인터넷 연결 상태에 따라 달라짐
- 대용량 파일(>2GB) 처리 시 시간이 오래 걸릴 수 있음