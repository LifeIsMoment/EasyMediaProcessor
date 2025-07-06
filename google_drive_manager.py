import os
import io
import json
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QListWidget, QProgressBar, QMessageBox,
                              QInputDialog, QFileDialog)

# Google Drive API 스코프
SCOPES = ['https://www.googleapis.com/auth/drive']

class GoogleDriveAuth:
    """Google Drive 인증 관리"""
    
    def __init__(self):
        self.creds = None
        self.service = None
        self.credentials_file = 'credentials.json'
        self.token_file = 'token.json'
    
    def authenticate(self):
        """Google Drive 인증"""
        # 기존 토큰 파일이 있으면 로드
        if os.path.exists(self.token_file):
            self.creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # 토큰이 유효하지 않으면 새로 인증
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    return False, "credentials.json 파일이 없습니다. Google Cloud Console에서 다운로드하세요."
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # 토큰 저장
            with open(self.token_file, 'w') as token:
                token.write(self.creds.to_json())
        
        # Drive API 서비스 생성
        self.service = build('drive', 'v3', credentials=self.creds)
        return True, "인증 성공"
    
    def is_authenticated(self):
        """인증 상태 확인"""
        return self.service is not None


class DriveFileUploader(QThread):
    """파일 업로드 스레드"""
    progress = Signal(int)
    finished = Signal(str, str)  # file_id, message
    error = Signal(str)
    
    def __init__(self, service, file_path, folder_id=None):
        super().__init__()
        self.service = service
        self.file_path = file_path
        self.folder_id = folder_id
    
    def run(self):
        try:
            file_name = Path(self.file_path).name
            
            # 파일 메타데이터
            metadata = {'name': file_name}
            if self.folder_id:
                metadata['parents'] = [self.folder_id]
            
            # 미디어 업로드
            media = MediaFileUpload(
                self.file_path,
                resumable=True,
                chunksize=1024*1024  # 1MB chunks
            )
            
            # 업로드 실행
            request = self.service.files().create(
                body=metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    self.progress.emit(progress)
            
            file_id = response.get('id')
            self.finished.emit(file_id, f"업로드 완료: {file_name}")
            
        except Exception as e:
            self.error.emit(f"업로드 실패: {str(e)}")


class DriveFileDownloader(QThread):
    """파일 다운로드 스레드"""
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, service, file_id, save_path):
        super().__init__()
        self.service = service
        self.file_id = file_id
        self.save_path = save_path
    
    def run(self):
        try:
            request = self.service.files().get_media(fileId=self.file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    self.progress.emit(progress)
            
            # 파일 저장
            with open(self.save_path, 'wb') as f:
                file_io.seek(0)
                f.write(file_io.read())
            
            self.finished.emit(f"다운로드 완료: {self.save_path}")
            
        except Exception as e:
            self.error.emit(f"다운로드 실패: {str(e)}")


class GoogleDriveDialog(QDialog):
    """Google Drive 파일 관리 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.auth = GoogleDriveAuth()
        self.current_folder_id = None
        self.files_data = []
        
        self.setWindowTitle("Google Drive 연동")
        self.setMinimumSize(600, 500)
        
        self.setup_ui()
        self.authenticate_drive()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 인증 상태
        self.auth_label = QLabel("인증 중...")
        layout.addWidget(self.auth_label)
        
        # 컨트롤 버튼들
        control_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh_files)
        control_layout.addWidget(self.refresh_btn)
        
        self.upload_btn = QPushButton("파일 업로드")
        self.upload_btn.clicked.connect(self.upload_file)
        control_layout.addWidget(self.upload_btn)
        
        self.download_btn = QPushButton("파일 다운로드")
        self.download_btn.clicked.connect(self.download_file)
        control_layout.addWidget(self.download_btn)
        
        self.create_folder_btn = QPushButton("폴더 생성")
        self.create_folder_btn.clicked.connect(self.create_folder)
        control_layout.addWidget(self.create_folder_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 파일 목록
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.file_list)
        
        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 상태 라벨
        self.status_label = QLabel("준비됨")
        layout.addWidget(self.status_label)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def authenticate_drive(self):
        """Google Drive 인증"""
        success, message = self.auth.authenticate()
        
        if success:
            self.auth_label.setText("✅ Google Drive 연결됨")
            self.auth_label.setStyleSheet("color: green;")
            self.refresh_files()
        else:
            self.auth_label.setText(f"❌ 인증 실패: {message}")
            self.auth_label.setStyleSheet("color: red;")
            
            # credentials.json 파일 안내
            QMessageBox.information(
                self, "Google Drive 인증",
                "Google Drive 연동을 위해 다음 단계를 수행하세요:\n\n"
                "1. Google Cloud Console (console.cloud.google.com) 접속\n"
                "2. 새 프로젝트 생성 또는 기존 프로젝트 선택\n"
                "3. Google Drive API 활성화\n"
                "4. 사용자 인증 정보 → OAuth 2.0 클라이언트 ID 생성\n"
                "5. credentials.json 파일을 프로그램 폴더에 저장"
            )
    
    def refresh_files(self):
        """파일 목록 새로고침"""
        if not self.auth.is_authenticated():
            return
        
        try:
            self.status_label.setText("파일 목록 로딩 중...")
            
            # 파일 목록 쿼리
            query = f"'{self.current_folder_id}' in parents" if self.current_folder_id else "'root' in parents"
            query += " and trashed=false"
            
            results = self.auth.service.files().list(
                q=query,
                pageSize=50,
                fields="files(id, name, mimeType, size, modifiedTime)",
                orderBy="folder,name"
            ).execute()
            
            files = results.get('files', [])
            self.files_data = files
            
            # 파일 목록 업데이트
            self.file_list.clear()
            
            # 상위 폴더로 가기 (루트가 아닌 경우)
            if self.current_folder_id:
                self.file_list.addItem("📁 .. (상위 폴더)")
            
            for file in files:
                icon = "📁" if file['mimeType'] == 'application/vnd.google-apps.folder' else "📄"
                size_text = ""
                if 'size' in file:
                    size_mb = int(file['size']) / (1024 * 1024)
                    size_text = f" ({size_mb:.1f} MB)"
                
                item_text = f"{icon} {file['name']}{size_text}"
                self.file_list.addItem(item_text)
            
            self.status_label.setText(f"{len(files)}개 파일/폴더")
            
        except Exception as e:
            self.status_label.setText(f"오류: {str(e)}")
    
    def on_item_double_clicked(self, item):
        """아이템 더블클릭 처리"""
        row = self.file_list.row(item)
        
        # 상위 폴더로 가기
        if item.text().startswith("📁 .. "):
            self.current_folder_id = None  # 간단히 루트로 이동
            self.refresh_files()
            return
        
        # 상위 폴더 항목이 있으면 인덱스 조정
        file_index = row - (1 if self.current_folder_id else 0)
        
        if 0 <= file_index < len(self.files_data):
            file = self.files_data[file_index]
            
            # 폴더인 경우 들어가기
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                self.current_folder_id = file['id']
                self.refresh_files()
    
    def upload_file(self):
        """파일 업로드"""
        if not self.auth.is_authenticated():
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "업로드할 파일 선택", "",
            "모든 파일 (*)"
        )
        
        if not file_path:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("업로드 중...")
        
        self.uploader = DriveFileUploader(
            self.auth.service, file_path, self.current_folder_id
        )
        self.uploader.progress.connect(self.progress_bar.setValue)
        self.uploader.finished.connect(self.on_upload_finished)
        self.uploader.error.connect(self.on_upload_error)
        self.uploader.start()
    
    def download_file(self):
        """선택된 파일 다운로드"""
        if not self.auth.is_authenticated():
            return
        
        current_row = self.file_list.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "알림", "다운로드할 파일을 선택하세요.")
            return
        
        # 상위 폴더 항목이 있으면 인덱스 조정
        file_index = current_row - (1 if self.current_folder_id else 0)
        
        if 0 <= file_index < len(self.files_data):
            file = self.files_data[file_index]
            
            # 폴더는 다운로드 불가
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                QMessageBox.information(self, "알림", "폴더는 다운로드할 수 없습니다.")
                return
            
            # 저장 위치 선택
            save_path, _ = QFileDialog.getSaveFileName(
                self, "저장 위치", file['name']
            )
            
            if not save_path:
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("다운로드 중...")
            
            self.downloader = DriveFileDownloader(
                self.auth.service, file['id'], save_path
            )
            self.downloader.progress.connect(self.progress_bar.setValue)
            self.downloader.finished.connect(self.on_download_finished)
            self.downloader.error.connect(self.on_download_error)
            self.downloader.start()
    
    def create_folder(self):
        """새 폴더 생성"""
        if not self.auth.is_authenticated():
            return
        
        folder_name, ok = QInputDialog.getText(self, "폴더 생성", "폴더 이름:")
        if not ok or not folder_name.strip():
            return
        
        try:
            metadata = {
                'name': folder_name.strip(),
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if self.current_folder_id:
                metadata['parents'] = [self.current_folder_id]
            
            self.auth.service.files().create(body=metadata).execute()
            self.status_label.setText(f"폴더 생성됨: {folder_name}")
            self.refresh_files()
            
        except Exception as e:
            self.status_label.setText(f"폴더 생성 실패: {str(e)}")
    
    def on_upload_finished(self, file_id, message):
        """업로드 완료"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        self.refresh_files()
    
    def on_upload_error(self, error):
        """업로드 오류"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(error)
    
    def on_download_finished(self, message):
        """다운로드 완료"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
    
    def on_download_error(self, error):
        """다운로드 오류"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(error)