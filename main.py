import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                              QProgressBar, QTextEdit, QGroupBox, QGridLayout,
                              QComboBox, QSpinBox, QCheckBox, QTabWidget,
                              QListWidget, QSplitter, QFrame, QMenuBar, QMenu)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QPixmap, QDragEnterEvent, QDropEvent, QAction
import cv2
import numpy as np
from PIL import Image
import ffmpeg

# 구글 드라이브 모듈 import
try:
    from google_drive_manager import GoogleDriveDialog
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# AI 객체 감지 모듈 import
try:
    from ai_detection import AIDetectionDialog
    AI_DETECTION_AVAILABLE = True
except ImportError:
    AI_DETECTION_AVAILABLE = False

# 고급 처리 모듈 import
try:
    from advanced_processing import AdvancedProcessingDialog
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False

# 일괄 처리 모듈 import
try:
    from batch_processing import BatchProcessingDialog
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False


class MediaProcessor(QThread):
    """미디어 처리 작업을 별도 스레드에서 실행"""
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, file_path, output_path, options):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.options = options
    
    def run(self):
        try:
            if self.file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.process_video()
            elif self.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.process_image()
            else:
                self.error.emit("지원하지 않는 파일 형식입니다.")
        except Exception as e:
            self.error.emit(f"처리 중 오류 발생: {str(e)}")
    
    def process_video(self):
        """비디오 처리"""
        try:
            stream = ffmpeg.input(self.file_path)
            
            if self.options.get('resize'):
                width = self.options.get('width', 1920)
                height = self.options.get('height', 1080)
                stream = ffmpeg.filter(stream, 'scale', width, height)
            
            if self.options.get('format_change'):
                output_format = self.options.get('output_format', 'mp4')
                output_path = self.output_path.replace('.mp4', f'.{output_format}')
            else:
                output_path = self.output_path
            
            for i in range(0, 101, 10):
                self.progress.emit(i)
                self.msleep(100)
            
            stream = ffmpeg.output(stream, output_path)
            ffmpeg.run(stream, overwrite_output=True)
            
            self.finished.emit(f"비디오 처리 완료: {output_path}")
            
        except Exception as e:
            self.error.emit(f"비디오 처리 실패: {str(e)}")
    
    def process_image(self):
        """이미지 처리"""
        try:
            img = Image.open(self.file_path)
            
            if self.options.get('resize'):
                width = self.options.get('width', 1920)
                height = self.options.get('height', 1080)
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            if self.options.get('format_change'):
                output_format = self.options.get('output_format', 'PNG')
                output_path = self.output_path.replace('.png', f'.{output_format.lower()}')
            else:
                output_path = self.output_path
            
            for i in range(0, 101, 20):
                self.progress.emit(i)
                self.msleep(50)
            
            img.save(output_path)
            
            self.finished.emit(f"이미지 처리 완료: {output_path}")
            
        except Exception as e:
            self.error.emit(f"이미지 처리 실패: {str(e)}")


class DropZone(QFrame):
    """드래그 앤 드롭 영역"""
    files_dropped = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
            QFrame:hover {
                border-color: #4CAF50;
                background-color: #e8f5e8;
            }
        """)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("파일을 여기로 드래그하거나 클릭하여 선택하세요")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #666; font-size: 14px;")
        
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                files.append(url.toLocalFile())
        
        if files:
            self.files_dropped.emit(files)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(
                self, "파일 선택", "", 
                "미디어 파일 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;모든 파일 (*)"
            )
            if files:
                self.files_dropped.emit(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EasyMediaProcessor v1.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메뉴바 생성
        self.create_menu_bar()
        
        # 메인 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)
        
        # 스플리터로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 왼쪽 패널 (파일 선택 및 옵션)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 오른쪽 패널 (결과 및 로그)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 스플리터 비율 설정
        splitter.setSizes([600, 600])
        
        # 상태 표시줄
        self.statusBar().showMessage("준비됨")
        
        # 변수 초기화
        self.current_files = []
        self.processor = None
        self.google_drive_dialog = None
        self.ai_detection_dialog = None
        self.advanced_processing_dialog = None
        self.batch_processing_dialog = None
    
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu('파일')
        
        # 파일 열기
        open_action = QAction('파일 열기', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Google Drive 연동
        if GOOGLE_DRIVE_AVAILABLE:
            drive_action = QAction('Google Drive 연동', self)
            drive_action.setShortcut('Ctrl+D')
            drive_action.triggered.connect(self.open_google_drive)
            file_menu.addAction(drive_action)
        else:
            drive_action = QAction('Google Drive 연동 (설치 필요)', self)
            drive_action.setEnabled(False)
            file_menu.addAction(drive_action)
        
        file_menu.addSeparator()
        
        # 종료
        exit_action = QAction('종료', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도구 메뉴
        tools_menu = menubar.addMenu('도구')
        
        # AI 객체 감지
        if AI_DETECTION_AVAILABLE:
            ai_action = QAction('AI 객체 감지', self)
            ai_action.setShortcut('Ctrl+I')
            ai_action.triggered.connect(self.open_ai_detection)
            tools_menu.addAction(ai_action)
        else:
            ai_action = QAction('AI 객체 감지 (설치 필요)', self)
            ai_action.setEnabled(False)
            tools_menu.addAction(ai_action)
        
        tools_menu.addSeparator()
        
        # 고급 처리
        if ADVANCED_PROCESSING_AVAILABLE:
            advanced_action = QAction('고급 처리', self)
            advanced_action.setShortcut('Ctrl+A')
            advanced_action.triggered.connect(self.open_advanced_processing)
            tools_menu.addAction(advanced_action)
        else:
            advanced_action = QAction('고급 처리 (설치 필요)', self)
            advanced_action.setEnabled(False)
            tools_menu.addAction(advanced_action)
        
        tools_menu.addSeparator()
        
        # 일괄 처리
        if BATCH_PROCESSING_AVAILABLE:
            batch_action = QAction('일괄 처리', self)
            batch_action.setShortcut('Ctrl+B')
            batch_action.triggered.connect(self.open_batch_processing)
            tools_menu.addAction(batch_action)
        else:
            batch_action = QAction('일괄 처리 (설치 필요)', self)
            batch_action.setShortcut('Ctrl+B')
            batch_action.setEnabled(False)
            tools_menu.addAction(batch_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu('도움말')
        
        about_action = QAction('정보', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_left_panel(self):
        """왼쪽 패널 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 파일 선택 영역
        file_group = QGroupBox("파일 선택")
        file_layout = QVBoxLayout(file_group)
        
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.on_files_selected)
        file_layout.addWidget(self.drop_zone)
        
        # 선택된 파일 목록
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_layout.addWidget(self.file_list)
        
        # 파일 조작 버튼
        file_btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("목록 지우기")
        self.clear_btn.clicked.connect(self.clear_files)
        file_btn_layout.addWidget(self.clear_btn)
        
        # Google Drive 버튼 추가
        if GOOGLE_DRIVE_AVAILABLE:
            self.drive_btn = QPushButton("Google Drive")
            self.drive_btn.clicked.connect(self.open_google_drive)
            self.drive_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4285f4;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #3367d6;
                }
            """)
            file_btn_layout.addWidget(self.drive_btn)
        
        file_btn_layout.addStretch()
        file_layout.addLayout(file_btn_layout)
        
        layout.addWidget(file_group)
        
        # 처리 옵션 탭
        options_group = QGroupBox("처리 옵션")
        options_layout = QVBoxLayout(options_group)
        
        self.tab_widget = QTabWidget()
        
        # 기본 옵션 탭
        basic_tab = QWidget()
        basic_layout = QGridLayout(basic_tab)
        
        # 크기 조정
        self.resize_check = QCheckBox("크기 조정")
        basic_layout.addWidget(self.resize_check, 0, 0, 1, 2)
        
        basic_layout.addWidget(QLabel("너비:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 7680)
        self.width_spin.setValue(1920)
        basic_layout.addWidget(self.width_spin, 1, 1)
        
        basic_layout.addWidget(QLabel("높이:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 4320)
        self.height_spin.setValue(1080)
        basic_layout.addWidget(self.height_spin, 2, 1)
        
        # 포맷 변경
        self.format_check = QCheckBox("포맷 변경")
        basic_layout.addWidget(self.format_check, 3, 0, 1, 2)
        
        basic_layout.addWidget(QLabel("출력 포맷:"), 4, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp4", "avi", "mov", "mkv", "jpg", "png", "bmp"])
        basic_layout.addWidget(self.format_combo, 4, 1)
        
        basic_layout.addItem(QVBoxLayout(), 5, 0, 1, 2)  # 스페이서
        
        self.tab_widget.addTab(basic_tab, "기본")
        
        # 고급 옵션 탭
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # AI 객체 감지 버튼
        if AI_DETECTION_AVAILABLE:
            self.ai_detect_btn = QPushButton("AI 객체 감지")
            self.ai_detect_btn.clicked.connect(self.open_ai_detection)
            self.ai_detect_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF6B6B;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #FF5252;
                }
            """)
            advanced_layout.addWidget(self.ai_detect_btn)
        
        # 컬러 채널 분리
        if ADVANCED_PROCESSING_AVAILABLE:
            self.channel_split_btn = QPushButton("채널 분리")
            self.channel_split_btn.clicked.connect(self.open_advanced_processing)
            self.channel_split_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            advanced_layout.addWidget(self.channel_split_btn)
        else:
            self.channel_split_btn = QPushButton("채널 분리 (설치 필요)")
            self.channel_split_btn.setEnabled(False)
            advanced_layout.addWidget(self.channel_split_btn)
        
        # 도메인 변환
        if ADVANCED_PROCESSING_AVAILABLE:
            self.domain_convert_btn = QPushButton("색상 도메인 변환")
            self.domain_convert_btn.clicked.connect(self.open_advanced_processing)
            self.domain_convert_btn.setStyleSheet("""
                QPushButton {
                    background-color: #9C27B0;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #7B1FA2;
                }
            """)
            advanced_layout.addWidget(self.domain_convert_btn)
        else:
            self.domain_convert_btn = QPushButton("색상 도메인 변환 (설치 필요)")
            self.domain_convert_btn.setEnabled(False)
            advanced_layout.addWidget(self.domain_convert_btn)
        
        # 일괄 처리 버튼
        if BATCH_PROCESSING_AVAILABLE:
            self.batch_process_btn = QPushButton("일괄 처리")
            self.batch_process_btn.clicked.connect(self.open_batch_processing)
            self.batch_process_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            advanced_layout.addWidget(self.batch_process_btn)
        else:
            self.batch_process_btn = QPushButton("일괄 처리 (설치 필요)")
            self.batch_process_btn.setEnabled(False)
            advanced_layout.addWidget(self.batch_process_btn)
        
        advanced_layout.addWidget(QLabel("더 많은 고급 기능이 곧 추가됩니다..."))
        advanced_layout.addStretch()
        
        self.tab_widget.addTab(advanced_tab, "고급")
        
        options_layout.addWidget(self.tab_widget)
        layout.addWidget(options_group)
        
        # 출력 설정
        output_group = QGroupBox("출력 설정")
        output_layout = QVBoxLayout(output_group)
        
        output_path_layout = QHBoxLayout()
        self.output_path_label = QLabel("출력 폴더: 선택되지 않음")
        self.output_path_btn = QPushButton("출력 폴더 선택")
        self.output_path_btn.clicked.connect(self.select_output_folder)
        
        output_path_layout.addWidget(self.output_path_label)
        output_path_layout.addWidget(self.output_path_btn)
        output_layout.addLayout(output_path_layout)
        
        layout.addWidget(output_group)
        
        # 처리 버튼
        self.process_btn = QPushButton("처리 시작")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)
        
        return widget
    
    def create_right_panel(self):
        """오른쪽 패널 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 진행률 표시
        progress_group = QGroupBox("처리 진행률")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("대기 중...")
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
        
        # 로그 표시
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 로그 조작 버튼
        log_btn_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("로그 지우기")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        log_btn_layout.addWidget(self.clear_log_btn)
        log_btn_layout.addStretch()
        
        log_layout.addLayout(log_btn_layout)
        layout.addWidget(log_group)
        
        return widget
    
    def on_files_selected(self, files):
        """파일 선택 시 처리"""
        self.current_files = files
        self.file_list.clear()
        
        for file_path in files:
            self.file_list.addItem(Path(file_path).name)
        
        self.add_log(f"{len(files)}개 파일이 선택되었습니다.")
    
    def clear_files(self):
        """파일 목록 지우기"""
        self.current_files = []
        self.file_list.clear()
        self.add_log("파일 목록이 지워졌습니다.")
    
    def select_output_folder(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if folder:
            self.output_folder = folder
            self.output_path_label.setText(f"출력 폴더: {folder}")
            self.add_log(f"출력 폴더 설정: {folder}")
    
    def start_processing(self):
        """처리 시작"""
        if not self.current_files:
            self.add_log("처리할 파일이 선택되지 않았습니다.")
            return
        
        if not hasattr(self, 'output_folder'):
            self.add_log("출력 폴더를 선택해주세요.")
            return
        
        # 옵션 수집
        options = {
            'resize': self.resize_check.isChecked(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'format_change': self.format_check.isChecked(),
            'output_format': self.format_combo.currentText()
        }
        
        # 첫 번째 파일 처리
        input_file = self.current_files[0]
        input_path = Path(input_file)
        output_path = Path(self.output_folder) / f"processed_{input_path.name}"
        
        self.add_log(f"처리 시작: {input_path.name}")
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("처리 중...")
        
        # 워커 스레드 시작
        self.processor = MediaProcessor(str(input_file), str(output_path), options)
        self.processor.progress.connect(self.update_progress)
        self.processor.finished.connect(self.on_processing_finished)
        self.processor.error.connect(self.on_processing_error)
        self.processor.start()
    
    def update_progress(self, value):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"처리 중... {value}%")
    
    def on_processing_finished(self, message):
        """처리 완료"""
        self.add_log(message)
        self.progress_bar.setValue(100)
        self.progress_label.setText("완료!")
        self.process_btn.setEnabled(True)
        self.statusBar().showMessage("처리 완료")
    
    def on_processing_error(self, error):
        """처리 오류"""
        self.add_log(f"오류: {error}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("오류 발생")
        self.process_btn.setEnabled(True)
        self.statusBar().showMessage("처리 실패")
    
    def add_log(self, message):
        """로그 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def open_files(self):
        """파일 열기 다이얼로그"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "파일 선택", "", 
            "미디어 파일 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;모든 파일 (*)"
        )
        if files:
            self.on_files_selected(files)
    
    def open_google_drive(self):
        """Google Drive 다이얼로그 열기"""
        if not GOOGLE_DRIVE_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Google Drive", 
                "Google Drive 연동을 위해 다음 패키지를 설치하세요:\n\n"
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
            return
        
        if self.google_drive_dialog is None:
            self.google_drive_dialog = GoogleDriveDialog(self)
        
        self.google_drive_dialog.show()
        self.google_drive_dialog.raise_()
        self.add_log("Google Drive 다이얼로그 열림")
    
    def open_ai_detection(self):
        """AI 객체 감지 다이얼로그 열기"""
        if not AI_DETECTION_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "AI 객체 감지", 
                "AI 객체 감지를 위해 다음 패키지를 설치하세요:\n\n"
                "pip install tensorflow opencv-python"
            )
            return
        
        # 현재 선택된 파일이 이미지인지 확인
        selected_image = None
        if self.current_files:
            for file_path in self.current_files:
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    selected_image = file_path
                    break
        
        if self.ai_detection_dialog is None:
            self.ai_detection_dialog = AIDetectionDialog(self, selected_image)
        else:
            # 새 이미지가 선택되었으면 업데이트
            if selected_image:
                self.ai_detection_dialog.image_path = selected_image
                self.ai_detection_dialog.file_label.setText(Path(selected_image).name)
        
        self.ai_detection_dialog.show()
        self.ai_detection_dialog.raise_()
        self.add_log("AI 객체 감지 다이얼로그 열림")
    
    def open_advanced_processing(self):
        """고급 처리 다이얼로그 열기"""
        if not ADVANCED_PROCESSING_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "고급 처리", 
                "고급 처리를 위해 다음 패키지를 설치하세요:\n\n"
                "pip install matplotlib opencv-python pillow ffmpeg-python"
            )
            return
        
        # 현재 선택된 파일 확인
        selected_file = None
        if self.current_files:
            selected_file = self.current_files[0]  # 첫 번째 파일 사용
        
        if self.advanced_processing_dialog is None:
            self.advanced_processing_dialog = AdvancedProcessingDialog(self, selected_file)
        else:
            # 새 파일이 선택되었으면 업데이트
            if selected_file:
                self.advanced_processing_dialog.file_path = selected_file
                self.advanced_processing_dialog.file_label.setText(Path(selected_file).name)
        
        self.advanced_processing_dialog.show()
        self.advanced_processing_dialog.raise_()
        self.add_log("고급 처리 다이얼로그 열림")
    
    def open_batch_processing(self):
        """일괄 처리 다이얼로그 열기"""
        if not BATCH_PROCESSING_AVAILABLE:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "일괄 처리", 
                "일괄 처리를 위해 다음 패키지를 설치하세요:\n\n"
                "pip install opencv-python pillow ffmpeg-python matplotlib"
            )
            return
        
        # 현재 선택된 파일들을 초기값으로 전달
        initial_files = self.current_files.copy() if self.current_files else None
        
        if self.batch_processing_dialog is None:
            self.batch_processing_dialog = BatchProcessingDialog(self, initial_files)
        else:
            # 새 파일들이 선택되었으면 추가
            if initial_files:
                self.batch_processing_dialog.add_files(initial_files)
        
        self.batch_processing_dialog.show()
        self.batch_processing_dialog.raise_()
        self.add_log("일괄 처리 다이얼로그 열림")
    
    def show_about(self):
        """정보 다이얼로그"""
        from PySide6.QtWidgets import QMessageBox
        
        # 사용 가능한 기능 체크
        features = []
        if GOOGLE_DRIVE_AVAILABLE:
            features.append("✅ Google Drive 연동")
        else:
            features.append("❌ Google Drive 연동 (설치 필요)")
        
        if AI_DETECTION_AVAILABLE:
            features.append("✅ AI 객체 감지")
        else:
            features.append("❌ AI 객체 감지 (설치 필요)")
        
        if ADVANCED_PROCESSING_AVAILABLE:
            features.append("✅ 고급 처리 (채널 분리, 색상 변환, 필터 등)")
        else:
            features.append("❌ 고급 처리 (설치 필요)")
        
        if BATCH_PROCESSING_AVAILABLE:
            features.append("✅ 일괄 처리 (10개 이상 파일 동시 처리)")
        else:
            features.append("❌ 일괄 처리 (설치 필요)")
        
        features_text = "<br>".join(features)
        
        QMessageBox.about(
            self, "EasyMediaProcessor 정보",
            f"""
            <h3>EasyMediaProcessor v1.0</h3>
            <p>전문적인 영상 및 이미지 처리 도구</p>
            
            <p><b>주요 기능:</b></p>
            <ul>
            <li>기본 이미지/비디오 포맷 변환 및 크기 조정</li>
            <li>Google Drive 클라우드 연동</li>
            <li>AI 기반 객체 감지 (얼굴, 객체 등)</li>
            <li>고급 이미지 처리 (채널 분리, 색상 변환, 필터)</li>
            <li>비디오 편집 및 자르기</li>
            <li>이미지 품질 향상 및 노이즈 감소</li>
            <li>히스토그램 분석 및 메타데이터 추출</li>
            <li>일괄 처리 (10개 이상 파일 동시 처리)</li>
            </ul>
            
            <p><b>현재 상태:</b><br>
            {features_text}</p>
            
            <p><b>지원 포맷:</b><br>
            비디오: MP4, AVI, MOV, MKV<br>
            이미지: JPG, PNG, BMP</p>
            
            <p><b>단축키:</b><br>
            Ctrl+O: 파일 열기<br>
            Ctrl+D: Google Drive 연동<br>
            Ctrl+I: AI 객체 감지<br>
            Ctrl+A: 고급 처리<br>
            Ctrl+B: 일괄 처리<br>
            Ctrl+Q: 종료</p>
            
            <p>© 2024 EasyMediaProcessor</p>
            """
        )


def main():
    # 고해상도 디스플레이 지원
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # 앱 정보 설정
    app.setApplicationName("EasyMediaProcessor")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("윤재성")
    app.setOrganizationDomain("crosefrog@naver.com")
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as e:
        print(f"애플리케이션 시작 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())