import os
import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QGroupBox, QComboBox, QSpinBox, 
                              QListWidget, QProgressBar, QTextEdit,
                              QFileDialog, QCheckBox, QGridLayout,
                              QMessageBox, QFrame, QSplitter, QTabWidget,
                              QWidget, QSlider)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import ffmpeg

class BatchProcessor(QThread):
    """일괄 처리 스레드"""
    progress = Signal(int)  # 전체 진행률
    file_progress = Signal(int, str, int)  # 파일 인덱스, 파일명, 파일 진행률
    finished = Signal(list)  # 완료된 파일 목록
    error = Signal(str, str)  # 파일명, 오류 메시지
    
    def __init__(self, files, output_dir, operations):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.operations = operations
        self.should_stop = False
    
    def stop(self):
        """처리 중단"""
        self.should_stop = True
    
    def run(self):
        completed_files = []
        total_files = len(self.files)
        
        try:
            for i, file_path in enumerate(self.files):
                if self.should_stop:
                    break
                
                file_name = Path(file_path).name
                self.file_progress.emit(i, file_name, 0)
                
                try:
                    # 파일 처리
                    output_path = self.process_single_file(file_path, i)
                    if output_path:
                        completed_files.append((file_path, output_path))
                        self.file_progress.emit(i, file_name, 100)
                    
                except Exception as e:
                    self.error.emit(file_name, str(e))
                    continue
                
                # 전체 진행률 업데이트
                overall_progress = int((i + 1) / total_files * 100)
                self.progress.emit(overall_progress)
                
                # 짧은 지연 (UI 업데이트용)
                self.msleep(50)
            
            self.finished.emit(completed_files)
            
        except Exception as e:
            self.error.emit("전체 처리", f"일괄 처리 실패: {str(e)}")
    
    def process_single_file(self, file_path, file_index):
        """단일 파일 처리"""
        input_path = Path(file_path)
        
        # 출력 파일명 생성
        if self.operations.get('add_prefix'):
            prefix = self.operations.get('prefix', 'processed_')
            output_name = f"{prefix}{input_path.stem}"
        else:
            output_name = f"{input_path.stem}_batch"
        
        # 확장자 변경
        if self.operations.get('change_format'):
            new_format = self.operations.get('output_format', input_path.suffix[1:])
            output_path = Path(self.output_dir) / f"{output_name}.{new_format}"
        else:
            output_path = Path(self.output_dir) / f"{output_name}{input_path.suffix}"
        
        # 파일 종류에 따른 처리
        if self.is_image_file(file_path):
            return self.process_image(file_path, output_path, file_index)
        elif self.is_video_file(file_path):
            return self.process_video(file_path, output_path, file_index)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {input_path.suffix}")
    
    def process_image(self, input_path, output_path, file_index):
        """이미지 파일 처리"""
        self.file_progress.emit(file_index, Path(input_path).name, 20)
        
        # OpenCV로 이미지 로드
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다")
        
        # 크기 조정
        if self.operations.get('resize'):
            width = self.operations.get('width', 1920)
            height = self.operations.get('height', 1080)
            image = cv2.resize(image, (width, height))
            self.file_progress.emit(file_index, Path(input_path).name, 40)
        
        # 이미지 향상 (PIL 사용)
        if self.operations.get('enhance'):
            # OpenCV -> PIL 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 향상 적용
            if self.operations.get('brightness', 1.0) != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(self.operations['brightness'])
            
            if self.operations.get('contrast', 1.0) != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(self.operations['contrast'])
            
            if self.operations.get('saturation', 1.0) != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(self.operations['saturation'])
            
            # PIL -> OpenCV 변환
            image_rgb = np.array(pil_image)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            self.file_progress.emit(file_index, Path(input_path).name, 60)
        
        # 필터 적용
        if self.operations.get('apply_filter'):
            filter_type = self.operations.get('filter_type', 'none')
            if filter_type == 'blur':
                image = cv2.GaussianBlur(image, (5, 5), 0)
            elif filter_type == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                image = cv2.filter2D(image, -1, kernel)
            elif filter_type == 'edge':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            self.file_progress.emit(file_index, Path(input_path).name, 80)
        
        # 저장
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise ValueError("이미지 저장 실패")
        
        self.file_progress.emit(file_index, Path(input_path).name, 100)
        return str(output_path)
    
    def process_video(self, input_path, output_path, file_index):
        """비디오 파일 처리"""
        self.file_progress.emit(file_index, Path(input_path).name, 20)
        
        try:
            stream = ffmpeg.input(input_path)
            
            # 크기 조정
            if self.operations.get('resize'):
                width = self.operations.get('width', 1920)
                height = self.operations.get('height', 1080)
                stream = ffmpeg.filter(stream, 'scale', width, height)
                self.file_progress.emit(file_index, Path(input_path).name, 40)
            
            # 비디오 자르기 (시간)
            if self.operations.get('trim_video'):
                start_time = self.operations.get('start_time', 0)
                duration = self.operations.get('duration', None)
                if duration:
                    stream = ffmpeg.input(input_path, ss=start_time, t=duration)
                else:
                    stream = ffmpeg.input(input_path, ss=start_time)
                self.file_progress.emit(file_index, Path(input_path).name, 60)
            
            # 품질 설정
            codec_options = {}
            if self.operations.get('change_quality'):
                quality = self.operations.get('quality', 'medium')
                if quality == 'high':
                    codec_options = {'crf': '18', 'preset': 'slow'}
                elif quality == 'medium':
                    codec_options = {'crf': '23', 'preset': 'medium'}
                elif quality == 'low':
                    codec_options = {'crf': '28', 'preset': 'fast'}
            
            self.file_progress.emit(file_index, Path(input_path).name, 80)
            
            # 출력
            stream = ffmpeg.output(stream, str(output_path), **codec_options)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            self.file_progress.emit(file_index, Path(input_path).name, 100)
            return str(output_path)
            
        except ffmpeg.Error as e:
            raise ValueError(f"FFmpeg 오류: {e.stderr.decode() if e.stderr else str(e)}")
    
    def is_image_file(self, file_path):
        """이미지 파일 확인"""
        return Path(file_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    def is_video_file(self, file_path):
        """비디오 파일 확인"""
        return Path(file_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']


class BatchDropZone(QFrame):
    """일괄 처리용 드래그 앤 드롭 영역"""
    files_dropped = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #999;
                border-radius: 8px;
                background-color: #f5f5f5;
            }
            QFrame:hover {
                border-color: #2196F3;
                background-color: #e3f2fd;
            }
        """)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("📁 파일들을 여기로 드래그하거나 클릭하여 선택\n(최소 10개 파일까지 일괄 처리 가능)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #666; font-size: 13px; font-weight: bold;")
        
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                # 폴더인 경우 내부 파일들 추가
                if os.path.isdir(file_path):
                    for root, dirs, filenames in os.walk(file_path):
                        for filename in filenames:
                            full_path = os.path.join(root, filename)
                            if self.is_supported_file(full_path):
                                files.append(full_path)
                else:
                    if self.is_supported_file(file_path):
                        files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(
                self, "일괄 처리할 파일들 선택", "", 
                "미디어 파일 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;모든 파일 (*)"
            )
            if files:
                self.files_dropped.emit(files)
    
    def is_supported_file(self, file_path):
        """지원되는 파일 형식 확인"""
        ext = Path(file_path).suffix.lower()
        return ext in ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']


class BatchProcessingDialog(QDialog):
    """일괄 처리 다이얼로그"""
    
    def __init__(self, parent=None, initial_files=None):
        super().__init__(parent)
        self.files = initial_files or []
        self.processor = None
        self.output_dir = None
        
        self.setWindowTitle("일괄 처리")
        self.setMinimumSize(900, 700)
        
        self.setup_ui()
        
        if self.files:
            self.update_file_list()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 상단: 파일 선택 영역
        file_group = QGroupBox("처리할 파일 선택")
        file_layout = QVBoxLayout(file_group)
        
        self.drop_zone = BatchDropZone()
        self.drop_zone.files_dropped.connect(self.add_files)
        file_layout.addWidget(self.drop_zone)
        
        # 파일 목록 및 조작
        list_layout = QHBoxLayout()
        
        # 파일 목록
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        list_layout.addWidget(self.file_list, 2)
        
        # 파일 조작 버튼들
        button_layout = QVBoxLayout()
        
        self.add_files_btn = QPushButton("파일 추가")
        self.add_files_btn.clicked.connect(self.select_files)
        button_layout.addWidget(self.add_files_btn)
        
        self.add_folder_btn = QPushButton("폴더 추가")
        self.add_folder_btn.clicked.connect(self.select_folder)
        button_layout.addWidget(self.add_folder_btn)
        
        self.remove_selected_btn = QPushButton("선택 제거")
        self.remove_selected_btn.clicked.connect(self.remove_selected)
        button_layout.addWidget(self.remove_selected_btn)
        
        self.clear_all_btn = QPushButton("모두 제거")
        self.clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_all_btn)
        
        button_layout.addStretch()
        list_layout.addLayout(button_layout, 1)
        
        file_layout.addLayout(list_layout)
        layout.addWidget(file_group)
        
        # 중간: 처리 옵션
        options_group = QGroupBox("처리 옵션")
        options_layout = QVBoxLayout(options_group)
        
        # 탭으로 옵션 분류
        self.options_tab = QTabWidget()
        
        # 기본 옵션 탭
        self.create_basic_options_tab()
        
        # 이미지 옵션 탭
        self.create_image_options_tab()
        
        # 비디오 옵션 탭
        self.create_video_options_tab()
        
        # 출력 옵션 탭
        self.create_output_options_tab()
        
        options_layout.addWidget(self.options_tab)
        layout.addWidget(options_group)
        
        # 하단: 출력 및 실행
        bottom_layout = QHBoxLayout()
        
        # 출력 디렉토리
        output_group = QGroupBox("출력 설정")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel("출력 폴더: 선택되지 않음")
        self.select_output_btn = QPushButton("출력 폴더 선택")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        
        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(self.select_output_btn)
        output_layout.addLayout(output_dir_layout)
        
        bottom_layout.addWidget(output_group, 2)
        
        # 실행 버튼
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("일괄 처리 시작")
        self.start_btn.clicked.connect(self.start_batch_processing)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("중단")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        control_layout.addWidget(self.stop_btn)
        
        bottom_layout.addLayout(control_layout, 1)
        
        layout.addLayout(bottom_layout)
        
        # 진행률 표시
        progress_group = QGroupBox("처리 진행률")
        progress_layout = QVBoxLayout(progress_group)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setFormat("전체 진행률: %p%")
        progress_layout.addWidget(self.overall_progress)
        
        self.current_file_label = QLabel("대기 중...")
        progress_layout.addWidget(self.current_file_label)
        
        self.file_progress = QProgressBar()
        self.file_progress.setFormat("현재 파일: %p%")
        progress_layout.addWidget(self.file_progress)
        
        layout.addWidget(progress_group)
        
        # 로그
        log_group = QGroupBox("처리 로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def create_basic_options_tab(self):
        """기본 옵션 탭 생성"""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # 크기 조정
        self.resize_check = QCheckBox("크기 조정")
        layout.addWidget(self.resize_check, 0, 0, 1, 2)
        
        layout.addWidget(QLabel("너비:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 7680)
        self.width_spin.setValue(1920)
        layout.addWidget(self.width_spin, 1, 1)
        
        layout.addWidget(QLabel("높이:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 4320)
        self.height_spin.setValue(1080)
        layout.addWidget(self.height_spin, 2, 1)
        
        # 포맷 변경
        self.format_check = QCheckBox("포맷 변경")
        layout.addWidget(self.format_check, 3, 0, 1, 2)
        
        layout.addWidget(QLabel("출력 포맷:"), 4, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png", "bmp", "mp4", "avi", "mov"])
        layout.addWidget(self.format_combo, 4, 1)
        
        # 파일명 접두사
        self.prefix_check = QCheckBox("파일명에 접두사 추가")
        layout.addWidget(self.prefix_check, 5, 0, 1, 2)
        
        layout.addWidget(QLabel("접두사:"), 6, 0)
        self.prefix_combo = QComboBox()
        self.prefix_combo.setEditable(True)
        self.prefix_combo.addItems(["processed_", "batch_", "converted_", "resized_"])
        layout.addWidget(self.prefix_combo, 6, 1)
        
        layout.setRowStretch(7, 1)
        
        self.options_tab.addTab(tab, "기본")
    
    def create_image_options_tab(self):
        """이미지 옵션 탭 생성"""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # 이미지 향상
        self.enhance_check = QCheckBox("이미지 품질 향상")
        layout.addWidget(self.enhance_check, 0, 0, 1, 3)
        
        # 밝기
        layout.addWidget(QLabel("밝기:"), 1, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(50, 150)
        self.brightness_slider.setValue(100)
        self.brightness_label = QLabel("1.0")
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(f"{v/100:.1f}")
        )
        layout.addWidget(self.brightness_slider, 1, 1)
        layout.addWidget(self.brightness_label, 1, 2)
        
        # 대비
        layout.addWidget(QLabel("대비:"), 2, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 150)
        self.contrast_slider.setValue(100)
        self.contrast_label = QLabel("1.0")
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(f"{v/100:.1f}")
        )
        layout.addWidget(self.contrast_slider, 2, 1)
        layout.addWidget(self.contrast_label, 2, 2)
        
        # 채도
        layout.addWidget(QLabel("채도:"), 3, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(50, 150)
        self.saturation_slider.setValue(100)
        self.saturation_label = QLabel("1.0")
        self.saturation_slider.valueChanged.connect(
            lambda v: self.saturation_label.setText(f"{v/100:.1f}")
        )
        layout.addWidget(self.saturation_slider, 3, 1)
        layout.addWidget(self.saturation_label, 3, 2)
        
        # 필터
        self.filter_check = QCheckBox("필터 적용")
        layout.addWidget(self.filter_check, 4, 0, 1, 3)
        
        layout.addWidget(QLabel("필터 종류:"), 5, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "blur", "sharpen", "edge"])
        layout.addWidget(self.filter_combo, 5, 1, 1, 2)
        
        layout.setRowStretch(6, 1)
        
        self.options_tab.addTab(tab, "이미지")
    
    def create_video_options_tab(self):
        """비디오 옵션 탭 생성"""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # 비디오 자르기
        self.trim_check = QCheckBox("비디오 자르기 (시간)")
        layout.addWidget(self.trim_check, 0, 0, 1, 3)
        
        layout.addWidget(QLabel("시작 시간 (초):"), 1, 0)
        self.start_time_spin = QSpinBox()
        self.start_time_spin.setRange(0, 3600)
        self.start_time_spin.setValue(0)
        layout.addWidget(self.start_time_spin, 1, 1, 1, 2)
        
        layout.addWidget(QLabel("지속 시간 (초):"), 2, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(30)
        layout.addWidget(self.duration_spin, 2, 1, 1, 2)
        
        # 품질 설정
        self.quality_check = QCheckBox("품질 변경")
        layout.addWidget(self.quality_check, 3, 0, 1, 3)
        
        layout.addWidget(QLabel("품질:"), 4, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["high", "medium", "low"])
        self.quality_combo.setCurrentText("medium")
        layout.addWidget(self.quality_combo, 4, 1, 1, 2)
        
        layout.setRowStretch(5, 1)
        
        self.options_tab.addTab(tab, "비디오")
    
    def create_output_options_tab(self):
        """출력 옵션 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("출력 파일 구성 및 품질 설정"))
        
        # 출력 구조
        structure_group = QGroupBox("출력 구조")
        structure_layout = QGridLayout(structure_group)
        
        self.keep_structure_check = QCheckBox("원본 폴더 구조 유지")
        structure_layout.addWidget(self.keep_structure_check, 0, 0, 1, 2)
        
        self.create_date_folder_check = QCheckBox("날짜별 폴더 생성")
        structure_layout.addWidget(self.create_date_folder_check, 1, 0, 1, 2)
        
        layout.addWidget(structure_group)
        
        # 오류 처리
        error_group = QGroupBox("오류 처리")
        error_layout = QVBoxLayout(error_group)
        
        self.skip_errors_check = QCheckBox("오류 발생 시 건너뛰고 계속")
        self.skip_errors_check.setChecked(True)
        error_layout.addWidget(self.skip_errors_check)
        
        self.create_log_check = QCheckBox("처리 로그 파일 생성")
        error_layout.addWidget(self.create_log_check)
        
        layout.addWidget(error_group)
        layout.addStretch()
        
        self.options_tab.addTab(tab, "출력")
    
    def add_files(self, new_files):
        """파일 목록에 파일들 추가"""
        for file_path in new_files:
            if file_path not in self.files:
                self.files.append(file_path)
        
        self.update_file_list()
        self.add_log(f"{len(new_files)}개 파일이 추가되었습니다.")
    
    def select_files(self):
        """파일 선택 다이얼로그"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "일괄 처리할 파일들 선택", "",
            "미디어 파일 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;모든 파일 (*)"
        )
        if files:
            self.add_files(files)
    
    def select_folder(self):
        """폴더 선택 (내부 파일들 모두 추가)"""
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            files = []
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    if self.drop_zone.is_supported_file(full_path):
                        files.append(full_path)
            
            if files:
                self.add_files(files)
            else:
                QMessageBox.information(self, "알림", "폴더에서 지원되는 미디어 파일을 찾을 수 없습니다.")
    
    def remove_selected(self):
        """선택된 파일들 제거"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.file_list.row(item)
            if 0 <= row < len(self.files):
                del self.files[row]
        
        self.update_file_list()
        self.add_log(f"{len(selected_items)}개 파일이 제거되었습니다.")
    
    def clear_all(self):
        """모든 파일 제거"""
        if self.files:
            reply = QMessageBox.question(
                self, "확인", f"{len(self.files)}개 파일을 모두 제거하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.files.clear()
                self.update_file_list()
                self.add_log("모든 파일이 제거되었습니다.")
    
    def update_file_list(self):
        """파일 목록 업데이트"""
        self.file_list.clear()
        
        for file_path in self.files:
            path = Path(file_path)
            # 파일 크기 계산
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                size_text = f" ({size_mb:.1f} MB)"
            except:
                size_text = ""
            
            # 파일 형식 아이콘
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                icon = "🖼️"
            elif path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                icon = "🎬"
            else:
                icon = "📄"
            
            item_text = f"{icon} {path.name}{size_text}"
            self.file_list.addItem(item_text)
        
        # 상태 업데이트
        total_count = len(self.files)
        image_count = sum(1 for f in self.files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
        video_count = sum(1 for f in self.files if Path(f).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv'])
        
        self.drop_zone.label.setText(
            f"📁 총 {total_count}개 파일 선택됨\n"
            f"🖼️ 이미지: {image_count}개 | 🎬 비디오: {video_count}개\n"
            f"더 추가하려면 드래그 또는 클릭"
        )
    
    def select_output_directory(self):
        """출력 디렉토리 선택"""
        directory = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(f"출력 폴더: {directory}")
            self.add_log(f"출력 폴더 설정: {directory}")
    
    def start_batch_processing(self):
        """일괄 처리 시작"""
        # 유효성 검사
        if not self.files:
            QMessageBox.warning(self, "경고", "처리할 파일이 없습니다.")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "경고", "출력 폴더를 선택해주세요.")
            return
        
        if len(self.files) < 2:
            reply = QMessageBox.question(
                self, "확인", 
                "파일이 2개 미만입니다. 일괄 처리를 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 옵션 수집
        operations = self.collect_operations()
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.overall_progress.setValue(0)
        self.file_progress.setValue(0)
        self.current_file_label.setText("처리 시작...")
        
        # 처리 시작
        self.add_log(f"일괄 처리 시작: {len(self.files)}개 파일")
        
        self.processor = BatchProcessor(self.files, self.output_dir, operations)
        self.processor.progress.connect(self.overall_progress.setValue)
        self.processor.file_progress.connect(self.update_file_progress)
        self.processor.finished.connect(self.on_batch_finished)
        self.processor.error.connect(self.on_file_error)
        self.processor.start()
    
    def collect_operations(self):
        """처리 옵션들 수집"""
        operations = {}
        
        # 기본 옵션
        operations['resize'] = self.resize_check.isChecked()
        operations['width'] = self.width_spin.value()
        operations['height'] = self.height_spin.value()
        operations['change_format'] = self.format_check.isChecked()
        operations['output_format'] = self.format_combo.currentText()
        operations['add_prefix'] = self.prefix_check.isChecked()
        operations['prefix'] = self.prefix_combo.currentText()
        
        # 이미지 옵션
        operations['enhance'] = self.enhance_check.isChecked()
        operations['brightness'] = self.brightness_slider.value() / 100.0
        operations['contrast'] = self.contrast_slider.value() / 100.0
        operations['saturation'] = self.saturation_slider.value() / 100.0
        operations['apply_filter'] = self.filter_check.isChecked()
        operations['filter_type'] = self.filter_combo.currentText()
        
        # 비디오 옵션
        operations['trim_video'] = self.trim_check.isChecked()
        operations['start_time'] = self.start_time_spin.value()
        operations['duration'] = self.duration_spin.value() if self.trim_check.isChecked() else None
        operations['change_quality'] = self.quality_check.isChecked()
        operations['quality'] = self.quality_combo.currentText()
        
        # 출력 옵션
        operations['keep_structure'] = self.keep_structure_check.isChecked()
        operations['create_date_folder'] = self.create_date_folder_check.isChecked()
        operations['skip_errors'] = self.skip_errors_check.isChecked()
        operations['create_log'] = self.create_log_check.isChecked()
        
        return operations
    
    def update_file_progress(self, file_index, file_name, progress):
        """파일별 진행률 업데이트"""
        self.current_file_label.setText(f"처리 중: {file_name}")
        self.file_progress.setValue(progress)
    
    def on_batch_finished(self, completed_files):
        """일괄 처리 완료"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.current_file_label.setText("처리 완료!")
        
        total_files = len(self.files)
        success_count = len(completed_files)
        failed_count = total_files - success_count
        
        self.add_log(f"일괄 처리 완료!")
        self.add_log(f"✅ 성공: {success_count}개")
        if failed_count > 0:
            self.add_log(f"❌ 실패: {failed_count}개")
        
        # 완료 다이얼로그
        message = f"일괄 처리가 완료되었습니다!\n\n"
        message += f"총 {total_files}개 파일 중 {success_count}개 성공"
        if failed_count > 0:
            message += f", {failed_count}개 실패"
        message += f"\n\n출력 폴더를 열어보시겠습니까?"
        
        reply = QMessageBox.question(
            self, "처리 완료", message,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self.output_dir:
            import os
            os.startfile(self.output_dir)  # Windows
    
    def on_file_error(self, file_name, error_message):
        """파일 처리 오류"""
        self.add_log(f"❌ 오류 - {file_name}: {error_message}")
    
    def stop_processing(self):
        """처리 중단"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "확인", "일괄 처리를 중단하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processor.stop()
                self.add_log("처리가 사용자에 의해 중단되었습니다.")
                
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.current_file_label.setText("중단됨")
    
    def add_log(self, message):
        """로그 추가"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """다이얼로그 닫기 시"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "확인", 
                "처리가 진행 중입니다. 정말 닫으시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processor.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()