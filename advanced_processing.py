import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QGroupBox, QComboBox, QSpinBox, 
                              QSlider, QGridLayout, QProgressBar, QTextEdit,
                              QFileDialog, QCheckBox, QTabWidget, QWidget,
                              QMessageBox)
from PySide6.QtCore import Qt
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AdvancedProcessor(QThread):
    """고급 이미지/비디오 처리 스레드"""
    progress = Signal(int)
    result = Signal(str, str)  # message, output_path
    error = Signal(str)
    
    def __init__(self, input_path, output_path, operation, parameters):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.operation = operation
        self.parameters = parameters
    
    def run(self):
        try:
            if self.operation == "channel_split":
                self.split_color_channels()
            elif self.operation == "domain_convert":
                self.convert_color_domain()
            elif self.operation == "histogram":
                self.generate_histogram()
            elif self.operation == "enhance":
                self.enhance_image()
            elif self.operation == "filter":
                self.apply_filter()
            elif self.operation == "crop_video":
                self.crop_video()
            elif self.operation == "upscale":
                self.upscale_image()
            elif self.operation == "noise_reduction":
                self.reduce_noise()
            else:
                self.error.emit(f"알 수 없는 작업: {self.operation}")
        except Exception as e:
            self.error.emit(f"처리 실패: {str(e)}")
    
    def split_color_channels(self):
        """RGB/HSV 채널 분리"""
        self.progress.emit(10)
        
        # 이미지 로드
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        color_space = self.parameters.get('color_space', 'RGB')
        
        if color_space == 'RGB':
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            channels = cv2.split(image_rgb)
            channel_names = ['Red', 'Green', 'Blue']
        elif color_space == 'HSV':
            # BGR to HSV
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channels = cv2.split(image_hsv)
            channel_names = ['Hue', 'Saturation', 'Value']
        elif color_space == 'HSI':
            # BGR to HSI (근사값)
            image_hsi = self.bgr_to_hsi(image)
            channels = cv2.split(image_hsi)
            channel_names = ['Hue', 'Saturation', 'Intensity']
        
        self.progress.emit(50)
        
        # 각 채널을 별도 파일로 저장
        base_path = Path(self.output_path)
        base_name = base_path.stem
        base_dir = base_path.parent
        
        saved_files = []
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            channel_path = base_dir / f"{base_name}_{name}.png"
            cv2.imwrite(str(channel_path), channel)
            saved_files.append(str(channel_path))
            self.progress.emit(50 + (i + 1) * 15)
        
        self.progress.emit(100)
        message = f"{color_space} 채널 분리 완료\n저장된 파일들:\n" + "\n".join(saved_files)
        self.result.emit(message, str(saved_files[0]))
    
    def convert_color_domain(self):
        """색상 도메인 변환"""
        self.progress.emit(20)
        
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        source_domain = self.parameters.get('source', 'BGR')
        target_domain = self.parameters.get('target', 'HSV')
        
        self.progress.emit(50)
        
        # 색상 공간 변환
        if source_domain == 'BGR' and target_domain == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif source_domain == 'BGR' and target_domain == 'RGB':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif source_domain == 'BGR' and target_domain == 'LAB':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif source_domain == 'BGR' and target_domain == 'GRAY':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif source_domain == 'BGR' and target_domain == 'HSI':
            converted = self.bgr_to_hsi(image)
        else:
            converted = image
        
        self.progress.emit(80)
        
        # 저장
        cv2.imwrite(self.output_path, converted)
        
        self.progress.emit(100)
        message = f"색상 도메인 변환 완료: {source_domain} → {target_domain}"
        self.result.emit(message, self.output_path)
    
    def generate_histogram(self):
        """히스토그램 생성"""
        self.progress.emit(20)
        
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.progress.emit(50)
        
        # 히스토그램 계산
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Image Histogram Analysis')
        
        # 원본 이미지 표시
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # RGB 히스토그램
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
            axes[0, 1].plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        axes[0, 1].set_title('RGB Histogram')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 그레이스케일 히스토그램
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axes[1, 0].plot(hist_gray, color='black')
        axes[1, 0].set_title('Grayscale Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # HSV 히스토그램
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        axes[1, 1].plot(hist_h, color='purple')
        axes[1, 1].set_title('Hue Histogram')
        axes[1, 1].set_xlabel('Hue Value')
        axes[1, 1].set_ylabel('Frequency')
        
        self.progress.emit(80)
        
        # 히스토그램 저장
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.progress.emit(100)
        message = f"히스토그램 분석 완료"
        self.result.emit(message, self.output_path)
    
    def enhance_image(self):
        """이미지 품질 향상"""
        self.progress.emit(20)
        
        # PIL로 이미지 로드
        image = Image.open(self.input_path)
        
        brightness = self.parameters.get('brightness', 1.0)
        contrast = self.parameters.get('contrast', 1.0)
        saturation = self.parameters.get('saturation', 1.0)
        sharpness = self.parameters.get('sharpness', 1.0)
        
        self.progress.emit(40)
        
        # 각 속성 조정
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        self.progress.emit(60)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        self.progress.emit(80)
        
        # 저장
        image.save(self.output_path)
        
        self.progress.emit(100)
        message = f"이미지 품질 향상 완료"
        self.result.emit(message, self.output_path)
    
    def apply_filter(self):
        """필터 적용"""
        self.progress.emit(20)
        
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        filter_type = self.parameters.get('filter_type', 'blur')
        kernel_size = self.parameters.get('kernel_size', 5)
        
        self.progress.emit(50)
        
        if filter_type == 'blur':
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == 'sharpen':
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(image, -1, kernel)
        elif filter_type == 'edge':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'emboss':
            kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
            result = cv2.filter2D(image, -1, kernel)
        else:
            result = image
        
        self.progress.emit(80)
        
        cv2.imwrite(self.output_path, result)
        
        self.progress.emit(100)
        message = f"{filter_type} 필터 적용 완료"
        self.result.emit(message, self.output_path)
    
    def crop_video(self):
        """비디오 자르기 (시간 범위)"""
        import ffmpeg
        
        self.progress.emit(20)
        
        start_time = self.parameters.get('start_time', 0)  # 초
        duration = self.parameters.get('duration', 10)     # 초
        
        self.progress.emit(50)
        
        try:
            (
                ffmpeg
                .input(self.input_path, ss=start_time, t=duration)
                .output(self.output_path)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg 오류: {e.stderr.decode()}")
        
        self.progress.emit(100)
        message = f"비디오 자르기 완료: {start_time}초부터 {duration}초간"
        self.result.emit(message, self.output_path)
    
    def upscale_image(self):
        """이미지 업스케일링"""
        self.progress.emit(20)
        
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        scale_factor = self.parameters.get('scale_factor', 2)
        method = self.parameters.get('method', 'INTER_CUBIC')
        
        self.progress.emit(50)
        
        # 업스케일링 방법
        if method == 'INTER_LINEAR':
            interpolation = cv2.INTER_LINEAR
        elif method == 'INTER_CUBIC':
            interpolation = cv2.INTER_CUBIC
        elif method == 'INTER_LANCZOS4':
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_CUBIC
        
        # 새 크기 계산
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.progress.emit(70)
        
        # 업스케일링 수행
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        self.progress.emit(90)
        
        cv2.imwrite(self.output_path, upscaled)
        
        self.progress.emit(100)
        message = f"이미지 업스케일링 완료: {width}x{height} → {new_width}x{new_height}"
        self.result.emit(message, self.output_path)
    
    def reduce_noise(self):
        """노이즈 감소"""
        self.progress.emit(20)
        
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        method = self.parameters.get('method', 'bilateral')
        strength = self.parameters.get('strength', 10)
        
        self.progress.emit(50)
        
        if method == 'bilateral':
            result = cv2.bilateralFilter(image, 9, strength * 2, strength * 2)
        elif method == 'gaussian':
            result = cv2.GaussianBlur(image, (5, 5), strength / 10)
        elif method == 'median':
            result = cv2.medianBlur(image, min(strength, 15))
        elif method == 'nlmeans':
            result = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            result = image
        
        self.progress.emit(80)
        
        cv2.imwrite(self.output_path, result)
        
        self.progress.emit(100)
        message = f"{method} 노이즈 감소 완료"
        self.result.emit(message, self.output_path)
    
    def bgr_to_hsi(self, image):
        """BGR을 HSI로 변환 (근사값)"""
        # BGR을 RGB로 변환
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # HSI 계산
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        # Intensity
        i = (r + g + b) / 3.0
        
        # Saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = 1 - (3 * min_rgb) / (r + g + b + 1e-6)
        
        # Hue
        numerator = 0.5 * ((r - g) + (r - b))
        denominator = np.sqrt((r - g)**2 + (r - b) * (g - b)) + 1e-6
        h = np.arccos(np.clip(numerator / denominator, -1, 1))
        h[b > g] = 2 * np.pi - h[b > g]
        h = h / (2 * np.pi)
        
        # HSI를 0-255 범위로 변환
        hsi = np.stack([h * 255, s * 255, i * 255], axis=2)
        return hsi.astype(np.uint8)


class AdvancedProcessingDialog(QDialog):
    """고급 처리 다이얼로그"""
    
    def __init__(self, parent=None, file_path=None):
        super().__init__(parent)
        self.file_path = file_path
        self.processor = None
        
        self.setWindowTitle("고급 영상/이미지 처리")
        self.setMinimumSize(700, 600)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 파일 선택
        file_group = QGroupBox("입력 파일")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("파일이 선택되지 않음")
        if self.file_path:
            self.file_label.setText(str(Path(self.file_path).name))
        
        self.select_file_btn = QPushButton("파일 선택")
        self.select_file_btn.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_btn)
        layout.addWidget(file_group)
        
        # 처리 옵션 탭
        self.tab_widget = QTabWidget()
        
        # 채널 분리 탭
        self.create_channel_split_tab()
        
        # 색상 변환 탭
        self.create_color_convert_tab()
        
        # 이미지 향상 탭
        self.create_enhance_tab()
        
        # 필터 탭
        self.create_filter_tab()
        
        # 비디오 편집 탭
        self.create_video_edit_tab()
        
        # 분석 탭
        self.create_analysis_tab()
        
        layout.addWidget(self.tab_widget)
        
        # 처리 버튼
        self.process_btn = QPushButton("처리 시작")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.process_btn)
        
        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 결과 표시
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(100)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def create_channel_split_tab(self):
        """채널 분리 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("컬러 채널을 분리하여 각각의 이미지로 저장합니다."))
        
        # 색상 공간 선택
        color_group = QGroupBox("색상 공간")
        color_layout = QGridLayout(color_group)
        
        self.channel_color_combo = QComboBox()
        self.channel_color_combo.addItems(["RGB", "HSV", "HSI"])
        color_layout.addWidget(QLabel("색상 공간:"), 0, 0)
        color_layout.addWidget(self.channel_color_combo, 0, 1)
        
        layout.addWidget(color_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "채널 분리")
    
    def create_color_convert_tab(self):
        """색상 변환 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("색상 도메인을 변환합니다."))
        
        convert_group = QGroupBox("변환 설정")
        convert_layout = QGridLayout(convert_group)
        
        # 소스 색상 공간
        convert_layout.addWidget(QLabel("소스:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["BGR", "RGB", "HSV", "LAB"])
        convert_layout.addWidget(self.source_combo, 0, 1)
        
        # 타겟 색상 공간
        convert_layout.addWidget(QLabel("타겟:"), 1, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["HSV", "RGB", "LAB", "GRAY", "HSI"])
        convert_layout.addWidget(self.target_combo, 1, 1)
        
        layout.addWidget(convert_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "색상 변환")
    
    def create_enhance_tab(self):
        """이미지 향상 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("이미지 품질을 향상시킵니다."))
        
        enhance_group = QGroupBox("향상 설정")
        enhance_layout = QGridLayout(enhance_group)
        
        # 밝기
        enhance_layout.addWidget(QLabel("밝기:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        self.brightness_label = QLabel("1.0")
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(f"{v/100:.1f}")
        )
        enhance_layout.addWidget(self.brightness_slider, 0, 1)
        enhance_layout.addWidget(self.brightness_label, 0, 2)
        
        # 대비
        enhance_layout.addWidget(QLabel("대비:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_label = QLabel("1.0")
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(f"{v/100:.1f}")
        )
        enhance_layout.addWidget(self.contrast_slider, 1, 1)
        enhance_layout.addWidget(self.contrast_label, 1, 2)
        
        # 채도
        enhance_layout.addWidget(QLabel("채도:"), 2, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_label = QLabel("1.0")
        self.saturation_slider.valueChanged.connect(
            lambda v: self.saturation_label.setText(f"{v/100:.1f}")
        )
        enhance_layout.addWidget(self.saturation_slider, 2, 1)
        enhance_layout.addWidget(self.saturation_label, 2, 2)
        
        # 선명도
        enhance_layout.addWidget(QLabel("선명도:"), 3, 0)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 200)
        self.sharpness_slider.setValue(100)
        self.sharpness_label = QLabel("1.0")
        self.sharpness_slider.valueChanged.connect(
            lambda v: self.sharpness_label.setText(f"{v/100:.1f}")
        )
        enhance_layout.addWidget(self.sharpness_slider, 3, 1)
        enhance_layout.addWidget(self.sharpness_label, 3, 2)
        
        layout.addWidget(enhance_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "이미지 향상")
    
    def create_filter_tab(self):
        """필터 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("다양한 필터를 적용합니다."))
        
        filter_group = QGroupBox("필터 설정")
        filter_layout = QGridLayout(filter_group)
        
        # 필터 종류
        filter_layout.addWidget(QLabel("필터 종류:"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["blur", "sharpen", "edge", "emboss"])
        filter_layout.addWidget(self.filter_combo, 0, 1)
        
        # 커널 크기 (블러용)
        filter_layout.addWidget(QLabel("강도:"), 1, 0)
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(3, 15)
        self.kernel_size_spin.setValue(5)
        self.kernel_size_spin.setSingleStep(2)  # 홀수만
        filter_layout.addWidget(self.kernel_size_spin, 1, 1)
        
        layout.addWidget(filter_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "필터")
    
    def create_video_edit_tab(self):
        """비디오 편집 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("비디오를 시간 범위로 자릅니다."))
        
        video_group = QGroupBox("자르기 설정")
        video_layout = QGridLayout(video_group)
        
        # 시작 시간
        video_layout.addWidget(QLabel("시작 시간 (초):"), 0, 0)
        self.start_time_spin = QSpinBox()
        self.start_time_spin.setRange(0, 3600)
        self.start_time_spin.setValue(0)
        video_layout.addWidget(self.start_time_spin, 0, 1)
        
        # 지속 시간
        video_layout.addWidget(QLabel("지속 시간 (초):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(10)
        video_layout.addWidget(self.duration_spin, 1, 1)
        
        layout.addWidget(video_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "비디오 편집")
    
    def create_analysis_tab(self):
        """분석 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("이미지 분석 및 히스토그램을 생성합니다."))
        
        analysis_group = QGroupBox("분석 옵션")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.histogram_check = QCheckBox("히스토그램 생성")
        self.histogram_check.setChecked(True)
        analysis_layout.addWidget(self.histogram_check)
        
        self.metadata_check = QCheckBox("메타데이터 추출")
        analysis_layout.addWidget(self.metadata_check)
        
        self.color_analysis_check = QCheckBox("색상 분포 분석")
        analysis_layout.addWidget(self.color_analysis_check)
        
        layout.addWidget(analysis_group)
        
        # 업스케일링 옵션
        upscale_group = QGroupBox("이미지 업스케일링")
        upscale_layout = QGridLayout(upscale_group)
        
        upscale_layout.addWidget(QLabel("배율:"), 0, 0)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(2, 8)
        self.scale_spin.setValue(2)
        upscale_layout.addWidget(self.scale_spin, 0, 1)
        
        upscale_layout.addWidget(QLabel("방법:"), 1, 0)
        self.upscale_method_combo = QComboBox()
        self.upscale_method_combo.addItems(["INTER_CUBIC", "INTER_LINEAR", "INTER_LANCZOS4"])
        upscale_layout.addWidget(self.upscale_method_combo, 1, 1)
        
        layout.addWidget(upscale_group)
        
        # 노이즈 감소
        noise_group = QGroupBox("노이즈 감소")
        noise_layout = QGridLayout(noise_group)
        
        noise_layout.addWidget(QLabel("방법:"), 0, 0)
        self.noise_method_combo = QComboBox()
        self.noise_method_combo.addItems(["bilateral", "gaussian", "median", "nlmeans"])
        noise_layout.addWidget(self.noise_method_combo, 0, 1)
        
        noise_layout.addWidget(QLabel("강도:"), 1, 0)
        self.noise_strength_spin = QSpinBox()
        self.noise_strength_spin.setRange(1, 50)
        self.noise_strength_spin.setValue(10)
        noise_layout.addWidget(self.noise_strength_spin, 1, 1)
        
        layout.addWidget(noise_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "분석/향상")
    
    def select_file(self):
        """파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "파일 선택", "",
            "미디어 파일 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv);;모든 파일 (*)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(Path(file_path).name)
    
    def start_processing(self):
        """처리 시작"""
        if not self.file_path or not Path(self.file_path).exists():
            QMessageBox.warning(self, "경고", "유효한 파일을 선택하세요.")
            return
        
        # 현재 탭에 따라 작업 결정
        current_tab_index = self.tab_widget.currentIndex()
        tab_names = ["channel_split", "domain_convert", "enhance", "filter", "crop_video", "analysis"]
        operation = tab_names[current_tab_index]
        
        # 출력 파일 경로 설정
        input_path = Path(self.file_path)
        if operation == "analysis":
            output_path = input_path.parent / f"{input_path.stem}_analysis.png"
        else:
            output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        
        # 파라미터 수집
        parameters = self.collect_parameters(operation)
        
        # 처리 시작
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        
        self.processor = AdvancedProcessor(
            str(self.file_path), str(output_path), operation, parameters
        )
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.result.connect(self.on_processing_finished)
        self.processor.error.connect(self.on_processing_error)
        self.processor.start()
    
    def collect_parameters(self, operation):
        """작업별 파라미터 수집"""
        parameters = {}
        
        if operation == "channel_split":
            parameters['color_space'] = self.channel_color_combo.currentText()
        
        elif operation == "domain_convert":
            parameters['source'] = self.source_combo.currentText()
            parameters['target'] = self.target_combo.currentText()
        
        elif operation == "enhance":
            parameters['brightness'] = self.brightness_slider.value() / 100.0
            parameters['contrast'] = self.contrast_slider.value() / 100.0
            parameters['saturation'] = self.saturation_slider.value() / 100.0
            parameters['sharpness'] = self.sharpness_slider.value() / 100.0
        
        elif operation == "filter":
            parameters['filter_type'] = self.filter_combo.currentText()
            parameters['kernel_size'] = self.kernel_size_spin.value()
        
        elif operation == "crop_video":
            parameters['start_time'] = self.start_time_spin.value()
            parameters['duration'] = self.duration_spin.value()
        
        elif operation == "analysis":
            # 현재는 히스토그램만 구현
            parameters['histogram'] = self.histogram_check.isChecked()
            # 다른 분석 옵션도 추가할 수 있음
            
            # 업스케일링 파라미터도 포함
            parameters['scale_factor'] = self.scale_spin.value()
            parameters['upscale_method'] = self.upscale_method_combo.currentText()
            
            # 노이즈 감소 파라미터
            parameters['noise_method'] = self.noise_method_combo.currentText()
            parameters['noise_strength'] = self.noise_strength_spin.value()
        
        return parameters
    
    def on_processing_finished(self, message, output_path):
        """처리 완료"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.result_text.append(f"✅ {message}")
        self.result_text.append(f"📁 저장 위치: {output_path}")
        
        # 결과 파일 열기 옵션
        reply = QMessageBox.question(
            self, "처리 완료", 
            f"{message}\n\n결과 파일을 열어보시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            import os
            os.startfile(output_path)  # Windows
    
    def on_processing_error(self, error):
        """처리 오류"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.result_text.append(f"❌ 오류: {error}")
        QMessageBox.critical(self, "오류", error)