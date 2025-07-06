from setuptools import setup, find_packages

setup(
    name="EasyMediaProcessor",
    version="1.0.0",
    description="영상 및 이미지 처리 도구",
    author="Your Name",
    author_email="crosefrog@naver.com",
    packages=find_packages(),
    install_requires=[
        "PySide6>=6.0.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.20.0",
        "tensorflow>=2.8.0",
        "google-api-python-client>=2.0.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=0.5.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "easymediaprocessor=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)