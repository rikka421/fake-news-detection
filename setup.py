"""
Setup script for News Classification Analysis Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="data-analysis-progress",
    version="0.1.0",
    author="rikka421",
    author_email="3550124064@qq.com",
    description="Fake-news detection experiments with classical and deep learning baselines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rikka421/data-analysis-progress",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "deeplearning": [
            "torch>=2.0.0",
            "transformers>=4.25.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "data-analysis-progress=data_analysis_progress.cli:main",
            "data-analysis-api=data_analysis_progress.api:main",
        ],
    },
)