"""
Setup script для YANTRA
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
readme = Path("README.md").read_text(encoding="utf-8")

setup(
    name="yantra-ml",
    version="0.1.0",
    author="YANTRA Team",
    author_email="contact@yantra-ml.org",
    description="Deterministic Neural Network on Finite Groups",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yantra-ml/yantra",
    project_urls={
        "Bug Tracker": "https://github.com/yantra-ml/yantra/issues",
        "Documentation": "https://github.com/yantra-ml/yantra#readme",
        "Source Code": "https://github.com/yantra-ml/yantra",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Zero dependencies - только stdlib!
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "examples": [
            "jupyter>=1.0",
            "matplotlib>=3.5",
            "numpy>=1.20",
        ],
    },
    keywords=[
        "machine-learning",
        "deterministic",
        "finite-groups",
        "verification",
        "algebraic-ml",
        "exhaustive-search",
    ],
    include_package_data=True,
    zip_safe=False,
)
