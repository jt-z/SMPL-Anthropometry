"""
SMPL-Anthropometry Package Setup
"""

from setuptools import setup, find_packages
import os

# 读取README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smpl-anthropometry",
    version="1.0.0",
    author="David Bojanic",
    author_email="",
    description="Measure the SMPL/SMPLX body models and visualize the measurements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DavidBoja/SMPL-Anthropometry",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    package_data={
        "src": [
            "data/smpl/*.json",
            "data/smplx/*.json",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "smpl-measure=src.core.measure:main",
            "smpl-fit-txt=src.fitting.fit_smpl_from_txt_fixed:main",
            "smpl-view-3d=src.visualization.view_smpl_3d:main",
            "smpl-check=tools.check_models:main",
        ],
    },
)
