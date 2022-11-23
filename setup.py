# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup


# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="mlfromscratch",
    version='0.0.1',
    description="Implementation of different machine learning methods.",
    author="Giannis Manousaridis",
    author_email="giannismanu97@gmail.com",
    url="",
    python_requires=">=3.7",
    install_requires=[required_packages],
)
