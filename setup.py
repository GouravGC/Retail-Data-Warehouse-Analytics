"""
Setup script for the Retail Data Warehouse Analytics project.

Install with:
    pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="retail_data_warehouse_analytics",
    version="1.0.0",
    author="GCS Portfolio Projects",
    description="End-to-end retail analytics pipeline with SQL + ML",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GCS/retail-data-warehouse-analytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
        "xgboost>=1.5",
        "shap>=0.40",
        "joblib>=1.1",
        "streamlit>=1.15",
        "pathlib>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
