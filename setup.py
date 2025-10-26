from setuptools import setup, find_packages

setup(
    name="fraudguard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
    ],
)
