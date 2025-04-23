from setuptools import setup, find_packages

setup(
    name="mm-neural-adjoint",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "tqdm>=4.67.1",
        "mlflow>=2.21.3",
        "scikit-learn>=1.4.0"
    ],
    author="Nicholas Barsi-Rhyne",
    author_email="nick@quantumventura.com",
    description="A neural adjoint method implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MM-Neural-Adjoint",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 