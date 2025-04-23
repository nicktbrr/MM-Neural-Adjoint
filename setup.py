from setuptools import setup, find_packages

setup(
    name="mm-neural-adjoint",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        # For example:
        # "numpy>=1.21.0",
        # "torch>=1.9.0",
    ],
    author="Nicholas Barsi-Rhyne",
    author_email="your.email@example.com",
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