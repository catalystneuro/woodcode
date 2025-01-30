from setuptools import setup, find_packages


setup(
    name="woodcode",
    version="0.1.0",
    author="Adrian Duszkiewicz",
    author_email="a.j.duszkiewicz@gmail.com",
    description="A python package for neural data analysis in Wood/Dudchenko lab",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Wood-Dudchenko-lab/woodcode",
    packages=find_packages(include=["woodcode", "woodcode.*"]),  # Includes all submodules  # Automatically finds packages inside the repo
    install_requires=[  # Dependencies your package needs
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pynapple==0.7.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)