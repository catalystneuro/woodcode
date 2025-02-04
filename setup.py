from setuptools import setup, find_packages

setup(
    name="woodcode",
    version="0.1.1",
    author="Wood/Dudchenko lab members",
    author_email="a.j.duszkiewicz@gmail.com",
    description="A python package for neural and behavioural data analysis in Wood/Dudchenko lab",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Wood-Dudchenko-lab/woodcode",
    packages=find_packages(include=["woodcode", "woodcode.*"]),  # Includes all submodules  # Automatically finds packages inside the repo
    install_requires=[  # Dependencies your package needs
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pynwb",
        "openpyxl",
        "hdmf",
        "h5py",
        "pytz",
        "pynapple==0.7.1",
        "selenium"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)