from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="p3r-headgate",
    version="1.0.0",
    author="Fahim Sultan",
    author_email="fahim.sultan34167@gmail.com",
    description="Parameter-Efficient Fine-tuning for Code Language Models using P3R and HeadGate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FahimSultan-cyb/P3R",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
    ],
    keywords="transformer, code, language model, parameter efficient, fine-tuning",
)
