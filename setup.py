from setuptools import setup, find_packages

setup(
    name="qbhl",
    version="0.1.0",
    description="Query-Based Hierarchical Labeling (QBHL) Library",
    author="Bae Jongok",
    author_email="abc@bigdt.co.kr",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "torch",
    ],
    python_requires=">=3.8",
)
