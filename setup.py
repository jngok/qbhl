from setuptools import setup, find_packages

setup(
    name='qbhl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # 실제 qbhl 라이브러리에 필요한 종속성 목록을 여기에 추가하세요.
        # 예: 'pandas', 'numpy', 'scikit-learn'
    ],
    author='jngok',
    description='A library for Quality-Based Hierarchical Legal/Law analysis.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jngok/qbhl',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
