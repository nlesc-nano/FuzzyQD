from setuptools import setup

setup(
    name='fuzzyqd',
    version='0.1.0',
    author='Zeger Hens and Ivan Infante',
    author_email='zeger.hens@ugent.be, ivan.infante@bcmaterials.net',
    description='A package for Bloch state expansion and quantum dot fuzzy band structures',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nlesc-nano/fuzzyqd',  # Replace with your GitHub URL
    py_modules=['fuzzyqd', 'funcs', 'logger_config'],  # Specify your module names here
    package_dir={'': 'src'},  # Maps the module root to src
    entry_points={
        'console_scripts': [
            'fuzzyqd=fuzzyqd:main',  # Maps the 'fuzzyqd' command to fuzzyqd.py's main() function
        ],
    },
    scripts=[
        'analysis/plot_fuzzyqd.py',
        'analysis/process_pickles.py'
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'h5py',
        'PyYAML',
        'scipy',
        'torch',
        'joblib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

