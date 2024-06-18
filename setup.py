from setuptools import find_packages, setup

__version__ = "1.0.0"

setup(
    name="protonmof",
    version='1.0',
    license='KAIST',
    author='Seunghee Han',
    author_email= 'sisifhro@kaist.ac.kr',
    install_requires = [

        'moftransformer',
        'simpletransformers',
        'torch == 1.13.1',
        'pytorch-lightning==1.7.0',
        'easydict==1.10',
        'pyyaml==6.0',

    ]
    )
