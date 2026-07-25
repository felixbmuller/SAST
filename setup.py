
from setuptools import find_namespace_packages, setup

setup(
    name='sast',
    packages=find_namespace_packages(include=['sast*', 'bam_poses*']),
    version='0.1.0',
    description='Scene-Aware Social Transformer',
    author='Felix Benjamin Mueller',
    license='',
)
