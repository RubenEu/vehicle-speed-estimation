from setuptools import setup, find_packages

setup(
    name='vehicle-speed-estimation',
    version='0.0.1',
    description='Utils to work with homography, image distances and speed estimation.',
    author='Rubén García Rojas',
    author_email='garcia.ruben@outlook.es',
    packages=find_packages(),
    install_requires=[
        'simple-object-detection',
        'simple-object-tracking',
        'opencv-python',
        'numpy',
    ]
)
