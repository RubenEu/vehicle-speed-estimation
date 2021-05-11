#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'numpy', 'opencv-python']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Rubén García Rojas",
    author_email='garcia.ruben@outlook.es',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Modelos de estimación de velocidad de vehículos en un plano utilizando técnicas dehomografía y suavizados de posiciones.",
    entry_points={
        'console_scripts': [
            'vehicle_speed_estimation=vehicle_speed_estimation.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='vehicle_speed_estimation',
    name='vehicle_speed_estimation',
    packages=find_packages(include=['vehicle_speed_estimation', 'vehicle_speed_estimation.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/rubeneu/vehicle_speed_estimation',
    version='1.1.0',
    zip_safe=False,
)
