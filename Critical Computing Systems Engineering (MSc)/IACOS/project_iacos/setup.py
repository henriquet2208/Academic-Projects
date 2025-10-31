from setuptools import setup
import os
from glob import glob

package_name = 'project_iacos'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),  # Include launch files
        ('share/' + package_name + '/launch', glob('launch/*.world')),  # Include world files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ATC sign detection and vehicle control',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sign_detector = project_iacos.sign_detector:main',
            'vehicle_controller = project_iacos.vehicle_controller:main',
        ],
    },
)
