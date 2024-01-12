from setuptools import setup, find_packages
from glob import glob

package_name = 'stereovoxelnet'
node_name =  'stereovoxelnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
	    ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name + '/params/', glob('params/*.yml')),
        ('share/' + package_name + '/resource/', glob('resource/*.rviz')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    author='Finn Gegenmantel',
    author_email='f.gegenmantel@pem.rwth-aachen.de',
    maintainer='Finn Gegenmantel',
    maintainer_email='f.gegenmantel@pem.rwth-aachen.de',
    keywords=['ROS'],
    classifiers=["gini"],
    description='obstacle clustering from stero images using stereovoxelnet',
    license='PEM',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            '%s=%s.Main:main' % (package_name, node_name),
            'cameradummy=%s.Cameradummy:main' % (node_name),
            'disparity=%s.DisparityExtractor:main' % (node_name),
            'world_visualizer=%s.WorldVisualizer:main' % (node_name),
        ],
    },
)
