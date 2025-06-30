from setuptools import find_packages, setup

package_name = 'llm_fatigue_handling'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llm_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ibtahaj',
    maintainer_email='ibtahaj.athar@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_node = llm_fatigue_handling.llm_node:main',
        ],
    },
)
