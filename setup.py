from setuptools import setup, find_packages
from pip.req import parse_requirements
import uuid

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt", session=uuid.uuid1())

reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='caffemachine',
    version='0.0.1',
    description='A caffe experiment runner',
    url='https://github.com/berleon/caffemachine',
    author='Leon Sixt',
    author_email='caffemachine@leon-sixt.de',
    license='Apache2',
    packages=['caffemachine'],
    install_requires=reqs,
    entry_points={
        'console_scripts': [
            'caffemachine=coffeemachine.main:main',
        ],
    },
)
