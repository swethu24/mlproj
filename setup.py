from setuptools import find_packages,setup
from typing import List

def get_requirements(path)-> List[str]:
    """ Function reads requirements and returns a list"""
    requirements = []
    with open(path) as f:
        requirements = f.readlines()
        [req.replace("\n","") for req in requirements]
setup(
    name= 'mlproject',
    version = '0.0.1',
    author = "Swetha",
    author_email = "swethamurali2402@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)