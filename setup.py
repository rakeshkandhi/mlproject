from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    
    This function returns the requirements of the list
    
    """
    requirements=[]
    with open(file_path) as fp:
        requirements = fp.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
    name="mlproject",
    version = "0.0.1",
    author = "Rakesh Kandhi",
    author_email = "rakeshkandhi1432@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)