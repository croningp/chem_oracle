from setuptools import setup, find_packages

setup(
    name="chem-oracle",
    version="0.1",
    description="Chemical space exploration algorithm for the discovery rig",
    url="http://datalore.chem.gla.ac.uk/HessamMehr/ChemOracle",
    author="Hessam Mehr, Dario Caramelli",
    author_email="Hessam.Mehr@glasgow.ac.uk",
    license="MIT",
    install_requires=["advion-wrapper", "numpy", "scikit-learn", "scipy"],
    packages=find_packages(),
)
