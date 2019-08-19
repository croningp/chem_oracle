from setuptools import setup, find_packages

setup(name="MSAnalyze",
      version="0.1",
      description="HPLC-MS analysis library",
      url="http://datalore.chem.gla.ac.uk/HessamMehr/MSAnalyze",
      author="Hessam Mehr, Dario Caramelli",
      author_email="Hessam.Mehr@glasgow.ac.uk",
      license="MIT",
      install_requires=["advion-wrapper", "numpy", "scikit-learn", "scipy"],
      packages=find_packages())