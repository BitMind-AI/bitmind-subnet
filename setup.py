from setuptools import setup, find_packages

# Define the version directly here instead of importing
__version__ = [line.strip() for line in open("VERSION").readlines()][0]

setup(
    name="bitmind",
    version=__version__,
    author="BitMind",
    author_email="intern@bitmind.ai",
    description="SN34 on bittensor",
    long_description_content_type="text/markdown",
    url="http://bitmind.ai",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    python_requires=">=3.10",
)
