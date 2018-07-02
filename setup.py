import setuptools

REQUIRED_PACKAGES = ['torch==0.4.0', 'numpy==1.14.3', 'scikit-learn==0.19.1']

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swae",
    version="0.1.0",
    author="Emmanuel Fuentes",
    description="Pytorch Sliced Wasserstein Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eifuentes/swae-pytorch",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=setuptools.find_packages()
)
