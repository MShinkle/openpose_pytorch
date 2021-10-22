import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openpose_pytorch", # Replace with your own username
    version="0.0.1",
    author="Matthew Shinkle",
    author_email="mshinkle@nevada.unr.edu",
    description="pytorch openpose wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MShinkle/pytorch-openpose",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
