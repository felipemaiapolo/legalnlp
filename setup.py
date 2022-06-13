
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="legalnlp",
    version="1.0.0",
    author="Maia Polo et al.",
    author_email="felipemaiapolo@gmail.com",
    description="Pre-trained language models forthe  Brazilian  legal  language.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/legalnlp21/legalnlp',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['ftfy','wget']
) 
