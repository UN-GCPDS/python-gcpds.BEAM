import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name="gcpds",
    version='0.1',
    packages=["gcpds"],
    author="Daniel Aníbal Loaiza Garzón",
    author_email="daloaizag@unal.edu.co",
    maintainer="Daniel Aníbal Loaiza Garzón",
    maintainer_email="daloaizag@unal.edu.co",
    download_url='',
    install_requires=[
        "certifi==2020.6.20",
        "charset-normalizer==3.3.2",
        "contourpy==1.2.1",
        "cycler==0.12.1",
        "decorator==5.1.1",
        "fonttools==4.54.1",
        "idna==3.3",
        "Jinja2==3.0.3",
        "kiwisolver==1.4.5",
        "lazy_loader==0.4",
        "MarkupSafe==2.1.5",
        "matplotlib==3.9.2",
        "mne==1.7.1",
        "numpy==1.26.4",
        "packaging==24.1",
        "pandas==2.2.2",
        "pillow==10.4.0",
        "platformdirs==4.2.2",
        "pooch==1.8.2",
        "pyparsing==2.4.7",
        "python-dateutil==2.9.0.post0",
        "pytz==2022.1",
        "requests==2.32.3",
        "scipy==1.13.1",
        "six==1.16.0",
        "tqdm==4.66.4",
        "tzdata==2024.1",
        "urllib3==1.26.19"
    ],
    scripts=[
    ],
    include_package_data=True,
    license='Simplified BSD License',
    description="",
    zip_safe=False,
    long_description=README,
    long_description_content_type='text/markdown',
    python_requires='>=3.7',

    #https://pypi.org/classifiers/
    classifiers=[
    ],
)
  
