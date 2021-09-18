from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  name = 'ImageSeeker',
  packages = find_packages(),
  include_package_data=True,
  version = '0.7',
  license='MIT',
  description = 'Its an auto image classification library',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/entbappy/ImageSeeker-Package",
  author = 'Bappy Ahmed',
  author_email = 'entbappy73@gmail.com',
  keywords = ['imageseeker'],
  install_requires=[
        'tensorflow==2.4.1',
        'scipy==1.6.3',
        'numpy',
        'pandas',
        'Pillow==8.3.2',
        'Flask',
        'Flask-Cors==3.0.10'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  entry_points={
        "console_scripts": [
            "imgseeker = ImageSeeker.main:start_app",
        ]},
)
