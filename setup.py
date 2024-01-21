import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vadmapper',
    version='0.0.3',
    author='Wilton Beltre',
    author_email='beltre.wilton@gmail.com',
    description='Simple mapper class from vad model to categorical.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/beltrewilton/VAD',
    project_urls={
        "Bug Tracker": "https://github.com/beltrewilton/VAD/issues"
    },
    include_package_data=True,
    packages=['vad'],
    package_dir={'vad': 'vad'}, # the one line where all the magic happens
    package_data={
      'vad': ['*.csv'],
    },
    license='MIT',
    install_requires=['numpy', 'pandas', 'plotly'],
)