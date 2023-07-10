import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vadmapper',
    version='0.0.1',
    author='Wilton Beltre',
    author_email='beltre.wilton@gmail.com',
    description='Simple mapper class from vad model to categorical.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/beltrewilton/VAD',
    project_urls = {
        "Bug Tracker": "https://github.com/beltrewilton/VAD/issues"
    },
    license='MIT',
    packages=['vad'],
    install_requires=['numpy', 'pandas', 'plotly'],
)