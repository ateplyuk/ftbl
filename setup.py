from setuptools import setup

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setup(name='tst',
      version='1.0.0',
      description='test',
      url='https://github.com/ateplyuk/ftbl',
      packages=['ftbl'],
      install_requires=requirements
      )