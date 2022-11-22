from setuptools import setup, find_packages

setup(name                = 'wisardpy',
      version             = '1.0',
      author              = 'Maurizio Giordano',
      author_email        = 'maurizio.giordano@cnr.it',
      maintainer          = 'Maurizio Giordano',
      maintainer_email    = 'maurizio.giordano@cnr.it',
      description         = 'WiSARD Classifier and Regressor in Python',
      long_description    = """WiSARD Classifier and Regressor in Python
          """,
      platforms           = ['Mac OSX', 'POSIX',],
      classifiers         = [
                             "Programming Language :: Python :: 3",
                             "Operating System :: OS Independent"
                             ],
      license             = 'GNU Lesser General Public License v2 or later (LGPLv2+)',

      install_requires=[
                        'scipy>=1.5.2',
                        'scikit_learn>=1.0.2',
                        'numpy>=1.14.5',
                        'matplotlib>=3.3.2'
                        ],
      packages            = ['base'],
      url='https://github.com/giordamaug/WiSARDpy',
      )
