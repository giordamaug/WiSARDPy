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
                             'Machine Learning',
                             ],
      license             = 'GNU Lesser General Public License v2 or later (LGPLv2+)',

      packages            = find_packages(include=['wisardpy', 'wisardpy.*']),
      url='https://github.com/giordamaug/WiSARDpy',
      )
