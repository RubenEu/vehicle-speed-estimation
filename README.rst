========================
Vehicle Speed Estimation
========================


.. image:: https://img.shields.io/pypi/v/vehicle_speed_estimation.svg
        :target: https://pypi.python.org/pypi/vehicle_speed_estimation

.. image:: https://img.shields.io/travis/rubeneu/vehicle_speed_estimation.svg
        :target: https://travis-ci.com/rubeneu/vehicle_speed_estimation

.. image:: https://readthedocs.org/projects/vehicle-speed-estimation/badge/?version=latest
        :target: https://vehicle-speed-estimation.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Speed estimation models for vehicles using homography and kernel regression.


* Free software: MIT license
* Documentation: https://vehicle-speed-estimation.readthedocs.io.


Features
--------

* Wrapper for speed estimation models.
* Homography tools to work with the sequence in the homography.
* Average speed estimation model based on the initial and final position.
* Instantaneous speed based on the increments of positions.
* Nadaraya-watson estimator derivated to calculate the instantaneous speed.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
