## Important
* Please execute all scripts from repository root directory.
* Concrete implementations are replicated in `aomip` directory as described in the description.
* In general, scripts generate the desired output once executed. For future work, sample image generation for deliverables may be handled in `tests`.

## Requirements
* Inside the homework file, you will find a `requirements.txt` file for dependencies.
* One of the most important dependencies is the [python-dotenv]("https://pypi.org/project/python-dotenv/") package, which configures the script variables from a sourced `.env` file. To establish this, `.gitignore` was modified accordingly.
* Please download this [dataset](https://zenodo.org/record/2688112#.ZFBTsOxByu4) and place it under a folder named `datasets` in repository root.

## Homework 3, Task 1: Flat-field correction
* Execute the following script to generate the output for flat-field correction.
    * `python -B homework/hw01/preprocessor.py`
* Script loads the [raw](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/raw) input files, stores them into matrices and performs flat-field correction given the formula. 
* Below is a sample output for flat-field correction, with raw image on the left and flat-field corrected output on the right. Complete set of results can be found by accessing this [hyperlink](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/flat_field_corrected).

Raw Image        |  Flat-field corrected
:-------------------------:|:-------------------------:
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/raw/0001.png)  |  ![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/flat_field_corrected/0001.png)
* Note that since there are two flat field images, they are averaged to get a reasonable estimate. In contrast, only one dark image exists.

## Homework 3, Task 2: Transmission to Absorption conversion
* Execute the following script to generate the output for the transformations.
    * `python -B homework/hw01/transform.py`
* Script will load the files, estimate an initial density by sampling some pixels from the input image, and implement [Beer-Lambert](https://www.edinst.com/blog/the-beer-lambert-law/) law.
* As specified in the homework statement, for simulation purposes, an inverse is also implemented.


