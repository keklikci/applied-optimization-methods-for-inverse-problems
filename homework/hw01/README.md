## Important
* Please execute all scripts from repository root directory.
* Concrete implementations are replicated in `aomip` directory as described in the description.
* In general, scripts generate the desired output once executed. For future work, sample image generation for deliverables may be handled in `tests`.
* Some utility scripts exist under `util` folder inside this homework directory. Scripts inside were implemented for scaling and plotting purposes, in order to obtain higher definitions.

## Requirements
* Inside the homework file, you will find a `requirements.txt` file for dependencies.
* One of the most important dependencies is the [python-dotenv]("https://pypi.org/project/python-dotenv/") package, which configures the script variables from a sourced `.env` file. To establish this, `.gitignore` was modified accordingly.
* Please download this [dataset](https://zenodo.org/record/2688112#.ZFBTsOxByu4) and place it under a folder named `datasets` in repository root.

## Homework 3, Task 1: Flat-field correction
* Execute the following script to generate the output for flat-field correction.
    * `python -B homework/hw01/preprocessor.py`
* Script loads the [raw](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/raw) input files, stores them into matrices and performs flat-field correction given the formula.
* Complete set of flat-field corrected results can be found by accessing this [hyperlink](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/flat_field_corrected).
* Note that since there are two flat field images, they are averaged to get a reasonable estimate. In contrast, only one dark image exists.

## Homework 3, Task 2: Transmission to Absorption conversion
* Execute the following script to generate the output for the transformations.
    * `python -B homework/hw01/transform.py`
* Script will load the files, estimate an initial density by sampling some pixels from the input image, and implement [Beer-Lambert](https://www.edinst.com/blog/the-beer-lambert-law/) law. Number of samples is parametrized.
* As specified in the homework statement, for simulation purposes, an inverse is also implemented.
* Complete set of results can be found by accessing this [hyperlink](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/transformed).

## Homework 3, Task 3: Cleaning the signal
* Truncation is implemented in `truncation.py`.
* Depending on the direction of transformation, cases are handled separately.
* Error handling is also implemented. No other transformation modes besides `absorption` and `transmission` are implemented. If attempted, a `NotImplementedError` is raised.
* A concrete implementation is present in `aomip` folder.

## Homework 3, Task 4: Binning
* Binning is implemented based on 2D signals, specifically for images in this implementation.
* Error handling is implemented. For a correct approach, binning factor must be a power of 2. Otherwise, a `NotImplementedError` is raised and execution terminates.
* Execute the following script to see all functionality in action embedded in the script.
    * `python -B tests/test_binning.py`

## Homework 3, Task 5: Center of rotation correction
* Center of rotation correction is implemented for a 2D signal, i.e., image.
* A rotational offset is sourced from the `.env` file to shift the image over the last axis.
* Execute the following script to save the output associated with the task.
    * `python -B homework/hw01/rotation.py`
* Script takes flat-field corrected images as input, and saves the rotated / shifted images to the corresponding [directory](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/rotated), where the rotational effect can be observed.

## Homework 3, Task 6: Padding
* Execute the following script to save the output associated with the task.
    * `python -B homework/hw01/padding.py`
* Complete set of results can be found by accessing this [hyperlink](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/scan/padded).
* Padding is important because it reduces probability of losing informative pixels from the borders. For instance, during reconstruction, we use filters. If the images are not padded, some information from the borders could get lost.

## Homework 4: Analytical Reconstruction
* Some concrete implementations associated with this part such as `phantom.py` and `sinogram.py` are replicated in `aomip` folder for future use.
* Three different filters, as mentioned in the problem statement, are implemented with reference and inspiration from [sckit-image](https://scikit-image.org/docs/stable/api/skimage.filters.html).
* Reconstruction involves a sequence of steps, with all steps performed in `backproject.py`.
* Execute the following script to save the backprojected output of a Shepp-Logan phantom.
    * `python -B homework/hw01/backproject.py`
* Click on each of these hyperlinks to access the corresponding results.
    * [Shepp-Logan phantom](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/phantom/raw/phantom.png)
    * [Sinogram](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/phantom/sinogram/sinogram.png)
    * [Backprojection](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01/output/phantom/backprojection/backprojection.png)
