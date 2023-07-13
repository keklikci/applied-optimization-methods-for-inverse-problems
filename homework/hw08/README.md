## Overview

* Algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* With this release, I put out my best effort out there to re-implement some of my methods so they are optimized by specifically assessing the objective function value given the current iterate.

* By default, I set the objective function for solving LASSO problem. To achieve this, I implemented the least squares method in my development folder [aomip](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/aomip) and calculated the norm on-the-fly. Least squares method, in modular fashion, can be found in [objective.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/objective.py).

* Furthermore, it is worth noting down that I modified my usual setup to return a cost history for convergence analysis so I returned a `tuple` instead of `np.ndarray` x where x is the reconstruction. Please note that I implemented this change for methods that were working stable and reliable enough for me to plot the objective function values of the following.
    * [Fast Proximal Gradient Method (FPGM)](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py)
    * [Gradient Descent](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py)
    * [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py)

* In fact, [ADMM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/admm.py) was also a part of this list as you can verify from some of my recent commits but I was not satisfied by the oscillations in the cost history and the algorithm in general is not stable enough to make a comment on my side. I do get nice reconstructions out of it as it can be seen from my [previous homework images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07/images) with different 𝜏 values, however the objective function suggests it is not being optimized properly. Nonetheless, below are the ADMM output images from previous homework, as this is stated as a requirement for one of the tasks in the assignment sheet.

![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_0.001.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_0.01.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_0.1.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_1.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_10.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_100.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_1000.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_10000.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_100000.0.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw07/images/notebook/tau_1000000.0.png)

* As it can be easily inferred that reconstructions are quite similar and almost independent of 𝜏 values. This could be the case if the proximal operators (specifically f-proximal) works near perfectly to proxy the intermediate value to the current iterate. At the very least, this is the only plausible explanation I could structure around my ADMM implementation.


* With that being said, I want to talk about some of the refinements I have done since the feedback I gathered from the presentation sessions and the overall expectation from the assignment. So far, I had implemented every optimization method and had some of my best reconstructions with the proximal methods. What was missing though in some of my submissions was a nice convergence analysis that tracks the cost history. I usually handled this by just checking the norm of reconstruction error, similar to a [CrossValidated thread](https://stats.stackexchange.com/questions/130721/what-norm-of-the-reconstruction-error-is-minimized-by-the-low-rank-approximation) that addresses this issue in a different context. With this homework and by looking at what some of my friends had produced at the presentation, I added convergence analysis to the most recent methods I've implemented. Above the ADMM images, the methods that have support for convergence analysis are listed down.

* To conclude the Overview section, I also wanted to address another topic, a more structural one so that more work is better understood and my effort is justified in terms of finishing the course with my best work. After some feedback on [hw01](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw01) and [hw02](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw02), I considered putting all my images, which always reside in the [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07/images) folder of a given homework but I actually thought this was a bad idea because there are too many images and it would be difficult to judge / compare reconstructions across algorithms, which is why I stored all my images in a folder where I always indicated in the README.md to that folder as the image source for my reconstructions. 

* I don't know exactly if that was practically a bad decision. Maybe it's more intuitive to view images at a time from the folder and compare what you want to compare at a time instead of having to scroll through `README.md` back and forth. Nonetheless, I really took the time off to structure / re-structure my homeworks, specifically the last 4 where I adjusted my `README.md` files and added Jupyter notebooks to view the images better in tables, in a format which I deemed more convenient than what was proposed. I really hope you can take a look at my work in some retrospective fashion to observe at least some of the development I had in a very short amount of time, as soon as I had some feedback from the presentations.

* Inside the homework folder, you will also see a [Gradient Descent](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) implementation, I executed the descent method just to view what to expect out of my [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/subgradient.py).

* In general, I leave you here with some of my work that accounts for everything requested in a given assignment. I would appreciate if you could go back in my repository and at least have some sort of overview of the concepts I introduced, i.e. adjustment of `README.md` files and `Jupyter notebooks`.

* Every script was executed from the [hw08](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07), utilizing the concrete implementations in the top-level directory.

* Every task is implemented including proper convergence analysis, reconstruction quality and runtime performance among [Fast Proximal Gradient Method (FPGM)](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py), [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) and [ADMM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/admm.py)**.

    * **reconstruction-quality wise 

* All implementation is available under [hw08](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07). Please refer to the aforementioned scripts for concrete implementations that reside in [aomip](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/aomip).

* My final presentation that I gave out in-class can be found [here](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/Practical%20Course%20-%20Applied%20Optimization%20Methods%20for%20Inverse%20Problems.pdf).

## Homework 1: Subgradient Method

* 
