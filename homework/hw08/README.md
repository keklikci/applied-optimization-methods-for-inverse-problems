## Overview

* Algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* With this release, I put out my best effort out there to re-implement some of my methods so they are optimized by specifically assessing the objective function value given the current iterate. 

* I also configured the main [Optimization](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/optimize.py) class that all my methods inherit from to take phantom `7c` as the default target, as it was one of my observations that everyone that took part in the presentations mainly focused on that.

* By default, I set the objective function for solving LASSO problem. To achieve this, I implemented the least squares method in my development folder [aomip](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/aomip) and calculated the norm on-the-fly. Least squares method, in modular fashion, can be found in [objective.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/objective.py).

* Furthermore, it is worth noting down that I modified my usual setup to return a cost history for convergence analysis so I returned a `tuple` instead of `np.ndarray` x where x is the reconstruction. Please note that I implemented this change for recent methods that were working stable and reliable enough for me to plot the objective function values of the following.
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

* Concerete implementation of the Subgradient resides in [gradient.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py). In context of the assignment, we were required to implement this method for three different step selections and analyze the convergence of it. For computing the subgradient, I used `np.sign(l1(dx))` where `dx` is the derivate in all directions and `l1` is the 1-norm.

* For constant step size (α), please run [subgradient.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/subgradient.py) to save reconstruction images to relevant subfolders dynamically created in [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw08/images). 

* For experimenting with other step size (α) configurations such as square summable or diminishing, please run [experiment.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/experiment.py) to save reconstruction images to relevant subfolders dynamically created in [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw08/images). 

* Below are the [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) reconstructions and their convergence plots tested on different  but constant step sizes (α-values).

![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/constant/lr_0.0001.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/convergence/constant/convergence_0.0001.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/constant/lr_0.001.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/convergence/constant/convergence_0.001.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/constant/lr_1e-05.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/convergence/constant/convergence_1e-05.png)

* When observed under fixed amount of iterations, we can see that it takes roughly 7-8 iterations for subgradient method to converge with α = 0.001, which produces the best and smoothest reconstruction among others. For α = 0.0001, convergence happens around 35-40 iterations. For α = 0.0001, the loss is still decreasing and some form of reconstruction is visible but subgradient is not a descent method, therefore there is no guarantee that reconstruction is better of when n-iterations is increased. In fact, that was not the case.

* For square summable but not summable step sizes (α), I set the step size to the following.

    * `α_k = a/(b + k), where a > 0 and b ≥ 0.`
    * In my implementation, I set `a = 1` and `b = (i + 1e3), where i is the i-th iteration.` In the formulation above, `i` corresponds to `k`.
    * I referenced this [lecture](https://web.stanford.edu/class/ee392o/subgrad_method.pdf) to satisfy this condition.

* Below is the reconstruction and convergence plot associated with this α setting.     
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/square_summable/square_summable.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/convergence/square_summamble/convergence.png)

* As observed, convergence behavior is very similar to setting α = 0.001 (constant), meaning that this works out better than other constant step sizes and loss becomes 0 around 5 iterations.
    
* For nonsummable but diminishing step sizes (α), I set the step size to the following.

    * `α_k = a/√k, where a > 0.`
    * In my implementation, I set `a = 1` and `b = np.sqrt((i + 1e6)), where i is the i-th iteration.` In the formulation above, `i` corresponds to `k`.
    * I referenced this [lecture](https://web.stanford.edu/class/ee392o/subgrad_method.pdf) to satisfy this condition.
    
* Below is the reconstruction and convergence plot associated with this α setting.     
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/diminishing/nonsummable_diminishing.png)
![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/convergence/diminishing/convergence.png)

* Similar remarks to square summable and constant α = 0.001 persist because I set the step size (α) configuration carefully, these selections work out.

* At this point, please run [experiment.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/experiment.py) if you haven't already to save the comparison plot between [Fast Proximal Gradient Method (FPGM)](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py) and [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) method. The reason why this analysis was carried out with [Fast Proximal Gradient Method (FPGM)](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py) instead of [ADMM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/admm.py) was clarified in the sections above.

* Below is a plot that outlines the difference between reconstructions and their convergence. Clearly, [FPGM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py) reconstruction quality just in 10 iterations compared to [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) in 100 iterations is much better. Although the loss is decreasing, because of using momentum, it's not wise with this method to exceed 500 iterations to my experience. Reconstruction quality-wise, judging from the [ADMM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/admm.py) images provided in the Overview section, [ADMM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/admm.py) is also better than the [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) method but worse than [FPGM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py). 

* Script also prints out the elapsed time for both methods. Owed to fast reconstruction, [FPGM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py) takes around 2 seconds compared to [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) taking roughly around 8 seconds for convergence.

    * | Method      | # of Iterations | Elapsted Total Time (seconds) | Convergence Rate 
      | ---------   | ----------------| ------------------            | -----------
      | FPGM        |       10        | ~ 2 seconds                   | O(1/k^2)
      | Subgradient |       100       | ~ 10 seconds                  | O(1/√k)

* Analysis is satisfactory because FPGM convergence rate can be as fast as `O(1/k^2)` [FPGM](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py) whereas [Subgradient](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/gradient.py) convergence is as slow as `O(1/√k)`.

![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/notebook/subgradient/comparison/comparison.png)

## Homework 2: Challenge

![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/leaderboard/leaderboard.png)

* By now, you will have already run [experiment.py](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/experiment.py), which will also save my challenge submission for the full-arc (360) to [challenge](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw08/images/challenge).

* Below is the challenge image, reconstructed by [Fast Proximal Gradient Method (FPGM)](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/fpgm.py). This puts me in the **first place** in the [Ultimate leaderboard](https://submission.ciip.in.tum.de/difficulty-07/arc-360.html) as of the date of this release.

* You will find two files saved to [challenge](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw08/images/challenge), with one of them being the `.png` export below and the other the `.tif` export for submision to the challenge.

* Additionally, I implemented [masking](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/mask.py) and used it for lower angles to boost my scores there which bumped up my reconstructions around 15-20%.

    * | Method        | Arc | Difficulty | Phantom | Score
      | ---------     | ----| ---------- | ------- | -----
      | Configuration | 360 | 7          | c       | 0.998811

![](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/raw/main/homework/hw08/images/challenge/challenge.png)

## Homework 3: Presentation

* My final presentation that I gave out in-class can be found [here](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw08/Practical%20Course%20-%20Applied%20Optimization%20Methods%20for%20Inverse%20Problems.pdf).

