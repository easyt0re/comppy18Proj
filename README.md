# comppy18Proj
A Pre-processing Script for PyTorch ML Problem

[![Coverage Status](https://coveralls.io/repos/github/easyt0re/comppy18Proj/badge.svg?branch=master)](https://coveralls.io/github/easyt0re/comppy18Proj?branch=master)
[![Documentation Status](https://readthedocs.org/projects/comppy18proj/badge/?version=latest)](https://comppy18proj.readthedocs.io/en/latest/?badge=latest)

## Intro
The idea is to touch and try out things learned in the course: [Computational Python BB3110](https://comppy-info.readthedocs.io/en/latest/index.html)

Machine Learning / Deep Learning (ML/DL) is a hot topic and many frameworks/platforms/environments are built in Python.
Learning-based methods also need heavy computations, which fits this course nicely.
The overall idea is to develop an Artificial Neural Network (ANN) and train it to fit a function which is the Forward Kinematics of a 6-DOF parallel manipulator in this case.
Forward Kinematics refers to the mapping from input: active joint angles to output: end effector pose.
This is desirable because theoretical derivation of this function is quite hard.
Fortunately, we could build a simulation model and get data out of it.
This becomes the dataset of our training process.

As a first step for this course, we would like to develop a pre-processing script to prepare the data from the simulation software (MATLAB) to match the requirement of PyTorch.
Based on the tutorial and the code offered by PyTorch, we modified necessary parts to work with our dataset.
Through this process, we practise what we learned in this course, for example, git, testing, numpy, pandas, the idea of classes, and so on.

## Method
### Create New Class for Dataset
For this new dataset, we define a new subclass based on `Class Dataset` which is presumably defined in PyTorch.
We modified the function to initialize, get length, and get item.

#### `__init__()`
For each sample point in time, the simulation gives us 18 numbers for input data and 15 numbers for output data.
These are saved in 2 separate \*.csv files.
Each row corresponds to one sample, which would be read by `pandas` as one entry of a `dataframe`.
This is essentially what happens with initialization.

#### `__getitem__()`
This function is supposed to return the item with the given index.
What we want to return is a `dict` which has the keys of `'jointspace'` and `'workspace'`.
This is what is asked by PyTorch and this function would make more sense if the dataset is an image dataset.
In that way, it is memory efficient because the image is only read when required.

### Create New class for Transformations
Apart from reading in the raw data, another important step is to "transform" the data into the right type and shape.
Currently, only a `ToTensor` subclass is created.
This is because what we read is a `ndarray` but what the next step needs is a `Tensor`.
This is a common concept in Learning and it is mostly a matrix with more dimensions in a specific order.
For an image dataset, it would be a tensor with the dimension Channels x Height x Width.
For our dataset, it is only a 1 x n vector.
But we have to have a similar shape nonetheless.
So we use `torch.unsqueeze()` to grow those dimensions.
In that sense, we treat our input as a single channel image with width equals to 1 pixel.

### Create Tests for the Code
One of the new things I have learned from the course is testing.
Since this is a pre-processing script, there is not much to assert if it is true or false.
I chose 2 things to be checked: if data (value) is stored correctly and if data size is correct.
I also try to use `fixture` to run multiple tests on different samples.

For `pytest`, I first tried them locally from the commandline.
Then I tried to use Travis-CI which can be hard to get it going for the first time.
I also combined `coverall` with this because I think this is actually a logical criterion to evaluate for testing.
Testing is supposed to test out every possibility.
And if every line of the code has been run, it at least means the code has gone through all `if` branches.

#### `test_item()`
This test is to assert if the value is correctly stored in the output from the input.
I guess it could be more paramiterized and have better criteria to compare, for example, compare 2 "vectors" directly.
However, this is more like a demo of what I could do istead of a real functional test that serves a meaningful purpose.

#### `test_tran_size()`
This test is to assert if the shape of the output is correct for the next step.
This is rather important and the reason why this pre-processing is needed.
Because the output has to follow the convention of the overall framework.

#### `test_read_all()`
To push the `coverall` percentage, this test is to assert if all the raw data is read into the dataset.
This would be practical if the raw data reading process would throw away some of the entries automatically.
Then this could check the number of samples that has been read in.
However, if that is the case, then correspondency would be a more important thing to check.

### Other Time-Consuming Things
#### Testing
It is really hard to get the test running.
Potential problems could be: dependency problem, import problem, too much stuff happening in the module, how to install certain packages, and so on.
From my `git` history, it is clear that I struggled with Travis-CI.
It actually has 2 different homepages which serve different purposes.
And since I do not know a local or an interactive way to test, every test is done after a `$ git push`.
I would definitely do `pytest` locally but those hiccups related to Travis-CI would show up in this way.
So what actually happened was my test was already running 
but I still spent a large amount of time making it work on Travis-CI.
Apart from `pytest`, I also tried `doctest`.
It had some errors that I did not understand.
I gave it up quite quickly because I would have `pytest` anyway.

#### Documentation
To be honest, I didn't expect documentation generation to be this time consuming.
I thought, behind the curtains, it is just parsing strings with some clever regular expressions.
But it is not.
ReadtheDocs also would like to "run" your code, or at least pass the importing step.
This basically means I have to do everything I did for Travis-CI again with ReadtheDocs.
Compared to Travis-CI, it gives you less control over what it does.
Moreover, it has more restrictions.
It seems that it only support Python 2.7 or 3.5.
And due to the fact that this is the place to host documentations,
it can be hard to search for documentation on itself.
Because some of the keywords won't get you closer to what you want.
It took me quite a while to realize what `mock` is and how to use it.

## Results
From the test, we know that with this simple setup, everything seems to be working properly.
After the data can be correctly read in,
it should also be OK to be read in as batches and iterated through them.
These were tested but commented out in the final version
because of some limitations in Travis-CI.

## Conclusion, Discussion, and Future Work
The conclusion is that the pre-processing is working.

However, I would like to use this section to discuss stuff beyond this project.
First of all, it is not guaranteed that this notion of solving Forward Kinematics with ML is correct or feasible.
But as mentioned previously, this serves as a nice project for the course.
Secondly, if we were to keep working on this, 
the next step would be developing the structure for the ANN.
This would be critical to the success of this method 
given the assumption it is doable theoretically.
And the rest of the tuning process for an ANN follows,
such as random initialization, grid search, and so on.
Given our dataset is quite simple,
maybe it is possible to run on a common PC.
But this is also something we should consider.
Would our dataset be sufficient for the ANN to learn this?
As we know, learning methods rely a lot on data.
Finally, if we really achieve a rather good mapping from the ANN,
how could we make use of it?
As stated, the data is actually from the simulation not the real machine.
And actually, it might not be practical to get huge amount of data from the real machine.
So we have to find a way to "project" this mapping onto the real machine to really make use of this work.

So far, that is some of my visions for the project.
And the immediate future work would be to code and finish the whole training process.

## Acknowledgment
The code is based PyTorch Official Beginner Tutorials and specifically [data loading and processing tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

Thanks to the course for the opportunity to implement this. 
