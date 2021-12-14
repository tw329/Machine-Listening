# HW0: Environment setup for Machine Listening @ NJIT

**IMPORTANT: PUT YOUR UCID IN THE FILE** `ucid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's ucid 
here, we will check. The ucid should be placed into that file by itself as the only line, 
with lower case letters and no carriage returns. 

The line that is marked UCID_GOES_HERE should be REPLACED with YOUR ucid.

**IF YOU DON'T PUT YOUR UCID INTO THAT FILE PROPERLY YOU WILL NOT RECIEVE CREDIT FOR YOUR WORK!**

In this assignment, the goals are:
- to set up a working programming environment using either `conda` or `virtualenv` environments.
- to get acclimated to the "pull, commit, push" development cycle for git. All programming assignments in this course will be submitted via Github (all free-response quesitons will be submitted via Canvas).
  
## Clone this repository

First, let's clone this repository. We'll use `git` for all submissions in this class. New to `git`? Not to worry, it's quite easy! Here's a [helpful guide](https://guides.github.com/activities/hello-world/). 

To clone this repository install GIT on your computer and copy the link of the repository (find above at "Clone or Download") and enter in the command line:

``git clone YOUR-LINK``

Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## What to do for this assignment

What to do for this assignment is detailed in **problems.md**. To pass the test cases you will edit `code.py`. You will also need to launch JupyterLab and evaluate all of the cells in the example notebook as detailed in **problems.md**.

**You should also create a separate .pdf file with answers to the questions in problems.md. This file should be submitted on Canvas.**

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited.

## Environment setup

The easiest way to get setup is to use [**miniconda**](https://docs.conda.io/en/latest/miniconda.html). 

Install the appropriate version of miniconda for your operating system (take care to select the Python 3 version). After miniconda is installed, you should be able to run `conda`. If you get an error (e.g. `-bash: conda: command not found`), make sure to source your bash file afterwards (`source ~/.bash_profile` worked for Prem) or if on Windows, restart your terminal.

Now let's create a virtual environment. Virtual environments are a simple way to isolate all the dependencies for a particular project, making it easy to work on multiple projects at once without them interfering with each other (e.g. conflicting versions of libraries between projects). To make sure your environment matches the testing environment that we use for grading exactly, it's best to make a new environment for each assignment in this course. For consistency, please use Python 3.7. Here's the command:

``conda create -n hw0-setup python=3.7``

`hw0-setup` is the name for the environment and it can be replaced for each assignment (e.g. for the next homework, this would be hw1-audio-signals or something similar). The name can be whatever you want, just make sure to remember it!

Once the environment is created you should activate it with:

``conda activate hw0-setup``

Your shell might look something like `(hw0-setup) username@machine`. The environment name is in parenthesis, indicating that it is active. Now let's install the requirements for this assignment. Do this by navigating to the top root of this folder on the terminal and running, with an activated conda environment:

``pip install -r requirements.txt``

Great! Your environment is all set up. Do this for every assignment in the course.

## Github development cycle

All assignments are submitted via Github in this class. Once you've accepted this assignment via the Github classroom link, it made a repository of the form `https://github.com/machine-listening-spring-2021/hw0-setup-\[your github username\]`. In the first part of this README, you cloned the repository to your local machine to develop on.

To make changes, simply open or create some file in your local version. If you created a file, you have to do:

``git add [new_file_name]`` 

to make `git` track the file. If you edited an already tracked file, you don't have to add it. Then:

``git commit -am [commit_message]``

will commit the change. `commit_message` is something that describes the type of change you made. Good commit messages are descriptive, easy to understand, and correspond well with the actual changes made. Finally:

```
git pull origin master
git push origin master
```

will pull the remote code and then push the commit to the repository on Github. 

**The code that is on the Github server (not the version on your local machine) is the code we will run our tests on. If you don't push a commit, we won't see it and we won't grade it.**

## Running the test cases

Most (if not all) assignments will come with a *testing suite* that is used for autograding. This compares every function you implement with the output of a solution made by the course instructors. The test cases are transparent in input/output and can be inspected in the `tests` directory in the assignment's repository. If you pass all the test cases, you will get 100% on the autograded portion of the assignment. If you pass 9/10 test cases, you will get 90% on the autograded portion. 8/10 = 80%. And so on. **However, for some assignments, there will be additional private test cases that are only known by the instructors.**  This is to avoid you overfitting to the distributed test cases. The distributed test cases are mostly to guide and help you with development.

The test cases can be run with:

``python -m pytest -s``

at the root directory of the assignment repository. This gives output that looks something like:

```
========== test session starts ===========
platform darwin -- Python 3.7.2, pytest-4.1.0, py-1.7.0, pluggy-0.8.1
rootdir: /Users/mcartwright/git/machine-listening-spring-2021/hw0-setup-mcartwright, inifile:
collected 1 item

tests/test_assignment.py F         [100%]

================ FAILURES ================
____________ test_sum_numbers ____________

    def test_sum_numbers():
        from code import sum_numbers
>       assert sum_numbers(3, 5) == 8
E       assert None == 8
E        +  where None = <function sum_numbers at 0x10f62de18>(3, 5)

tests/test_assignment.py:3: AssertionError
======== 1 failed in 0.04 seconds ========
```

Parsing this output we see we have failed one test: `test_sum_numbers`. Let's try to make this test pass by implementing the related function. 

**Do that now by editing "code.py".**

Then re-run the tests to see if they passed! If they did, you'll see something like this: 

```
========== test session starts ===========
platform darwin -- Python 3.7.2, pytest-4.1.0, py-1.7.0, pluggy-0.8.1
rootdir: /Users/mcartwright/git/machine-listening-spring-2021/hw0-setup-mcartwright, inifile:
collected 1 item

tests/test_assignment.py .         [100%]

======== 1 passed in 0.03 seconds ========
```

## Saving output files

For some assignments, we may ask you to run test cases or write functions that may generate output, e.g plots, audio files, data files, etc., to the `output` directory. When an assignment asks for this, add, commit, and push these files as described in the *Github development cycle* section above. One of the test cases in this assignment generates a plot to `output`. Please make sure to add, commit, and push this file when you submit this assignment.

## Running Jupyter notebooks

Sometimes it is easier to illustrate a point using a Jupyter notebook, thus we may at times use these in assignments. For this assignment, launch JupyterLab by running the following command at the root of the assignment directory with your conda environment active:

``jupyter lab``

Once it launches in your browser, navigate to `notebooks/example_notebook.ipynb`. Read through and evaluate each cell in the notebook. To evaluate a single cell, press **Shift + Enter**. Once you have evaluated all of the cells in notebook and inspected their outputs, save the notebook. Note: because the notebook's output is dependent on order of execution, it is good practice to restart the kernel and re-run the notebook before the final save and submission. To do so, run “Kernel > Restart & Run All" from the notebook. You don’t have to do this every time while you’re debugging - just for the final submission. Commit and push the changes to the notebook with `git`. 

## How to submit your work

CODE / NOTEBOOKS / OUTPUT FILES: The code, notebooks, and output files that are on the Github server (not the version on your local machine) as of the deadline is what will be evaluated and tested. If you don't push a commit, we won't see it and we won't grade it.

FREE RESPONSE: For your free response answers, **you must submit a .pdf file on Canvas.**  We will grade the last thing you put on Canvas by the deadline. You don't get credit for having the free response in your code repository on github. 

## Questions? Problems? Issues?

Please make sure first that something in this document doesn't already address your issue! If you still have problems, simply open an issue on the starter code repository for this assignment [here](https://github.com/machine-listening-spring-2021/hw0-setup/issues). Someone from the teaching staff will get back to you through there!

That's all! The workflow for every assignment in this class will work something like this.

## Acknowledgements

This HW0 and assignment structure was adapted from https://github.com/NUCS349/hw0-setup
