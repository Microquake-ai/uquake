# How to contribute (wip)

Currently this document is based on CONTRIBUTING.md of the ObsPy project. This
document certainly contains stuff unrelated to the microquake project. The
document will be fixed in the future.

This document aims to give an overview of how to contribute to microquake. It tries to answer commonly asked questions, and provide some insight into how the community process works in practice.

## Getting Started

 * Make sure you have a BitBucket account
 * [Download](http://git-scm.com/downloads) and install git
 * Read the [git documentation](http://git-scm.com/book/en/Git-Basics)
 

## Git workflow

We love pull requests! Here's a quick guide:

 1. Fork the repo.
 2. Make a new branch. For feature additions/changes base your new branch at "master", for pure bugfixes base your new branch at "releases" 
 3. Run the tests. We only take pull requests with passing tests.
 4. Add a test for your change. Only refactoring and documentation changes require no new tests. If you are adding functionality or fixing a bug, we need a test!
 5. Make the test pass.
 6. Push to your fork and submit a pull request.
    - for feature branches set base branch to "obspy:master"
    - for bugfix branches set base branch to "obspy:releases"
 7. Wait for our review. We may suggest some changes or improvements or alternatives.

## Additional Resources

 * [Style Guide](http://docs.obspy.org/coding_style.html)
 * [Docs or it doesn't exist!](http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
 * Performance Tips:
    * [Python](http://wiki.python.org/moin/PythonSpeed/PerformanceTips)
    * [NumPy and ctypes](http://www.scipy.org/Cookbook/Ctypes)
    * [SciPy](http://wiki.scipy.org/PerformancePython)
    * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)
 * [Interesting reading on git, github](https://github.com/obspy/obspy/wiki/Interesting-Reading-on-Git%2C-GitHub)
