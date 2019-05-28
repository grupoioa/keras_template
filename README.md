Keras Template
----------------

The objective of this template is to provide an initial code when
developing a new Deep Learning project and a suggestion in how to separate
the modules, configuration, files etc. The current template contains
the following files:

<img src="https://github.com/grupoioa/keras_template/blob/master/Tree.png?raw=true" width="229" height="389" />


This project assumes 4 main steps in training a model **preprocess,
training, classification, and visualization**. For every project you
will normally have to modify the way your data is being read and
visualized, and the DL models you want to use. But you should be able
to re-use lot of code. The configuration for these 4 steps is made
at `MainConfig.py`. The objective of separating the configuration in an
external file is to ease the collaboration.

The first file that need to be modified is `inout\readData`, this
should be able to read your own data. 

Then you need to modify `1_preproc` in order to perform any preprocess
necessary for your data. Here you can reuse some visualization functions
as well and some common ways to normalize data from the *preproc*  folder. 


