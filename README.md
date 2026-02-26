# First : fill in the 'images' and 'annotations' repositories

* ***'images'*** repositery : images from SimpleShapes1 and SimpleShapes2
* ***'annotations'*** repository : annotations files for SimpleShapes1 and SimpleShapes2 (you may need to rename them)

# How to use the program

On the command line :

`py main.py`

Several options are available :

* `-c [MLP, RF]` : choose the classifier (either MLP for Multi-layer Perceptron, or RF for Random Forests) [default = MLP]
* `--db [S1, S2]` : choose the database (either S1 : SimpleShapes1, or S2 : SimpleShapes2) [default = S1]
* `-d [RLM, FORCE, DIST]` : choose the descriptor, you can select multiple descriptors at once (RLM for the original one, FORCE for the Force descriptor, and DIST for the newly implemented descriptor of Distance) [default = RLM and FORCE]
* `-f number` : choose the force parameter (for now, it will only work for f=0 and f=2)
* `-h` : show the help menu for the arguments

