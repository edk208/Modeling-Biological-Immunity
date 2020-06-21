# Modeling-Biological-Immunity
## Installation
 - Install OpenPV from github.  I used the devel branch
 - Put the following folder in the tutorials folder (along with LCACifar)
## Update Paths
 - update the scratchPath in input/Retina_CIFAR10.lua
 - update the img128.txt file locations, 5 samples images are already provided.  Notice that these images have been resized to 128x128 for this application.
 - update the gan_pvp_custom.lua file locations of the img128.txt, the number of images in the for loop (currently set to 5) and the mv of the ON and OFF ganglion pvp files...
## Run Retina Code
 - run lua gan_pvp_custom.lua   This will create the ON and OFF spike trains for the images and move them to the Ganglion_pvp_file folder.  
## Visualize Retina output
 - copy the on and off pvp files, as well as the original image to the image directory
 - In matlab, run the showspikes code.  This will visualize the spikes, as well as create a PNG movie sequence
 - See the spikes summed together using the sumofspikes.m code.
