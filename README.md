# Modeling-Biological-Immunity
## Installation
 - Install OpenPV from github. https://github.com/PetaVision/OpenPV. I used the devel branch
 - Put the following folder in the tutorials folder (along with LCACifar)
## Update Paths in Retina
 - update the scratchPath in Retina/input/Retina_CIFAR10.lua
 - update the Retina/img128.txt file locations, 5 samples images are already provided.  Notice that these images have been resized to 128x128 for this application.
 - update the Retina/gan_pvp_custom.lua file locations of the img128.txt, the number of images in the for loop (currently set to 5) and the mv of the ON and OFF ganglion pvp files...
## Run Retina Code
 - run lua Retina/gan_pvp_custom.lua   This will create the ON and OFF spike trains for the images and move them to the Ganglion_pvp_file folder.  
## Visualize Retina output
 - copy the on and off pvp files, as well as the original image to the image directory
 - In matlab, run the showspikes code.  This will visualize the spikes, as well as create a PNG movie sequence
 - See the spikes summed together using the sumofspikes.m code.
 - Retina/mlab/util directory is provided if the original OpenPV code causes problems
## V1 Primary Cortex Code
 - Update ImageNet/img128.txt for your current image path
 - run ImageNet/input/ImageNet.lua.  This will run a 3 layer deep sparse code on the input images and store them in the output.
 - use the OpenPV scripts (from the LCACifar tutorial) to analyze and visualize the output
## V1 Primary Cortex Code with excitatory top down feedback
 - Update MNist/input.txt for mnist images.  These were resized to a power of 2, (32x32)
 - Run a 2 layer deep sparse code MNistOne.lua
 - Once trained, locate the layer weights, I used these weights Checkpoint6000000One/V1ToInputError_W.pvp  
 - and Checkpoint6000000One/P1ToV1Error_W.pvp
 - Identify the mean activation of various digits using the MNist/script/extractTopdownVectorsMnist.m  You can use these to drive the "prior" of the network
 - Run MNist/input/MNistOneTest2.lua for excitatory top down feedback
 - Adjust the weights of feeback.  For example InputPvpTpoP1 scale is 0... meaning it does not influence P1... this can be adjusted.

 - Adjust V1P1VisionRecontoV1 feedback.  Currently set to 0.85
