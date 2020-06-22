--package.path = package.path .. ";" .. os.getenv("HOME") .. "/openpv/parameterWrapper/?.lua"; local pv = require "PVModule";
package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";
--local subnets = require "PVSubnets";


local inputWidth                = 256;
local inputHeight               = 256;
local inputFeatures             = 1;		             
local numClasses                = 10;
local nbatch                    = 1;      --Batch size
local displayMultiple           = 1;
local displayPeriod             = 128; --200;
local numEpochs                 = 1;      --Number of times to run through dataset
local numImages                 = 1;--50000;  --Total number of training images in dataset
local stopTime                  = math.ceil(numImages / nbatch) * displayPeriod * displayMultiple * numEpochs;
local writeStep                 = 1;
local initialWriteTime          = writeStep;
local start_frame_index         = 1;


local scratchPath               = "/home/ek826/Documents/OpenPV/tutorials/Retina";
local runName                   = "Retina_CIFAR10"; --"Retina_square";
local imageInputPath            = "../img128.txt";
local outputPath                = scratchPath .. "/" .. "output" .. "/" .. runName
local initializeFromCheckpointDir = nil;
local checkpointPeriod          = displayPeriod;

local errorWrite = -1 -- displayPeriod*displayMultiple;

-- Retina model parameters
local feedforward_strength       = 1.0;
local feedback_strength          = 1.0;
local gap_junction_strength      = 0.0;
local HorizontalToHorizontalFlag = false; --true;
local VRest                      = -70.0;

-- Base table 
pvParams = {
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = checkpointPeriod;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = runName .. ".params";
      randomSeed                          = nil;
      nx                                  = inputWidth;
      ny                                  = inputHeight;
      nbatch                              = nbatch;
      initializeFromCheckpointDir         = nil; --initPath .. "/Checkpoints/Checkpoint" .. stopTime; -- 
      --defaultInitializeFromCheckpointFlag = false;
      checkpointWrite                     = false;
      lastCheckpointDir                   = outputPath .. "/Checkpoints"; 
      checkpointWriteDir                  = nil; --outputPath .. "/Checkpoints"; 
      checkpointWriteTriggerMode          = nil; --"step";
      checkpointWriteStepInterval         = nil; --checkpointPeriod; --How often to checkpoint
      deleteOlderCheckpoints              = nil; --true;
      numCheckpointsKept                  = nil; --2;
      suppressNonplasticCheckpoints       = false;
      writeTimescales                     = true;
      errorOnNotANumber                   = false;
   } 
}



-- Image Layers ----------------------------------------------------
pv.addGroup(pvParams,
	    "Image", 
	    {
	       groupType                           = "ImageLayer";
	       nxScale                             = 1;
	       nyScale                             = 1;
	       nf                                  = 1;
	       phase                               = 1;
	       mirrorBCflag                        = true;
	       writeStep                           = displayPeriod;
	       initialWriteTime                    = displayPeriod;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       inputPath                           = imageInputPath;
	       start_frame_index                   = start_frame_index;
	       offsetAnchor                        = "tl";
	       offsetX                             = 0;
	       offsetY                             = 0;
	       maxShiftX                           = 0;
	       maxShiftY                           = 0;
	       writeImages                         = 0;
	       inverseFlag                         = false;
	       normalizeLuminanceFlag              = false; 
	       normalizeStdDev                     = false;
	       useInputBCflag                      = false;
	       padValue                            = 0;
	       autoResizeFlag                      = true; 
	       aspectRatioAdjustment               = "pad";
	       interpolationMethod                 = "bicubic";
	       displayPeriod                       = displayPeriod*displayMultiple;
	       batchMethod                         = "byFile"; --"random";
	       randomSeed                          = nil;
	       writeFrameToTimestamp               = true;
	       resetToStartOnLoop                  = false;
	    }
)
if CIFAR10_flag then
   pvParams["Image"].autoResizeFlag                = true; 
end

local GroundTruthFlag = false;
if GroundTruthFlag then
   pv.addGroup(pvParams,
	       "GroundTruth", 
	       {
		  groupType                           = "FilenameParsingGroundTruthLayer";
		  nxScale                             = 1/inputWidth;
		  nyScale                             = 1/inputHeight;
		  nf                                  = numClasses;
		  phase                               = 0;
		  --triggerLayerName                    = "Image";
		  --triggerLayerBehavior                = "updateOnlyOnTrigger";
		  --triggerOffset                       = 0; --triggerOffsetGroundTruth;
		  inputLayerName                      = "Image";
		  gtClassTrueValue                    = 1.0;
		  gtClassFalseValue                   = 0.0;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  --initializeFromCheckpointFlag        = false;
		  writeStep                           = writeStepMLP;
		  initialWriteTime                    = initialWriteTimeMLP;
		  InitVType                           = "ZeroV";
		  sparseLayer                         = true;
		  updateGpu                           = false;
		  dataType                            = NULL;
		  VThresh                             = -infinity;
		  AMin                                = -infinity;
		  AMax                                = infinity;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
		  classList                           = scratchPath .. "/" .. "CIFAR10" .. "/" .. "classes.txt";
	       }
   )
end


BiasConeFlag = false;
if BiasConeFlag then
   pv.addGroup(pvParams,
	       "BiasCone",
	       {
		  groupType                           = "ConstantLayer";
		  nxScale                             = 1;
		  nyScale                             = 1;
		  nf                                  = 1;
		  phase                               = 0;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = false;
		  InitVType                           = "ConstantV";
		  valueV                              = 1.0;
		  writeStep                           = -1;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  clearGSynInterval                   = 0;
	       }
   )

   pv.addGroup(pvParams,
	       "BiasHorizontal",
	       pvParams["BiasCone"],
	       {
		  nxScale                             = 0.25;
		  nyScale                             = 0.25;
	       }
)

end

---------------------
--Cone
---------------------
local Cone_VRest = VRest;
pv.addGroup(pvParams,
	    "Cone",
	    {
	       groupType                           = "LIF";
	       nxScale                             = 1;
	       nyScale                             = 1;
	       nf                                  = 1;
	       phase                               = 2;
	       mirrorBCflag                        = true;
	       valueBC                             = 0;
	       --initializeFromCheckpointFlag        = false;
	       InitVType                           = "ConstantV";
	       valueV                              = Cone_VRest;
	       method                              = "arma";
	       triggerLayerName                    = NULL;
	       writeStep                           = errorWrite;
	       initialWriteTime                    = errorWrite;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       Vrest                               = -54; ---58.0; --Cone_VRest; 
	       Vexc                                = 0.0;     -- reversal potential 
	       Vinh                                = -60.0; --Cone_VRest; --75.0;   -- chloride channel
	       VinhB                               = -90.0;   -- potassium reversal
	       tau                                 = 10.0;    -- intrinsic leak membrane time constant (max)
	       tauE                                = 2.0;     -- how long glutamine stays bound
	       tauI                                = 5.0;     -- how long GABA stays bound
	       tauIB                               = 10.0;    -- inhibitory potassium channel 
	       VthRest                             = 100000.0;-- firing threshold disabled
	       tauVth                              = 10.0;    -- relative refractory period
	       deltaVth                            = 0.0;     -- jump of threshold when firing
	       noiseAmpE                           = 0.0;     -- 1 means conductance is equal to leak conductance
	       noiseAmpI                           = 0.0;     -- 0 == no noise
	       noiseAmpIB                          = 0.0;
	       noiseFreqE                          = 0.0;
	       noiseFreqI                          = 0.0;
	       noiseFreqIB                         = 0.0;
	    }
)

pv.addGroup(pvParams,
	    "ConeSigmoidON",
	    {
	       groupType                           = "SigmoidLayer";
	       nxScale                             = 1;
	       nyScale                             = 1;
	       nf                                  = 1;
	       phase                               = 3;
	       mirrorBCflag                        = true;
	       valueBC                             = 0;
	       triggerLayerName                    = NULL;
	       writeStep                           = errorWrite;
	       initialWriteTime                    = errorWrite;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       originalLayerName                   = "Cone";
	       Vrest                               = -52.0; --VRest; 
	       VthRest                             = -40.0;                     
	       SigmoidFlag                         = false;
	       SigmoidAlpha                        = 1/4;
	       InverseFlag                         = false;
	    }
)


pv.addGroup(pvParams,
	    "ConeSigmoidOFF",
	    pvParams["ConeSigmoidON"],
	    {
	       InverseFlag                         = 1.0;
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "ConeCloneV",
	       {
		  groupType                           = "CloneVLayer";
		  nxScale                             = 1.0;
		  nyScale                             = 1.0;
		  phase                               = 3;
		  mirrorBCflag                        = true;
		  valueBC                             = 0;
		  triggerLayerName                    = NULL;
		  writeStep                           = errorWrite;
		  initialWriteTime                    = errorWrite;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  originalLayerName                   = "Cone";
	       }
   )
end



---------------------
-- Horizontal Cells 
---------------------
local Horizontal_VRest = VRest;
pv.addGroup(pvParams,
	    "Horizontal",
	    pvParams["Cone"],
	    {
	       groupType                           = "LIFGap";
	       nxScale                             = 0.25;
	       nyScale                             = 0.25;
	       nf                                  = 1;
	       phase                               = 4;
	       Vrest                               = -60; --Horizontal_VRest; 
	       valueV                              = -45; --Horizontal_VRest;
	       mirrorBCflag                        = true;
	    }
)

pv.addGroup(pvParams,
	    "HorizontalSigmoid",
	    pvParams["ConeSigmoidON"],
 	    {
	       nxScale                             = 0.25;
	       nyScale                             = 0.25;
	       phase                               = 8;
	       Vrest                               = -60; --Horizontal_VRest; 
	       VthRest                             = -20;                     
	       originalLayerName                   = "Horizontal";
	    }
)

pv.addGroup(pvParams,
	    "HorizontalGap",
	    {
	       groupType                           = "GapLayer";
	       nxScale                             = 0.25;
	       nyScale                             = 0.25;
	       nf                                  = 1;
	       phase                               = 5;
	       mirrorBCflag                        = true;
	       valueBC                             = 0;
	       triggerLayerName                    = NULL;
	       writeStep                           = errorWrite;
	       initialWriteTime                    = errorWrite;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       originalLayerName                   = "Horizontal";
	       ampSpikelet                         = 150.0;
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "HorizontalCloneV",
	       pvParams["ConeCloneV"],
	       {
		  nxScale                             = 0.25;
		  nyScale                             = 0.25;
		  phase                               = 5;
		  originalLayerName                   = "Horizontal";
	       }
   )
end

---------------------
-- bipolar cells
---------------------
pv.addGroup(pvParams,
	    "BipolarON",
	    pvParams["Cone"],
	    {
	       groupType                           = "LIF";
	       phase                               = 7;
	       Vrest                               = VRest;
	       valueV                              = -45; --VRest;
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidON",
	    pvParams["ConeSigmoidON"],
 	    {
	       phase                               = 8;
	       Vrest                               = -35; ---37.5; -- -40; --VRest; 
	       VthRest                             = -15; -- -20;                     
	       originalLayerName                   = "BipolarON";
	    }
)

pv.addGroup(pvParams,
	    "BipolarOFF",
	    pvParams["BipolarON"],
	    {
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidOFF",
	    pvParams["BipolarSigmoidON"],
 	    {
	       originalLayerName                   = "BipolarOFF";
	    }
)


if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "BipolarONCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 8;
		  originalLayerName                   = "BipolarON";
	       }
   )
   pv.addGroup(pvParams,
	       "BipolarOFFCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 8;
		  originalLayerName                   = "BipolarOFF";
	       }
   )
end


------------------------------------------
-- Small Field Bistratified Amacrine cells
-----------------------------------------
pv.addGroup(pvParams,
	    "SmallBiAmacrineON",
	    pvParams["Horizontal"],
	    {
	       nxScale                             = 1;
	       nyScale                             = 1;
	       phase                               = 9;
	       tau                                 = 20.0;    
	       Vrest                               = VRest;
	       valueV                              = VRest;
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineSigmoidON",
	    pvParams["BipolarSigmoidON"],
 	    {
	       phase                               = 10;
	       Vrest                               = -70; --VRest; 
	       VthRest                             = -50;                     
	       originalLayerName                   = "SmallBiAmacrineON";
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineGapON",
	    pvParams["HorizontalGap"],
 	    {
	       phase                               = 10;
	       nxScale                             = 1;
	       nyScale                             = 1;
	       originalLayerName                   = "SmallBiAmacrineON";
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineOFF",
	    pvParams["SmallBiAmacrineON"],
	    {
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineSigmoidOFF",
	    pvParams["SmallBiAmacrineSigmoidON"],
 	    {
	       originalLayerName                   = "SmallBiAmacrineOFF";
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineGapOFF",
	    pvParams["SmallBiAmacrineGapON"],
 	    {
	       originalLayerName                   = "SmallBiAmacrineOFF";
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "SmallBiAmacrineONCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 8;
		  originalLayerName                   = "SmallBiAmacrineON";
	       }
   )
   pv.addGroup(pvParams,
	       "SmallBiAmacrineOFFCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 8;
		  originalLayerName                   = "SmallBiAmacrineOFF";
	       }
   )
end


----------------------------
-- Wide Field Amacrine cells
----------------------------
pv.addGroup(pvParams,
	    "WideFieldAmacrineON",
	    pvParams["BipolarON"],
	    {
	       nxScale                             = 0.25;
	       nyScale                             = 0.25;
	       phase                               = 9;
	       Vrest                               = Vrest;
	       valueV                              = Vrest;
	       tau                                 = 20.0;    
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidON",
	    pvParams["BipolarSigmoidON"],
 	    {
	       nxScale                             = 0.25;
	       nyScale                             = 0.25;
	       phase                               = 10;
	       Vrest                               = -60; --Vrest; 
	       VthRest                             = -40; -- -20;                     
	       originalLayerName                   = "WideFieldAmacrineON";
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineOFF",
	    pvParams["WideFieldAmacrineON"],
	    {
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidOFF",
	    pvParams["WideFieldAmacrineSigmoidON"],
 	    {
	       originalLayerName                   = "WideFieldAmacrineOFF";
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "WideFieldAmacrineONCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 8;
		  nxScale                             = 0.25;
		  nyScale                             = 0.25;
		  originalLayerName                   = "WideFieldAmacrineON";
	       }
   )
   pv.addGroup(pvParams,
	       "WideFieldAmacrineOFFCloneV",
	       pvParams["WideFieldAmacrineONCloneV"],
	       {
		  originalLayerName                   = "WideFieldAmacrineOFF";
	       }
   )
end



------------------------
-- Poly-Axonal Amacrine
------------------------
pv.addGroup(pvParams,
	    "PolyAxonalAmacrineON",
	    pvParams["Horizontal"],
	    {
	       phase                               = 9;
	       Vrest                               = Vrest;
	       valueV                              = Vrest;
	       tau                                 = 10.0;    
	       tauE                                = 1.0;     
	       tauI                                = 3.0;    
	       tauIB                               = 10.0;    
	       VthRest                             = -55.;    
	       tauVth                              = 10.0;    
	       deltaVth                            = 5.0;     
	       sparseLayer                         = true;
	       writeStep                           = 1.0;
	       initialWriteTime                    = 1.0;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapON",
	    pvParams["HorizontalGap"],
 	    {
	       phase                               = 10;
	       originalLayerName                   = "PolyAxonalAmacrineON";
	       ampSpikelet                         = 150;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineOFF",
	    pvParams["PolyAxonalAmacrineON"],
	    {
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapOFF",
	    pvParams["PolyAxonalAmacrineGapON"],
 	    {
	       originalLayerName                   = "PolyAxonalAmacrineOFF";
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "PolyAxonalAmacrineONCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 10;
		  nxScale                             = 0.25;
		  nyScale                             = 0.25;
		  originalLayerName                   = "PolyAxonalAmacrineON";
	       }
   )
   pv.addGroup(pvParams,
	       "PolyAxonalAmacrineOFFCloneV",
	       pvParams["PolyAxonalAmacrineONCloneV"],
	       {
		  originalLayerName                   = "PolyAxonalAmacrineOFF";
	       }
   )
end


---------------------
-- Ganglion 
---------------------
pv.addGroup(pvParams,
	    "GanglionON",
	    pvParams["PolyAxonalAmacrineON"],
	    {
	       phase                               = 11;
	       nxScale                             = 0.5;
	       nyScale                             = 0.5;
	       tau                                 = 10.0;    
	       tauE                                = 1.0;     
	       tauI                                = 3.0;    
	       VthRest                             = -55.0;
	    }
)

pv.addGroup(pvParams,
	    "GanglionGapON",
	    pvParams["PolyAxonalAmacrineGapON"],
 	    {
	       phase                               = 12;
	       nxScale                             = 0.5;
	       nyScale                             = 0.5;
	       originalLayerName                   = "GanglionON";
	    }
)

pv.addGroup(pvParams,
	    "GanglionOFF",
	    pvParams["GanglionON"],
	    {
	    }
)

pv.addGroup(pvParams,
	    "GanglionGapOFF",
	    pvParams["GanglionGapON"],
	    {
	       originalLayerName                   = "GanglionOFF";
	    }
)

if not (errorWrite == -1) then
   pv.addGroup(pvParams,
	       "GanglionONCloneV",
	       pvParams["ConeCloneV"],
	       {
		  phase                               = 12;
		  nxScale                             = 0.5;
		  nyScale                             = 0.5;
		  originalLayerName                   = "GanglionON";
	       }
   )
   pv.addGroup(pvParams,
	       "GanglionOFFCloneV",
	       pvParams["GanglionONCloneV"],
	       {
		  originalLayerName                   = "GanglionOFF";
	       }
   )
end



--------------------------------------------------------------------
--Retinal Connections
--------------------------------------------------------------------


--------------------------------------------------------------------
-- Image Connections
--------------------------------------------------------------------
pv.addGroup(pvParams,
	    "Image" .. "To" .. "Cone",
	    {
	       groupType                           = "HyPerConn";
	       preLayerName                        = "Image";
	       postLayerName                       = "Cone";
	       channelCode                         = 0;
	       nxp                                 = 1;
	       nyp                                 = 1;
	       nfp                                 = 1;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       sharedWeights                       = true;
	       plasticityFlag                      = false;
	       triggerLayerName                    = nil;
	       updateGSynFromPostPerspective       = true;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       initialWriteTime                    = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       shrinkPatches                       = false;
	       normalizeMethod                     = "normalizeSum";
	       normalizeArborsIndividually         = false;
	       normalizeOnInitialize               = true;
	       normalizeOnWeightUpdate             = true;
	       rMinX                               = 0;
	       rMinY                               = 0;
	       nonnegativeConstraintFlag           = false;
	       normalize_cutoff                    = 0;
	       normalizeFromPostPerspective        = true;
	       minL2NormTolerated                  = 0;
	       weightInitType                      = "Gauss2DWeight";
	       aspect                              = 1.0;
	       sigma                               = 1.0; -- strength falls off by 1/e, presynaptic units   
	       rMax                                = inputWidth;   
	       rMin                                = 0.0;
	       strength                            = feedforward_strength;
	    }
)

if BiasConeFlag then
   pv.addGroup(pvParams,
	       "BiasCone" .. "To" .. "Cone",
	       {
		  groupType                           = "RescaleConn";
		  preLayerName                        = "BiasCone";
		  postLayerName                       = "Cone";
		  channelCode                         = 1;
		  scale                               = 1.0;
	       }
   )
end

--------------------------------------------------------------------
-- Cone connections
--------------------------------------------------------------------

pv.addGroup(pvParams,
	    "ConeSigmoidON" .. "To" .. "BipolarON",
	    pvParams["Image" .. "To" .. "Cone"],
	    {
	       groupType                           = "HyPerConn";
	       preLayerName                        = "ConeSigmoidON";
	       postLayerName                       = "BipolarON";
	       nxp                                 = 3;
	       nyp                                 = 3;
	       sigma                               = 1.0; -- strength falls off by 1/e, presynaptic units   
	       strength                            = feedforward_strength*2.0;
	    }
)


pv.addGroup(pvParams,
	    "ConeSigmoidOFF" .. "To" .. "BipolarOFF",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "ConeSigmoidOFF";
	       postLayerName                       = "BipolarOFF";
	    }
)

pv.addGroup(pvParams,
	    "ConeSigmoidON" .. "To" .. "Horizontal",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "ConeSigmoidON";
	       postLayerName                       = "Horizontal";
	       sigma                               = 4.0; 
	       strength                            = feedforward_strength/2.0;
	    }
)


---------------------------
-- Horizontal Connections
---------------------------
if BiasConeFlag then
   pv.addGroup(pvParams,
	       "BiasHorizontal" .. "To" .. "Horizontal",
	       {
		  groupType                           = "RescaleConn";
		  preLayerName                        = "BiasHorizontal";
		  postLayerName                       = "Horizontal";
		  channelCode                         = 1;
		  scale                               = 1.0;
	       }
   )
end


if HorizontalToHorizontalFlag then
   pv.addGroup(pvParams,
	       "Horizontal" .. "To" .. "Horizontal",
	       pvParams["Image" .. "To" .. "Cone"],
	       {
		  --groupType                           = "GapConn";
		  groupType                           = "HyPerConn";
		  preLayerName                        = "HorizontalGap";
		  postLayerName                       = "Horizontal";
		  nxp                                 = 3.0; 
		  nyp                                 = 3.0; 
		  sigma                               = 1.0; 
		  rMin                                = 0.1;
		  strength                            = gap_junction_strength;
                  channelCode                         = 3;
	       }
   )
   --pvParams[ "Horizontal" .. "To" .. "Horizontal"].channelCode = nil;
else
   pvParams["HorizontalGap"]                       = nil;
   pvParams["Horizontal"].groupType                = "LIF";
end

pv.addGroup(pvParams,
	    "HorizontalSigmoid" .. "To" .. "Cone",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "HorizontalSigmoid";
	       postLayerName                       = "Cone";
	       channelCode                         = 1;
	       nxp                                 = 12.0; --4.0; 
	       nyp                                 = 12.0; --4.0; 
	       weightInitType                      = "UniformWeight";
	       strength                            = feedback_strength;
	    }
)
pvParams["HorizontalSigmoid" .. "To" .. "Cone"].weightInit  = 1.0/16;
pvParams["HorizontalSigmoid" .. "To" .. "Cone"].aspect      = nil;
pvParams["HorizontalSigmoid" .. "To" .. "Cone"].sigma       = nil;
pvParams["HorizontalSigmoid" .. "To" .. "Cone"].rMax        = nil;
pvParams["HorizontalSigmoid" .. "To" .. "Cone"].rMin        = nil;

----------------------
-- Bipolar Connections
----------------------
if BiasConeFlag then
   pv.addGroup(pvParams,
	       "BiasCone" .. "To" .. "BipolarON",
	       {
		  groupType                           = "RescaleConn";
		  preLayerName                        = "BiasCone";
		  postLayerName                       = "BipolarON";
		  channelCode                         = 1;
		  scale                               = 1.0;
	       }
   )

   pv.addGroup(pvParams,
	       "BiasCone" .. "To" .. "BipolarOFF",
	       {
		  groupType                           = "RescaleConn";
		  preLayerName                        = "BiasCone";
		  postLayerName                       = "BipolarOFF";
		  channelCode                         = 1.0;
	       }
   )
end

pv.addGroup(pvParams,
	    "BipolarSigmoidON" .. "To" .. "SmallBiAmacrineON",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "BipolarSigmoidON";
	       postLayerName                       = "SmallBiAmacrineON";
	       nxp                                 = 3.0;
	       nyp                                 = 3.0;
	       sigma                               = 1.0; 
	       strength                            = feedforward_strength;
	       pvpatchAccumulateType               = "stochastic";
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidOFF" .. "To" .. "SmallBiAmacrineOFF",
	    pvParams["BipolarSigmoidON" .. "To" .. "SmallBiAmacrineON"],
	    {
	       preLayerName                        = "BipolarSigmoidOFF";
	       postLayerName                       = "SmallBiAmacrineOFF";
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidON" .. "To" .. "WideFieldAmacrineON",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "BipolarSigmoidON";
	       postLayerName                       = "WideFieldAmacrineON";
	       nxp                                 = 3.0;
	       nyp                                 = 3.0;
	       sigma                               = 4.0; 
	       strength                            = feedforward_strength;
	       pvpatchAccumulateType               = "stochastic";
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidOFF" .. "To" .. "WideFieldAmacrineOFF",
	    pvParams["BipolarSigmoidON" .. "To" .. "WideFieldAmacrineON"],
	    {
	       preLayerName                        = "BipolarSigmoidOFF";
	       postLayerName                       = "WideFieldAmacrineOFF";
	    }
)

local BipolarToPolyAxonalFlag = false; --true;
if BipolarToPolyAxonalFlag then
   pv.addGroup(pvParams,
	       "BipolarSigmoidON" .. "To" .. "PolyAxonalAmacrineON",
	       pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	       {
		  preLayerName                        = "BipolarSigmoidON";
		  postLayerName                       = "PolyAxonalAmacrineON";
		  nxp                                 = 3.0;
		  nyp                                 = 3.0;
		  sigma                               = 4.0; 
		  strength                            = feedforward_strength/4;
		  pvpatchAccumulateType               = "stochastic";
	       }
   )
   
   pv.addGroup(pvParams,
	       "BipolarSigmoidOFF" .. "To" .. "PolyAxonalAmacrineOFF",
	       pvParams["BipolarSigmoidON" .. "To" .. "PolyAxonalAmacrineON"],
	       {
		  preLayerName                        = "BipolarSigmoidOFF";
		  postLayerName                       = "PolyAxonalAmacrineOFF";
	       }
   )
end

pv.addGroup(pvParams,
	    "BipolarSigmoidON" .. "To" .. "GanglionON",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "BipolarSigmoidON";
	       postLayerName                       = "GanglionON";
	       nxp                                 = 3.0;
	       nyp                                 = 3.0;
	       sigma                               = 2.0; 
	       strength                            = 5.0*feedforward_strength;
	       pvpatchAccumulateType               = "stochastic";
	    }
)

pv.addGroup(pvParams,
	    "BipolarSigmoidOFF" .. "To" .. "GanglionOFF",
	    pvParams["BipolarSigmoidON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "BipolarSigmoidOFF";
	       postLayerName                       = "GanglionOFF";
	    }
)


-------------------------------------------
-- Small Bistratified Amacrine Connections
-------------------------------------------

-- mimic bistratified by connection ON and OFF with gap junction
pv.addGroup(pvParams,
	    "SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF",
	    pvParams["Image" .. "To" .. "Cone"],
	    {
	       --groupType                           = "GapConn";
	       groupType                           = "HyPerConn";
	       preLayerName                        = "SmallBiAmacrineGapON";
	       postLayerName                       = "SmallBiAmacrineOFF";
	       nxp                                 = 1.0;
	       nyp                                 = 1.0;
	       sigma                               = 1.0; 
	       strength                            = 0.25;
	       rMin                                = 0.0;
               channelCode                         = 3;
	    }
)
--pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].channelCode          = nil;
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].weightInitType       = "UniformWeight";
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].weightInit           = 1.0/16;
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].aspect               = nil;
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].sigma                = nil;
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].rMax                 = nil;
pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"].rMin                 = nil;

pv.addGroup(pvParams,
	    "SmallBiAmacrineGapOFF" .. "To" .. "SmallBiAmacrineON",
	    pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"],
	    {
	       preLayerName                        = "SmallBiAmacrineGapOFF";
	       postLayerName                       = "SmallBiAmacrineON";
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineSigmoidON" .. "To" .. "GanglionON",
	    pvParams["ConeSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "SmallBiAmacrineSigmoidON";
	       postLayerName                       = "GanglionON";
	       channelCode                         = 1;
	       pvpatchAccumulateType               = "stochastic";
	       nxp                                 = 3.0;
	       nyp                                 = 3.0;
	       sigma                               = 2.0; 
	       strength                            = feedforward_strength;
	    }
)

pv.addGroup(pvParams,
	    "SmallBiAmacrineSigmoidOFF" .. "To" .. "GanglionOFF",
	    pvParams["SmallBiAmacrineSigmoidON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "SmallBiAmacrineSigmoidOFF";
	       postLayerName                       = "GanglionOFF";
	    }
)


----------------------------------
-- Wide Field Amacrine Connections
----------------------------------
pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidON" .. "To" .. "BipolarON",
	    pvParams["HorizontalSigmoid" .. "To" .. "Cone"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidON";
	       postLayerName                       = "BipolarON";
	       pvpatchAccumulateType               = "stochastic";
	       strength                            = feedback_strength;
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidOFF" .. "To" .. "BipolarOFF",
	    pvParams["WideFieldAmacrineSigmoidON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidOFF";
	       postLayerName                       = "BipolarOFF";
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidON" .. "To" .. "SmallBiAmacrineON",
	    pvParams["HorizontalSigmoid" .. "To" .. "Cone"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidON";
	       postLayerName                       = "SmallBiAmacrineON";
	       pvpatchAccumulateType               = "stochastic";
	       strength                            = feedback_strength;
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidOFF" .. "To" .. "SmallBiAmacrineOFF",
	    pvParams["WideFieldAmacrineSigmoidON" .. "To" .. "SmallBiAmacrineON"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidOFF";
	       postLayerName                       = "SmallBiAmacrineOFF";
	    }
)


pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidON" .. "To" .. "GanglionON",
	    pvParams["HorizontalSigmoid" .. "To" .. "Cone"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidON";
	       postLayerName                       = "GanglionON";
	       nxp                                 = 6.0;
	       nyp                                 = 6.0;
	       strength                            = feedforward_strength;
	    }
)

pv.addGroup(pvParams,
	    "WideFieldAmacrineSigmoidOFF" .. "To" .. "GanglionOFF",
	    pvParams["WideFieldAmacrineSigmoidON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "WideFieldAmacrineSigmoidOFF";
	       postLayerName                       = "GanglionOFF";
	    }
)


-------------------------------------
-- Poly-Axonal Amacrine Connections
-------------------------------------
pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapON" .. "To" .. "PolyAxonalAmacrineON",
	    pvParams["SmallBiAmacrineGapON" .. "To" .. "SmallBiAmacrineOFF"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineGapON";
	       postLayerName                       = "PolyAxonalAmacrineON";
	       pvpatchAccumulateType               = "convolve";
	       nxp                                 = 3;
	       nyp                                 = 3;
	       delay                               = 0;
	       strength                            = gap_junction_strength;
	       rMinX                               = 0.5;
	       rMinY                               = 0.5;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapOFF" .. "To" .. "PolyAxonalAmacrineOFF",
	    pvParams["PolyAxonalAmacrineGapON" .. "To" .. "PolyAxonalAmacrineON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineGapOFF";
	       postLayerName                       = "PolyAxonalAmacrineOFF";
	    }
)


pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapON" .. "To" .. "GanglionON",
	    pvParams["PolyAxonalAmacrineGapON" .. "To" .. "PolyAxonalAmacrineON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineGapON";
	       postLayerName                       = "GanglionON";
	       pvpatchAccumulateType               = "convolve";
	       nxp                                 = 6;
	       nyp                                 = 6;
	       delay                               = 0;
	       rMinX                               = 0.0;
	       rMinY                               = 0.0;
	       strength                            = 0.25 * gap_junction_strength;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineGapOFF" .. "To" .. "GanglionOFF",
	    pvParams["PolyAxonalAmacrineGapON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineGapOFF";
	       postLayerName                       = "GanglionOFF";
	    }
)


pv.addGroup(pvParams,
	    "PolyAxonalAmacrineON" .. "To" .. "BipolarON",
	    pvParams["HorizontalSigmoid" .. "To" .. "Cone"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineON";
	       postLayerName                       = "BipolarON";
	       pvpatchAccumulateType               = "convolve";
	       nxp                                 = 20;
	       nyp                                 = 20;
	       delay                               = 2;
	       rMinX                               = 0.0;
	       rMinY                               = 0.0;
	       strength                            = feedback_strength/4;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineOFF" .. "To" .. "BipolarOFF",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineOFF";
	       postLayerName                       = "BipolarOFF";
	    }
)


pv.addGroup(pvParams,
	    "PolyAxonalAmacrineON" .. "To" .. "WideFieldAmacrineON",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineON";
	       postLayerName                       = "WideFieldAmacrineON";
	       nxp                                 = 5;
	       nyp                                 = 5;
	       rMinX                               = 0.0;
	       rMinY                               = 0.0;
	       strength                            = feedback_strength;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineOFF" .. "To" .. "WideFieldAmacrineOFF",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "WideFieldAmacrineON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineOFF";
	       postLayerName                       = "WideFieldAmacrineOFF";
	    }
)


pv.addGroup(pvParams,
	    "PolyAxonalAmacrineON" .. "To" .. "PolyAxonalAmacrineON",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineON";
	       postLayerName                       = "PolyAxonalAmacrineON";
	       nxp                                 = 5;
	       nyp                                 = 5;
	       rMinX                               = 0.5;
	       rMinY                               = 0.5;
	       strength                            = 2*2.0;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineOFF" .. "To" .. "PolyAxonalAmacrineOFF",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "PolyAxonalAmacrineON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineOFF";
	       postLayerName                       = "PolyAxonalAmacrineOFF";
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineON" .. "To" .. "GanglionON",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "BipolarON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineON";
	       postLayerName                       = "GanglionON";
	       nxp                                 = 10;
	       nyp                                 = 10;
	       rMinX                               = 0.0;
	       rMinY                               = 0.0;
	       strength                            = 2*20.0;
	    }
)

pv.addGroup(pvParams,
	    "PolyAxonalAmacrineOFF" .. "To" .. "GanglionOFF",
	    pvParams["PolyAxonalAmacrineON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "PolyAxonalAmacrineOFF";
	       postLayerName                       = "GanglionOFF";
	    }
)


--Ganglion Connections
pv.addGroup(pvParams,
	    "GanglionGapON" .. "To" .. "PolyAxonalAmacrineON",
	    pvParams["PolyAxonalAmacrineGapON" .. "To" .. "GanglionON"],
	    {
	       preLayerName                        = "GanglionGapON";
	       postLayerName                       = "PolyAxonalAmacrineON";
	       nxp                                 = 3;
	       nyp                                 = 3;
	       delay                               = 0;
	       pvpatchAccumulateType               = "convolve";
	       rMinX                               = 0.0;
	       rMinY                               = 0.0;
	       strength                            = gap_junction_strength;
	    }
)

pv.addGroup(pvParams,
	    "GanglionGapOFF" .. "To" .. "PolyAxonalAmacrineOFF",
	    pvParams["GanglionGapON" .. "To" .. "PolyAxonalAmacrineON"],
	    {
	       preLayerName                        = "GanglionGapOFF";
	       postLayerName                       = "PolyAxonalAmacrineOFF";
	    }
)








-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParams)
