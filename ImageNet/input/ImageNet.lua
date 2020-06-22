-- Path to OpenPV. Replace this with an absolute path.
package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";

local nbatch              = 8;    --Number of images to process in parallel
local nxSize              = 128;    --CIFAR images are 32 x 32
local nySize              = 128;
local patchSize           = 8;
local stride              = 4
local displayPeriod       = 400;   --Number of timesteps to find sparse approximation
local numEpochs           = 1;     --Number of times to run through dataset
local numImages           = 5; --Total number of images in dataset
local stopTime            = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep           = displayPeriod;
local initialWriteTime    = displayPeriod;

local inputPath           = "../img128.txt";
local outputPath          = "../output/";
local checkpointPeriod    = (displayPeriod * 50); -- How often to write checkpoints

local dictionarySize      = 128;   --Number of patches/elements in dictionary 
local dictionaryFile      = nil;   --nil for initial weights, otherwise, specifies the weights file to load.
local plasticityFlag      = true;  --Determines if we are learning our dictionary or holding it constant
local timeConstantTauConn = 500;   --Weight momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax               = 0.04;    --The learning rate
local VThresh             = 0.1;  --The threshold, or lambda, of the network
local AMin                = 0;
local AMax                = infinity;
local AShift              = VThresh;  --This being equal to VThresh is a soft threshold
local VWidth              = 0;
local timeConstantTau   = 100;   --The integration tau for sparse approximation
local weightInit          = 1.0;

-- Base table variable to store
local pvParameters = {

   --Layers------------------------------------------------------------
   --------------------------------------------------------------------   
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = (displayPeriod * 10);
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = "ImageNet.params";
      randomSeed                          = 1234567890;
      nx                                  = nxSize;
      ny                                  = nySize;
      nbatch                              = nbatch;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints"; --The checkpoint output directory
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointPeriod; --How often to checkpoint
      checkpointIndexWidth                = -1; -- Automatically select width of index in checkpoint directory name
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      initializeFromCheckpointDir         = "";
      errorOnNotANumber                   = false;
   };

   AdaptiveTimeScales = {
      groupType = "AdaptiveTimeScaleProbe";
      targetName                          = "V1EnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "AdaptiveTimeScales.txt";
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      baseMax                             = 0.06;  -- Initial upper bound for timescale growth
      baseMin                             = 0.05;  -- Initial value for timescale growth
      tauFactor                           = 0.03;  -- Percent of tau used as growth target
      growthFactor                        = 0.025; -- Exponential growth factor. The smaller value between this and the above is chosen. 
      writeTimeScalesFieldnames           = false;
   };

   Input = {
      groupType = "ImageLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputPath;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = true;
      normalizeStdDev                     = true;
      useInputBCflag                      = false;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byImage";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };

   InputError = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };


   V1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                   = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };

   CloneV1 = {
      groupType = "CloneVLayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 2;
      writeStep                           = -1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      triggerLayerName                    = NULL;
      originalLayerName                   = "V1";
   };

   V1Error = {
      groupType = "HyPerLayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };


   InputRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   V1P1Recon = {
      groupType = "HyPerLayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };

   P1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/16;
      nyScale                             = 1/16;
      nf                                  = dictionarySize*2;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                   = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };

   CloneP1 = {
      groupType = "CloneVLayer";
      nxScale                             = 1/16;
      nyScale                             = 1/16;
      nf                                  = dictionarySize*2;
      phase                               = 2;
      writeStep                           = -1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      triggerLayerName                    = NULL;
      originalLayerName                   = "P1";
   };

   P1Error = {
      groupType = "HyPerLayer";
      nxScale                             = 1/16;
      nyScale                             = 1/16;
      nf                                  = dictionarySize*2;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };

   P1P2Recon = {
      groupType = "HyPerLayer";
      nxScale                             = 1/16;
      nyScale                             = 1/16;
      nf                                  = dictionarySize*2;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
 
   };

   P2 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/128;
      nyScale                             = 1/128;
      nf                                  = dictionarySize*4;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                   = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };



--Connections ------------------------------------------------------
--------------------------------------------------------------------

   InputToError = {
      groupType = "RescaleConn";
      preLayerName                        = "Input";
      postLayerName                       = "InputError";
      channelCode                         = 0;
      delay                               = {0.000000};
      scale                               = weightInit;
   };

   ErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName                        = "InputError";
      postLayerName                       = "V1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "V1ToInputError";
   };



   V1ToInputError = {
      groupType = "MomentumConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputError";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      minNNZ                              = 0;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      immediateWeightUpdate               = true;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      writeCompressedWeights              = false;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      combine_dW_with_W_flag              = false;
      nxp                                 = patchSize;
      nyp                                 = patchSize;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      timeConstantTauConn                 = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   V1ToRecon = {
      groupType = "CloneConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "V1ToInputError";
   };

   ReconToError = {
      groupType = "IdentConn";
      preLayerName                        = "InputRecon";
      postLayerName                       = "InputError";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   P1ToV1Error = {
      groupType = "MomentumConn";
      preLayerName                        = "P1";
      postLayerName                       = "V1Error";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      minNNZ                              = 0;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      immediateWeightUpdate               = true;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      writeCompressedWeights              = false;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      combine_dW_with_W_flag              = false;
      nxp                                 = patchSize;
      nyp                                 = patchSize;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      timeConstantTauConn                 = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   V1ErrorToP1 = {
      groupType = "TransposeConn";
      preLayerName                        = "V1Error";
      postLayerName                       = "P1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "P1ToV1Error";
   };

  V1ErrorToV1 = {
      groupType = "IdentConn";
      preLayerName                        = "V1Error";
      postLayerName                       = "V1";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

  V1ReconToV1Error = {
      groupType = "IdentConn";
      preLayerName                        = "V1P1Recon";
      postLayerName                       = "V1Error";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };


   P1ToV1Recon = {
      groupType = "CloneConn";
      preLayerName                        = "P1";
      postLayerName                       = "V1P1Recon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "P1ToV1Error";
   };

   P2ToP1Error = {
      groupType = "MomentumConn";
      preLayerName                        = "P2";
      postLayerName                       = "P1Error";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      minNNZ                              = 0;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      immediateWeightUpdate               = true;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      writeCompressedWeights              = false;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      combine_dW_with_W_flag              = false;
      nxp                                 = patchSize;
      nyp                                 = patchSize;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
      timeConstantTauConn                 = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   P1ErrorToP2 = {
      groupType = "TransposeConn";
      preLayerName                        = "P1Error";
      postLayerName                       = "P2";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "P2ToP1Error";
   };

  P1ErrorToP1 = {
      groupType = "IdentConn";
      preLayerName                        = "P1Error";
      postLayerName                       = "P1";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

  P1ReconToP1Error = {
      groupType = "IdentConn";
      preLayerName                        = "P1P2Recon";
      postLayerName                       = "P1Error";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };


   P2ToP1Recon = {
      groupType = "CloneConn";
      preLayerName                        = "P2";
      postLayerName                       = "P1P2Recon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      originalConnName                    = "P2ToP1Error";
   };
   V1ConeToV1Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneV1";
      postLayerName                       = "V1Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };
   P1ConeToP1Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneP1";
      postLayerName                       = "P1Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };




  -- Activity Triggered --

   P2ToInput = {
      groupType = "HyPerConn";
      preLayerName                        = "P2";
      postLayerName                       = "Input";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformWeight";
      weightInit                          = 0;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      immediateWeightUpdate               = true;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep*5;
      initialWriteTime                    = initialWriteTime;
      writeCompressedWeights              = false;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      combine_dW_with_W_flag              = false;
      nxp                                 = 128;
      nyp                                 = 128;
      nfp                                 = 3;
      shrinkPatches                       = false;
      normalizeMethod                     = "none";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      useMask                             = false;
   }; 


   --Probes------------------------------------------------------------
   --------------------------------------------------------------------

   V1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = nil;
   };
   P1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = "V1EnergyProbe";
   };

   P2EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P2EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = "V1EnergyProbe";
   };



   InputErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "InputError";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "InputErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };

   V1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "V1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1L1NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = VThresh;
      maskLayerName                       = nil;
   };
  InputV1ErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "V1Error";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1ErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };
   P1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "P1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P1L1NormEnergyProbe.txt";
      energyProbe                         = "P1EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };

  InputP1ErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "P1Error";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P1ErrorL2NormEnergyProbe.txt";
      energyProbe                         = "P1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };
   P2L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "P2";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "P2L1NormEnergyProbe.txt";
      energyProbe                         = "P2EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };





} --End of pvParameters

if dictionaryFile ~= nil then
   pvParameters.V1ToInputError.weightInitType  = "FileWeight";
   pvParameters.V1ToInputError.initWeightsFile = dictionaryFile;
end
-- Print out PetaVision approved parameter file to the console
local file = io.open("./ImageNet.params", "w");
io.output(file);
pv.printConsole(pvParameters)
io.close(file);
-- The & makes it run without blocking execution
--os.execute("../../../python/draw -p -a " .. "./ImageNet.params &");
os.execute("mpirun -np 1 ../../../build/tests/BasicSystemTest/Release/BasicSystemTest -p ImageNet.params -t 4");
