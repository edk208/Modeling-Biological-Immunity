-- Path to OpenPV. Replace this with an absolute path.
package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";

local nbatch              = 1;    --Number of images to process in parallel
local nxSize              = 32;    --CIFAR images are 32 x 32
local nySize              = 32;
local patchSize           = 8;
local stride              = 4; 
local displayPeriod       = 4000;   --Number of timesteps to find sparse approximation
local numEpochs           = 1;     --Number of times to run through dataset
local numImages           = 1; --Total number of images in dataset
local stopTime            = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep           = displayPeriod/10;
local initialWriteTime    = displayPeriod/10;

local inputPath           = "../inputTest2.txt";
local outputPath          = "../outputOneTest2/";
local checkpointPeriod    = (displayPeriod * 100); -- How often to write checkpoints

local dictionarySize      = 64;   --Number of patches/elements in dictionary 
local dictionaryFileV1      = "../Checkpoint6000000One/V1ToInputError_W.pvp";   --nil for initial weights, otherwise, specifies the weights file to load.
local dictionaryFileP1V1      = "../Checkpoint6000000One/P1ToV1Error_W.pvp";   --nil for initial weights, otherwise, specifies the weights file to load.
local plasticityFlag      = false;  --Determines if we are learning our dictionary or holding it constant
local timeConstantTauConn = 500;   --Weight momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax               = 0.05;    --The learning rate
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
      printParamsFilename                 = "MNistOneTest2.params";
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
      nf                                  = 1;
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

   InputPvp = {
      groupType = "PvpLayer";
      nxScale                             = 1/32;
      nyScale                             = 1/32;
      nf                                  = dictionarySize*2;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = "../Four.pvp";
      offsetAnchor                        = "tr";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = false;
      normalizeStdDev                     = false;
      useInputBCflag                      = false;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byFile";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };
   InputError = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 1;
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
  V1init = {
      groupType                           = "HyPerLayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 0;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      writeStep                           = -1;
      sparseLayer                         = false;
      displayPeriod                       = 0;
      updateGpu                           = false;
      dataType                            = nil;
      clearGSynInterval                   = 0;
      normalizeLuminanceFlag              = false;
      normalizeStdDev                     = false;
      autoResizeFlag                      = false;
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
      triggerLayerName                    = "Input";
      triggerBehavior                     = "resetStateOnTrigger";
      triggerResetLayerName               = "V1init";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = false;
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
      nf                                  = 1;
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

  P1init = {
      groupType                           = "HyPerLayer";
      nxScale                             = 1/32;
      nyScale                             = 1/32;
      nf                                  = dictionarySize*2;
      phase                               = 0;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      writeStep                           = -1;
      sparseLayer                         = false;
      displayPeriod                       = 0;
      updateGpu                           = false;
      dataType                            = nil;
      clearGSynInterval                   = 0;
      normalizeLuminanceFlag              = false;
      normalizeStdDev                     = false;
      autoResizeFlag                      = false;
   };

   P1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/32;
      nyScale                             = 1/32;
      nf                                  = dictionarySize*2;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      triggerLayerName                    = "Input";
      triggerBehavior                     = "resetStateOnTrigger";
      triggerResetLayerName               = "P1init";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = false;
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

   V1P1applyThresh = {
      groupType = "ANNLayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 5;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh*2;
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      dataType                            = nil;
      VThresh                             = VThresh*2;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
 
   };
   V1P1VisionRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 1;
      phase                               = 5;
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

   InputPvpToP1 = {
      groupType = "RescaleConn";
      preLayerName                        = "InputPvp";
      postLayerName                       = "P1";
      channelCode                         = 0;
      delay                               = {0.000000};
      scale                               = 0;
   };



   ErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName                        = "InputError";
      postLayerName                       = "V1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
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

   -------Multimdoal Connections ---------

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
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "P1ToV1Error";
   };

   V1ConeToV1Error = {
      groupType = "IdentConn";
      preLayerName                        = "CloneV1";
      postLayerName                       = "V1Error";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
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

   V1P1VisionReconConn = {
      groupType = "CloneConn";
      preLayerName                        = "V1P1applyThresh";
      postLayerName                       = "V1P1VisionRecon";
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

   V1P1ReconToThresh = {
      groupType = "RescaleConn";
      preLayerName                        = "V1P1Recon";
      postLayerName                       = "V1P1applyThresh";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      scale                               = 1;
   };
   V1P1VisionReconToV1 = {
      groupType = "RescaleConn";
      preLayerName                        = "V1P1VisionRecon";
      postLayerName                       = "InputError";
      channelCode                         = 0;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
      scale                               = 0.85; 
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

   InputVisionErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "InputError";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "InputVisionErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
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

   V1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "V1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1L1NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.025;
      maskLayerName                       = nil;
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


} --End of pvParameters
if dictionaryFileV1 ~= nil then
   pvParameters.V1ToInputError.weightInitType  = "FileWeight";
   pvParameters.V1ToInputError.initWeightsFile = dictionaryFileV1;

   pvParameters.P1ToV1Error.weightInitType  = "FileWeight";
   pvParameters.P1ToV1Error.initWeightsFile = dictionaryFileP1V1;
end

-- Print out PetaVision approved parameter file to the console
local file = io.open("./MNistOneTest2.params", "w");
io.output(file);
pv.printConsole(pvParameters)
io.close(file);
-- The & makes it run without blocking execution
--os.execute("../../../python/draw -p -a " .. "./MNistOneTest2.params &");
os.execute("mpirun -np 1 ../../../build/tests/BasicSystemTest/Release/BasicSystemTest -p MNistOneTest2.params");
