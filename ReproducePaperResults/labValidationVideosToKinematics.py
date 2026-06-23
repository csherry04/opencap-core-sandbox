"""
---------------------------------------------------------------------------
OpenCap: labValidationVideosToKinematics.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Scott Uhlrich, Antoine Falisse

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This script processes videos to estimate kinematics with the data from the 
"OpenCap: 3D human movement dynamics from smartphone videos" paper. The dataset
includes 10 subjects, with 2 sessions per subject. The first session includes
static, sit-to-stand, squat, and drop jump trials. The second session includes
walking trials.

The dataset is available on SimTK: https://simtk.org/frs/?group_id=2385. Make
sure you download the dataset with videos: LabValidation_withVideos.

In the paper, we compared 3 camera configurations: 
    2 cameras at +/- 45deg ('2-cameras'), 
    3 cameras at +/- 45deg and 0deg ('3-cameras'), and 
    5 cameras at +/- 45deg, +/- 70deg, and 0deg ('5-cameras'); 
where 0deg faces the participant. Use the variable cameraSetups below to select
which camera configuration to use. In the paper, we also compared three 
algorithms for pose detection: OpenPose at default resolution, OpenPose
with higher accuracy, and HRNet. HRNet is not supported on Windows, and we
therefore do not support it here (it is supported on the web application). To
use OpenPose at default resolution, set the variable resolutionPoseDetection to
'default'. To use OpenPose with higher accuracy, set the variable 
resolutionPoseDetection to '1x1008_4scales'. Take a look at 
Examples/reprocessSessions for more details about OpenPose settings and the
GPU requirements. Please note that we have updated OpenCap since submitting
the paper. As part of the updates, we re-trained the deep learning model we
use to predict anatomical markers from video keypoints, and we updated how
videos are time synchronized. These changes might have a slight effect
on the results.
"""

# %% Paths and imports.
import os
import sys
import shutil
import yaml

repoDir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
sys.path.append(repoDir)

from main import main
from utils import importMetadata
from camera_routing import select_camera_setup_for_trial, select_cameras_for_trial

# %% User inputs
# Enter the path to the folder where you downloaded the data. The data is on
# SimTK: https://simtk.org/frs/?group_id=2385 (LabValidation_withVideos).
# In this example, our path looks like:
#   C:/Users/opencap/Documents/LabValidation_withVideos/subject2
#   C:/Users/opencap/Documents/LabValidation_withVideos/subject3
#   ...
dataDir = '/Users/callumsherry/opencap-sandboxes/opencap-core-sandbox/Data/LabValidation/'

# The dataset includes 2 sessions per subject.The first session includes
# static, sit-to-stand, squat, and drop jump trials. The second session 
# includes walking trials. The sessions are named <subject_name>_Session0 and 
# <subject_name>_Session1.
# Smoke test: run one local lab-validation session before launching the full
# paper batch.
sessionNames = ['subject2_Session0']

# Full paper batch:
# sessionNames = ['subject2_Session0', 'subject2_Session1',
#                 'subject3_Session0', 'subject3_Session1',
#                 'subject4_Session0', 'subject4_Session1',
#                 'subject5_Session0', 'subject5_Session1',
#                 'subject6_Session0', 'subject6_Session1',
#                 'subject7_Session0', 'subject7_Session1',
#                 'subject8_Session0', 'subject8_Session1',
#                 'subject9_Session0', 'subject9_Session1',
#                 'subject10_Session0', 'subject10_Session1',
#                 'subject11_Session0', 'subject11_Session1']

# We only support OpenPose on Windows.
poseDetectors = ['OpenPose']

# Select the camera configuration you would like to use.
# cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
cameraSetups = ['2-cameras']

# Select the resolution at which you would like to use OpenPose. More details
# about the options in Examples/reprocessSessions. In the paper, we compared 
# 'default' and '1x1008_4scales'.
resolutionPoseDetection = 'default'

# Since the prepint release, we updated a new augmenter model. To use the model
# used for generating the paper results, select v0.1. To use the latest model
# (now in production), select v0.2.
augmenter_model = 'v0.2'

# %% Data re-organization
# To reprocess the data, we need to re-organize the data so that the folder
# structure is the same one as the one expected by OpenCap. It is only done
# once as long as the variable overwriteRestructuring is False. To overwrite
# flip the flag to True.
overwriteRestructuring = False
subjects = sorted({sessionName.split('_')[0] for sessionName in sessionNames})
for subject in subjects:
    pathSubject = os.path.join(dataDir, subject)
    pathVideos = os.path.join(pathSubject, 'VideoData')    
    for session in os.listdir(pathVideos):
        if 'Session' not in session:
            continue
        pathSession = os.path.join(pathVideos, session)
        pathSessionNew = os.path.join(dataDir, 'Data', subject + '_' + session)
        if os.path.exists(pathSessionNew) and not overwriteRestructuring:
            continue
        os.makedirs(pathSessionNew, exist_ok=True)
        # Copy metadata
        pathMetadata = os.path.join(pathSubject, 'sessionMetadata.yaml')
        shutil.copy2(pathMetadata, pathSessionNew)
        pathMetadataNew = os.path.join(pathSessionNew, 'sessionMetadata.yaml')
        # Adjust model name
        sessionMetadata = importMetadata(pathMetadataNew)
        sessionMetadata['openSimModel'] = (
            'LaiUhlrich2022')
        with open(pathMetadataNew, 'w') as file:
                yaml.dump(sessionMetadata, file)        
        for cam in os.listdir(pathSession):
            if "Cam" not in cam:
                continue            
            pathCam = os.path.join(pathSession, cam)
            pathCamNew = os.path.join(pathSessionNew, 'Videos', cam)
            pathInputMediaNew = os.path.join(pathCamNew, 'InputMedia')
            # Copy videos.
            for trial in os.listdir(pathCam):
                pathTrial = os.path.join(pathCam, trial)
                if not os.path.isdir(pathTrial):
                    continue
                pathVideo = os.path.join(pathTrial, trial + '.avi')
                pathTrialNew = os.path.join(pathInputMediaNew, trial)
                os.makedirs(pathTrialNew, exist_ok=True)
                shutil.copy2(pathVideo, pathTrialNew)
            # Copy camera parameters
            pathParameters = os.path.join(pathCam, 
                                          'cameraIntrinsicsExtrinsics.pickle')
            shutil.copy2(pathParameters, pathCamNew)

# %% Fixed settings.
# The dataset contains 5 videos per trial. The 5 videos are taken from cameras
# positioned at different angles: Cam0:-70deg, Cam1:-45deg, Cam2:0deg, 
# Cam3:45deg, and Cam4:70deg where 0deg faces the participant. Depending on the
# cameraSetup, we load different videos.
cam2sUse = {'5-cameras': ['Cam0', 'Cam1', 'Cam2', 'Cam3', 'Cam4'], 
            '3-cameras': ['Cam1', 'Cam2', 'Cam3'], 
            '2-cameras': ['Cam1', 'Cam3']}
fullCameraSetup = max(cam2sUse, key=lambda setup: len(cam2sUse[setup]))


def copy_model_files(source_model_dir, target_model_dir):
    os.makedirs(target_model_dir, exist_ok=True)
    for file in os.listdir(source_model_dir):
        shutil.copy2(
            os.path.join(source_model_dir, file),
            os.path.join(target_model_dir, file))


def copy_model_folder(data_dir, session_name, pose_detector,
                      resolution_pose_detection, source_camera_setup,
                      target_camera_setup):
    session_dir = os.path.join(data_dir, 'Data', session_name)
    open_sim_base = os.path.join(
        session_dir, 'OpenSimData',
        pose_detector + '_' + resolution_pose_detection)
    source_model_dir = os.path.join(open_sim_base, source_camera_setup, 'Model')
    target_model_dir = os.path.join(open_sim_base, target_camera_setup, 'Model')
    copy_model_files(source_model_dir, target_model_dir)

# # %% Functions for re-processing the data.
def process_trial(trial_name=None, session_name=None, isDocker=False,
                  cam2Use=['all'],
                  intrinsicsFinalFolder='Deployed', extrinsicsTrial=False,
                  alternateExtrinsics=None, markerDataFolderNameSuffix=None,
                  imageUpsampleFactor=4, poseDetector='OpenPose',
                  resolutionPoseDetection='default', scaleModel=False,
                  bbox_thr=0.8, augmenter_model='v0.2', benchmark=False,
                  calibrationOptions=None, offset=True, dataDir=None):

    # Run main processing pipeline.
    main(session_name, trial_name, trial_name, cam2Use, intrinsicsFinalFolder,
          isDocker, extrinsicsTrial, alternateExtrinsics, calibrationOptions,
          markerDataFolderNameSuffix, imageUpsampleFactor, poseDetector,
          resolutionPoseDetection=resolutionPoseDetection,
          scaleModel=scaleModel, bbox_thr=bbox_thr,
          augmenter_model=augmenter_model, benchmark=benchmark, offset=offset,
          dataDir=dataDir)

    return

# %% Process trials.
for count, sessionName in enumerate(sessionNames):    
    # Get trial names.
    pathCam0 = os.path.join(dataDir, 'Data', sessionName, 'Videos', 'Cam0',
                            'InputMedia')    
    # Work around to re-order trials and have the extrinsics trial firs, and
    # the static second (if available).
    trials_tmp = os.listdir(pathCam0)
    trials_tmp = [t for t in trials_tmp if
                  os.path.isdir(os.path.join(pathCam0, t))]
    session_with_static = False
    for trial in trials_tmp:
        if 'extrinsics' in trial.lower():                    
            extrinsics_idx = trials_tmp.index(trial) 
        if 'static' in trial.lower():                    
            static_idx = trials_tmp.index(trial) 
            session_with_static = True            
    trials = [trials_tmp[extrinsics_idx]]
    if session_with_static:
        trials.append(trials_tmp[static_idx])
        for trial in trials_tmp:
            if ('static' not in trial.lower() and 
                'extrinsics' not in trial.lower()):
                trials.append(trial)
    else:
        for trial in trials_tmp:
            if 'extrinsics' not in trial.lower():
                trials.append(trial)
    
    for poseDetector in poseDetectors:
        for cameraSetup in cameraSetups:
            # The second sessions (<>_1) have no static trial for scaling the
            # model. The static trials were collected as part of the first
            # session for each subject (<>_0). We here copy the Model folder
            # from the first session to the second session.
            if sessionName[-1] == '1':
                sessionDir = os.path.join(dataDir, 'Data', sessionName)
                sessionDir_0 = sessionDir[:-1] + '0'
                modelDir_0 = os.path.join(
                    sessionDir_0, 'OpenSimData',
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup,
                    'Model')
                modelDir_1 = os.path.join(
                    sessionDir, 'OpenSimData',
                    poseDetector + '_' + resolutionPoseDetection, cameraSetup,
                    'Model')
                copy_model_files(modelDir_0, modelDir_1)
                    
            # Process trial.
            for trial in trials:                
                print('Processing {}'.format(trial))
                
                # Detect if extrinsics trial to compute extrinsic parameters. 
                if 'extrinsics' in trial.lower():                    
                    extrinsicsTrial = True
                else:
                    extrinsicsTrial = False
                
                # Detect if static trial with neutral pose to scale model.
                if 'static' in trial.lower():                    
                    scaleModel = True
                else:
                    scaleModel = False

                # Static/neutral scaling requires all recorded cameras in
                # main.py. Dynamic trials can use the selected subset.
                trialCameraSetup = select_camera_setup_for_trial(
                    fullCameraSetup, cameraSetup,
                    is_extrinsics_trial=extrinsicsTrial,
                    scale_model=scaleModel)
                cam2Use = select_cameras_for_trial(
                    cam2sUse[fullCameraSetup], cam2sUse[cameraSetup],
                    is_extrinsics_trial=extrinsicsTrial,
                    scale_model=scaleModel)
                
                # Session specific intrinsic parameters
                if 'subject2' in sessionName or 'subject3' in sessionName:
                    intrinsicsFinalFolder = 'Deployed_720_240fps'
                else:
                    intrinsicsFinalFolder = 'Deployed_720_60fps'                    
                    
                process_trial(trial,
                              session_name=sessionName,
                              cam2Use=cam2Use, 
                              intrinsicsFinalFolder=intrinsicsFinalFolder,
                              extrinsicsTrial=extrinsicsTrial,
                              markerDataFolderNameSuffix=trialCameraSetup,
                              poseDetector=poseDetector,
                              resolutionPoseDetection=resolutionPoseDetection,
                              scaleModel=scaleModel, 
                              augmenter_model=augmenter_model,
                              dataDir=dataDir)

                if scaleModel and cameraSetup != trialCameraSetup:
                    copy_model_folder(dataDir, sessionName, poseDetector,
                                      resolutionPoseDetection,
                                      trialCameraSetup, cameraSetup)
