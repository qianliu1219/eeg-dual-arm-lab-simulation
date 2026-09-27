# Offline EEG supervisory control for simulated dual-arm laboratory automation

Research code for a simulation prototype combining recorded EEG classification, discrete deep Q-network (DQN) task sequencing, and ROS 2 / Gazebo arm routines. Maintained by Qian Liu, Department of Applied Computer Science, The University of Winnipeg.

**Status:** source release of an existing research prototype. EEG inputs are recordings and model predictions are computed before GUI playback. The interface starts the left-arm routine; the right-arm routine is invoked separately. This is not a live EEG acquisition system, a validated safety controller, or a demonstrated chemical synthesis platform.

## What is included

- Two five-position Gymnasium environments and their PyTorch DQN training scripts.
- Separate left-arm pouring-pose and right-arm stirring-pose execution scripts.
- Blink CNN and EEGNet-style rest-versus-motor-imagery training, supporting preprocessing, stream preparation, and replay GUI.
- Original file provenance and SHA-256 hashes in `SOURCE_MANIFEST.json`.

The 13 research scripts are preserved byte-for-byte. Historical comments such as “real”, “concentration”, “emergency stop” and “unseen subjects” should be read subject to the qualifications below. Packaging documentation and the smoke check were added for this release; they are not new experimental results.

## Quick start: abstract task environments

Use Python 3 and an isolated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dqn.txt
python smoke_check.py
```

The smoke check exercises the hand-specified successful sequence, rejected nonadjacent moves, and the 15-step limit in both environments. It does not evaluate a trained policy or reproduce published learning curves.

Optional commands to train policies (these run NEW training, not the archived experiment):

```bash
python dqn_train_sequence_v3.py
python dqn_train_sequence_right.py
```

Training writes `.pth` weights and PNG curves into the working directory. Original scripts do not fix random seeds and save the final policy, despite the `best_` filename prefix. Dependency files identify required packages, not recovered exact versions of the original experiment.

## EEG pipeline and source data

Install `requirements-eeg.txt` in a separate environment from ROS. Obtain the source datasets under their own terms:

- [PhysioNet EEG Motor Movement/Imagery, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), DOI `10.13026/C28G6P`. The MNE loader retrieves runs 4, 8, 12, using subjects 1–64 for training and 65–80 for development evaluation.
- [Atzingen/EEG_blink_detector](https://github.com/Atzingen/EEG_blink_detector): obtain the EEG-VV data and `trainTestVV` preprocessing artifact at `EEG_blink_detector/` relative to this directory. Underlying blink dataset: Agarwal and Sivakumar, Allerton 2019, DOI `10.1109/ALLERTON.2019.8919795`; CNN method: Iaquinta et al., 2021, DOI `10.33448/rsd-v10i15.22712`.

Run `eeg_blink_cnn_train.py` and `eeg_concentration_train_cnn.py` from this directory after acquiring data. `eeg_prepare_robot_test_streams.py` loads their `.h5` models and writes `eeg_robot_test_streams.pkl`. Original pickle artifacts must come from trusted sources; pickle loading executes Python serialization instructions. Raw EEG, pickles, trained weights, and third-party robot assets are not redistributed here.

## ROS 2 / Gazebo integration

The source targets ROS 2 Humble and Gazebo Fortress with a configured myBuddy robot. An external robot description, simulation world, controller configuration, grippers, and detachable-joint plugins are required. This repository is **not a self-contained ROS workspace**. It does not vendor the original workspace's robot meshes or third-party ROS packages.

The original scripts expect `~/colcon_ws_IGNITION/src/mybuddy_rl/` for the execution scripts and weights. Place the relevant files there or adapt the path constants for your workspace. Place the generated `eeg_robot_test_streams.pkl` in the GUI working directory. After bringing up the external simulation and controllers, run the arm scripts separately in a ROS-sourced terminal. Review their fixed joint poses before use with any different scene. The left controller is `/left_arm_controller/follow_joint_trajectory`; right controller is `/right_arm_controller/follow_joint_trajectory`; right gripper uses `/gripper_action_controller/follow_joint_trajectory`.

`python eeg_interface.py` replays precomputed predictions and launches the left script. Blink and motor-imagery replay are separate selectable modes. Stopping the subprocess does not establish that an already accepted joint trajectory has stopped. Do not connect this prototype to physical machinery as an emergency-stop mechanism.

## Interpretation of existing results

- Blink: reported 98.5% is a development-set result. Supplied upstream preparation balances by resampling before the split and fits scaling before the split; duplicate overlap is possible. The evaluation set is also used for early stopping. It is not a verified held-out-subject benchmark.
- Motor imagery: reported 66.6% is rest versus pooled hand-imagery labels, not general “concentration”. Evaluation subjects are disjoint from training subjects, but evaluation data are reused for model selection and normalized with their own aggregate statistics.
- Blink replay uses window-wise standardization, unlike the upstream training preparation. The classifier metric does not validate this replay preprocessing pipeline.
- DQN curves describe training in a small deterministic symbolic environment, not independent robot-trial success. DQN does not learn joint trajectories.
- No measured chemical outcome, simultaneous multimodal supervisor, physical-robot validation, or verified end-to-end automatic arm handoff is established by these files.

## License, citation and funding

Original project code is released under MIT at the corresponding author's direction. Portions of the blink CNN implementation follow Atzingen's MIT-licensed project; its copyright and license are preserved in `LICENSE-Atzingen.txt` and `THIRD_PARTY_NOTICES.md`. Dataset and robot-asset licenses remain separate. See `CITATION.cff` for software citation; there is no publication DOI yet.


