# Release checks

All 13 copied research scripts parsed successfully as Python and match their source SHA-256 hashes. The added smoke check passed for both environments: prescribed success path, rejected nonadjacent transition, and 15-step truncation.

Checks used Python 3.12, NumPy 2.3.5 and Gymnasium 1.3.0 in a temporary packaging environment. These are check-environment versions, not recovered training-environment versions. No CNN training, DQN training, ROS execution, physical experiment, or scientific performance re-evaluation was run for this release.
