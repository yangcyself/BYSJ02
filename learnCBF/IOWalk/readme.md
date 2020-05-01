The pipeline in this experiment is:

1. Random sample Bezier and PID parameters, run the controller and record
2. Pick the states that the robot is still 'walking' after one second, and the states that the robot fall in 0.5 seconds
3. Do SVM to classify these states, so the CBF is fitted

