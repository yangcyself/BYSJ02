# The learning algorithm of Control Barrier functions with application in bipedal robots

## dependencies

- pybullet
- cvxpy
- scikit-learn
- scipy
- dill
  - An alternative of pickle which supports lambda functions
- ExperimentSecretary
  - An self-developed utility for logging and analyzing experiment settings and results
  - [github](https://github.com/yangcyself/ExperimentSecretary.git)

## Repo Structure

- ctrl
  - The implementation of all the controllers.
  - Each file defines a class. The root class of all is `CTRL` defined in [rabbitCtrl](./ctrl/rabbitCtrl.py)
  - For detailed explaination about the framework, please see the section [Controllers](#controllers)
- tests
  - The executable files that verifies the implementation or visulize the results
  - The header of each of them writes the purpose of the test
- learnCBF
  - The algorithm implementation and the experiment executables for the learning algorithm of CBF
  - For detailed explaination about the files, please see the section [LearnCBF](#learncbf)
- util
- data
  - The place to store and load experimental data
- globalParameters
  - The settings of the model and flags for whether start a simulation or open a GUI

## Controllers

The controllers are responsible for running simulations, that is, to compute motor torques and then step the simulation. The most important interface of controllers is the [`step` method](./ctrl/rabbitCtrl.py) which forward the simulation for a period. In the `step` method, the procedure goes as the following:

1. Call all the *registered control components* to generate a torque
2. Call all the callback functions and check their return value to determine whether to stop
3. step the simulation
4. loop back to the first step until forwarded the desired amount of time

### Control Components

The controller framework is designed with reusablity and expandability in mind. Thus, the procedure for computing an input torque is seperated into the computing of a series of variables, which are wrapped into different control components. For example, in [`CBFWalker`](./ctrl/CBFWalker.py), the CBF-QP controller is the ultimate control component which uses the result of the PD control of the reference traj defined in [IOLinearCtrl](./ctrl/IOLinearCtrl.py). Both of them use the current states, which is another control components.

Control components are lazily evaluated and cached during one step. It is only evaluated when it is registered in the controller or is the dependency of a registered control component. Because each variable is a constant in each step, so at the second time (in one step) a control component is called, it will read from the cache rather than evaluate again.

To define a control component, one just have to define a method with `@CTRL_COMPONENT` decorator. To use a controll component, just use `self.xxxx` in the body of the method. The framework will take care of all the other dirty works.

```python
@CTRL_COMPONENT
    def gc_Lam(self):
        """
        The lambda of the operational space of the gc
        """
        lam = np.linalg.inv(self.J_gc @ self.RBD_A_inv @ self.J_gc.T
        return lam
```

The controller will run nothing unless at least one control component is registered. To regist a control component, call something like:
```python
ct =  WBC_CTRL() # initlize a WBC controller
WBC_CTRL.WBC.reg(ct,kp = np.array([50,50,50])) # register control component `WBC` to `ct`, with a parameter of `kp`
```

Control component also have other interfaces for debugging and developing, please see the code in [rabbitCtrl.py](./ctrl/rabbitCtrl.py)

### Structure of current controllers

- Root controller
  - rabbitCtrl.py
- WBC controller
  - A simple controller using WBC
- playBackCtrl
  - A utility controller for playing a recorded traj
- Walking controllers
  - IOLinearCtrl
    - The controller process HZD generated reference trajectory, and PD to follow it.
  - CBFCtrl
    - The controller that implemented calculations of CBFs and use a CBF-QP inside.
  - CBFWalker
    - The controller inherits `IOLinearCtrl` and `CBFCtrl`, that can walk with CBF constraints
    - The CBFWalker do not use relabling map, rather, I designed some other features in the state to handel the symmetry.

## LearnCBF

Currently, only the experiment for [HZD based walking](./learnCBF/IOWalk) are done, the algorithm introduced in the graduation paper is implemented in [`learncbf.py`](./tests/testlearncbf.py). To test the result or draw figures, use [`testIOWalkCBF.py`](./learnCBF/IOWalk/testIOWalkCBF.py)