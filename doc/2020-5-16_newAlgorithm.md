## outer loop

### Collect points

Given a CBF0, generate trajectories with random parameter.

given a trajectory $\xi_{i}[0,t_{i}]$ ,$t_{i}$ is the total time of the trajectory

$$
s_{safe_{i}} = \xi _{i}(t_{i} - 0.5)
\\
s_{bad_{i}} = \mathrm{A}_{discrete}(-0.01) \xi _{i} + \mathrm{B}_{discrete}(-0.01) u^{*}_{i} + \mathrm{g}_{discrete}(-0.01)
$$

$u_{i}$ is the input that can maximizes $\dot{B}$ at the time $t_{i}$

### Solve new CBF

With the collected points as the dataset, do SVM-like optimization.

$$
\min_{P,q,c} \left\|w \right\| \\
\text{s.t.}\quad \forall i\quad y_{i}(x_{i} ^{T} Px_{i} +q^{T} x_{i}  +c)>1  \qquad \text{SVM constraint}\\
\text{P is P.S.D and the ellipse is within the old CBF}\\
$$

where $w$ is all the parameterization of $P,q,c$

## example

整理SVM代码和数据给老师

background:
- lyapunov
- CBF
- paper

控制器框架 -> 实验部分 十几页

理论部分 -> 五页

reference 两三页
