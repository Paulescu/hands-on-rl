# todos

- [x] Make spinup implementation run without MPI.
- [x] Policy gradients with generalized advantage estimation.

- [x] implement working code with the most vanilla PG method that exists.
- [x] train optimal agent

- [x] re-run training.
- [x] evaluate solution and check it works like a charm.

- [x] Generate video best agent.
- [x] Notebook: Actions, states, rewards

- [x] random agent notebook

- [x] vpg v1 training notebook
- [x] vpg v2 training notebook

- [x] blog section vpg v1

- [x] Add success rate.

- [ ] blog section vpg v2
- [ ] key-takeaways
- [ ] Homework
- [ ] Policy gradients theory
- [ ] Polish readme and commit it to git

## equations
J(\theta) =
Q^*(s, a=1) = p_{10} * x + p_{11} * v + p_{12} * \theta + p_{13} * \omega + p_{14}
\tau = (s_0, a_0, s_1, a_1... s_T, a_T)

J(\theta) = E_{\tau \sim \pi_{\theta}}R(\tau)
\max_{\theta}J(\theta)
J(\theta_0) < J(\theta_1)
\theta_0 \rightarrow \theta_1
\theta_0 + \alpha * \nabla J(\theta_0) = \theta_1
\theta_0 \rightarrow \theta_1 \rightarrow \theta_2 \rightarrow ...  \theta_N

J(\theta_0) < J(\theta_1) < J(\theta_2) < ... < J(\theta_N)

\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m}

\pi_{\theta}

\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{T} \nabla{\log{\pi_{\theta}(a_t | s_t)}} * R(\tau_i)
\tau = (s_0, a_0, s_1, a_1... s_T, a_T)

\text{reward-to-go}(t) = \sum_{t'=t}^{T} r_{t'}

# references




