# IRL-market-model-DJI-calibration-project
## Econometric estimation of an IRL-based market portfolio model

In this project you will: 

- Explore and estimate an IRL-based model of market returns that is based on IRL of a market-optimal portfolio 
- Investigate the role and impact of choices of different signals on model estimation and trading strategies
- Compare simple IRL-based and UL-based trading strategies

**Instructions for project structure:**

- The project consists of Three parts. The parts are as follows (more detailed instructions are in specific cells below):

- **Part 1**: Completeing the model estimation for the DJI portfolio of 30 stocks, and simple signals such as simple moving averages constructed below.

- **Part 2**: Propose other signals and investigate the dynamics for market caps obtained with alternative signals. Present your conclusions and observations.

- **Part 3**: Repeated the analysis for the S&P portfolio. We have build a data file, build signals, and repeat the model estimation process with our new dataset.


## The IRL-based model of stock returns

We know that optimal investment policy in the problem of inverse portfolio optimization is a Gaussian policy

𝜋𝜃(𝐚𝑡|𝐲𝑡)=(𝐚𝑡|𝐀0+𝐀1𝐲𝐭,𝚺𝐩)

Here 𝐲𝑡
is a vector of dollar position in the portfolio, and 𝐀0, 𝐀1 and Σ𝑝

are parameters defining a Gaussian policy.

We know that such Gaussian policy is found for both cases of a single investor and a market portfolio. We also sketched a numerical scheme that can iteratively compute coefficients 𝐀0
, 𝐀1 and Σ𝑝

using a combination of a RL algorithm called G-learning and a trajectory optimization algorithm.

In this project, we will explore implications and estimation of this IRL-based model for the most interesting case - the market portfolio. It turns out that for this case, the model can be estimated in an easier way using a conventional Maximum Likelihood approach. To this end, we will re-formulate the model for this particular case in three easy steps.

Recall that for a vector of 𝑁
stocks, we introduced a size 2𝑁-action vector 𝐚𝑡=[𝐮(+)𝑡,𝐮(−)𝑡], so that an action 𝐮𝑡 was defined as a difference of two non-negative numbers 𝐮𝑡=𝐮(+)𝑡−𝐮(−)𝑡=[1,−1]𝐚𝑡≡1𝑇−1𝐚𝑡

.

Therefore, the joint distribution of 𝐚𝑡=[𝐮(+)𝑡,𝐮(−)𝑡]
is given by our Gaussian policy 𝜋𝜃(𝐚𝑡|𝐲𝑡). This means that the distribution of 𝐮𝑡=𝐮(+)𝑡−𝐮(−)𝑡

is also Gaussian. Let us write it therefore as follows:

𝜋𝜃(𝐮𝑡|𝐲𝑡)=(𝐮𝑡|𝐔0+𝐔1𝐲𝐭,𝚺𝐮)

Here 𝐔0=1𝐓−1𝐀0
and 𝐔1=1𝐓−1𝐀1

.

This means that 𝐮𝑡

is a Gaussian random variable that we can write as follows:

𝐮𝑡=𝐔0+𝐔1𝐲𝐭+𝜀(𝐮)𝐭=𝐔0+𝐔(𝐱)1𝐱𝐭+𝐔(𝐳)1𝐳𝐭+𝜀(𝐮)𝐭

where 𝜀(𝑢)𝑡∼(0,Σ𝑢)

is a Gaussian random noise.

The most important feature of this expression that we need going forward is is linear dependence on the state 𝐱𝑡
. This is the only result that we will use in order to construct a simple dynamic market model resulting from our IRL model. We use a deterministic limit of this equation, where in addition we set 𝐔0=𝐔(𝐳)1=0, and replace 𝐔(𝐱)1→𝜙

to simplify the notation. We thus obtain a simple deterministic policy

𝐮𝑡=𝜙𝐱𝑡

Next, let us recall the state equation and return equation (where we reinstate a time step Δ𝑡
, and ∘

stands for an element-wise (Hadamard) product):

𝑋𝑡+Δ𝑡=(1+𝑟𝑡Δ𝑡)∘(𝑋𝑡+𝑢𝑡Δ𝑡)
𝑟𝑡=𝑟𝑓+𝐰𝐳𝑡−𝜇𝑢𝑡+𝜎Δ𝑡⎯⎯⎯⎯√𝜀𝑡
where 𝑟𝑓 is a risk-free rate, Δ𝑡 is a time step, 𝐳𝑡 is a vector of predictors with weights 𝐰, 𝜇 is a market impact parameter with a linear impact specification, and 𝜀𝑡∼(⋅|0,1)

is a white noise residual.

Eliminating 𝑢𝑡
from these expressions and simplifying, we obtain
Δ𝑋𝑡=𝜇𝜙(1+𝜙Δ𝑡)∘𝑋𝑡∘(𝑟𝑓(1+𝜙Δ𝑡)+𝜙𝜇𝜙(1+𝜙Δ𝑡)−𝑋𝑡)Δ𝑡+(1+𝜙Δ𝑡)𝑋𝑡∘[𝐰𝐳𝑡Δ𝑡+𝜎Δ𝑡⎯⎯⎯⎯√𝜀𝑡]
Finally, assuming that 𝜙Δ𝑡≪1 and taking the continuous-time limit Δ𝑡→𝑑𝑡

, we obtain

𝑑𝑋𝑡=𝜅∘𝑋𝑡∘(𝜃𝜅−𝑋𝑡)𝑑𝑡+𝑋𝑡∘[𝐰𝐳𝑡𝑑𝑡+𝜎𝑑𝑊𝑡]
where 𝜅=𝜇𝜙, 𝜃=𝑟𝑓+𝜙, and 𝑊𝑡

is a standard Brownian motion.

Please note that this equation describes dynamics with a quadratic mean reversion. It is quite different from models with linear mean reversion such as the Ornstein-Uhlenbeck (OU) process.

Without signals 𝐳𝑡

, this process is known in the literature as a Geometric Mean Reversion (GMR) process. It has been used (for a one-dimensional setting) by Dixit and Pyndick (" Investment Under Uncertainty", Princeton 1994), and investigated (also for 1D) by Ewald and Yang ("Geometric Mean Reversion: Formulas for the Equilibrium Density and Analytic Moment Matching", {\it University of St. Andrews Economics Preprints}, 2007). We have found that such dynamics (in a multi-variate setting) can also be obtained for market caps (or equivalently for stock prices, so long as the number of shares is held fixed) using Inverse Reinforcement Learning!

(For more details, see I. Halperin and I. Feldshteyn, "Market Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand Inference with Free Energy. (or, How We Learned to Stop Worrying and Love Bounded Rationality)", https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=3174498)  

## Conclusion:

### Part 1 
When the code for TensorFlow (cell with session implementation) is executed, either it converges around 700 to 1000 iterations initially or after a few times, and sometimes after that but within 5000 iterations or does not converge after 5000 iterations. This is due to the randomization of the weights at each initialization. It is fact that the results are unpredictable due to randomization of weights, to be able to reproduce the results, set seed value to some number before running the session. To ensure that the model definitely converges, the number of iterations has to be set to a larger value which may be up to 10,000 or more. If the first session's iterations itself fails after applying the seed value, then re-run the session only after changing the seed value. We can see from the overlapping plot that we have convergence. The moving average signals have been calibrated.

### Part 2
Result I ran the model only once and converged at 1060 iterations, which is similar to Part 1 which converged around 700 to 1000 iterations or does not converge after 5000 iterations. I would assume it is due to the randomization of the weights at each initialization. Looking at the results, we can see the weight 1 is more favorable which is assigned to our short position. As we can see from the overlapping curves plotted, which means we converged to the predicted value.

### Part 3
Calibration got for the 418 assets of the S&P500 aren't as accurate as that got with the Dow Jones Data Set. We must keep in mind that data provided in the second course is not Market Cap but Closed Price. This model should be tested with actual market capitalization of the S&P500. I haven't been able to get this data so far. Thus, I upload the code with the data provided by the course
