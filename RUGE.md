## 3.1 Robust and Unbiased Graph Signal Estimator

Our study and analysis in Section 2 have shown that while $\ell_1$-based methods outperform $\ell_2$-based methods in robustness, they still suffer from the accumulated estimation bias, leading to severe performance degradation under large perturbation budgets. This motivates us to design a robust and unbiased graph signal estimator that derives unbiased robust aggregation for GNNs with stronger resilience to attacks.

Theoretically, the estimation bias in Lasso regression has been discovered and analyzed in high-dimensional statistics [23]. Statisticians have proposed adaptive Lasso [23] and many non-convex penalties such as Smoothly Clipped Absolute Deviation (SCAD) [24] and Minimax Concave Penalty (MCP) [25] to alleviate this bias. Motivated by these advancements, we propose a Robust and Unbiased Graph signal Estimator (RUGE) as follows:

$$\arg \min_{\boldsymbol{F}} \mathcal{H}(\boldsymbol{F}) = \sum_{(i,j) \in \mathcal{E}} \rho_{\gamma} \left( \left\| \frac{\boldsymbol{f}_i}{\sqrt{d_i}} - \frac{\boldsymbol{f}_j}{\sqrt{d_j}} \right\|_2 \right) + \lambda \sum_{i \in \mathcal{V}} \|\boldsymbol{f}_i - \boldsymbol{f}_i^{(0)}\|_2^2, \tag{3}$$

where $\rho_{\gamma}(y)$ denotes the function that penalizes the feature differences on edges by MCP:

$$\rho_{\gamma}(y) = \begin{cases} y - \frac{y^2}{2\gamma} & \text{if } y < \gamma \\ \frac{\gamma}{2} & \text{if } y \ge \gamma \end{cases}. \tag{4}$$

As shown in Figure 3, MCP closely approximates the $\ell_1$ norm when $y$ is small since the quadratic term $\frac{y^2}{2\gamma}$ is negligible, and it becomes a constant value when $y$ is large. This transition can be adjusted by the thresholding parameter $\gamma$. When $\gamma$ approaches infinity, the penalty $\rho_{\gamma}(y)$ reduces to the $\ell_1$ norm. Conversely, when $\gamma$ is very small, the "valley" of $\rho_{\gamma}$ near zero is exceptionally sharp, so $\rho_{\gamma}(y)$ approaches the $\ell_0$ norm and becomes a constant for a slightly larger $y$. This enables RUGE to suppress smoothing on edges whose node differences exceeding the threshold $\gamma$. This not only mitigates the estimation bias against outliers but also maintains the estimation accuracy in the absence of outliers. The simulation in Figure 2 verifies that our proposed estimator ($\eta(\boldsymbol{x}) := \rho_{\gamma}(\|\boldsymbol{x}\|_2)$) can recover the true mean despite the increasing outlier ratio when the outlier ratio is below the theoretical optimal breakdown point.

---

### Figure 3: Penalties

The image includes a plot comparing three penalty functions:

* **MCP** (Blue line): Increases linearly then plateaus at $\frac{\gamma}{2}$ when $y \ge \gamma$.
* **$\ell_1$** (Green line): A constant linear increase ($|y|$).
* **$\ell_2$** (Orange line): A quadratic increase ($y^2$).

Would you like me to help you implement the RUGE loss function or the MCP penalty in PyTorch or TensorFlow?
