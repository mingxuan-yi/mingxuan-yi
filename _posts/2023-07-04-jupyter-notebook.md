---
layout: post
title: On the Probability Flow ODE of the Langevin Dynamics
date: 2023-11-05 08:57:00-0400
description: an example of GANs without generators
#tags: formatting jupyter
#categories: sample-posts
giscus_comments: true
related_posts: false
toc:
  beginning: true
---

## 1. Introduction
This post provides a simple numerical approach using [PyTorch](https://pytorch.org/) to simulate the probability flow ordinary differential equation (ODE) of the Langevin dynamics. The implementation is super simple, one just needs to slightly modify a code of [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). Such implementation can be understood as ''non-parametric GANs'', which is an alternative view on GAN via the probability flow ODE, more details can be found in my paper ''[MonoFlow: Rethinking Divergence GANs via the Perspective of Wasserstein Gradient Flows](https://arxiv.org/abs/2302.01075)'' , or another excellant paper ''[Unifying GANs and Score-Based Diffusion as Generative Particle Models](https://arxiv.org/abs/2305.16150)'' by [Jean-Yves Franceschi](https://jyfranceschi.fr/). Briefly speaking, GANs can work without generators as a direct particle flow methods similar to diffusion models.

<figure style="text-align:center;">
  <img src="{{ site.baseurl }}/assets/img/blog_pic/particles.gif" alt="Prob flow ODE" class="img-fluid rounded z-depth-1" style="width: 40%; height: auto; margin-left: auto; margin-right: auto;">
  <figcaption>Figure 1. Transporting particles via the probability flow ODE.</figcaption>
</figure>

## 2. Langevin dynamics and its probability flow ODE
The Langevin dynamics reads a stochastic differential equation (SDE),
\begin{align}
\mathrm{d} \mathbf{x}\_t = \nabla\_\mathbf{x} \log p(\mathbf{x}\_t)\mathrm{d}t + \sqrt{2} \mathrm{d}\mathbf{w}\_t,
\end{align}
where $$\mathbf{w}_t$$ represents the Brownian motion. Under mild conditions, the marginal $$\mathbf{x}_t \sim q_t$$ converges to the stationary distribution $$ p$$ or at least some local optima with frist order statinary contision, if given sufficent long time. In order to numerically simulate the Langevin dynamics, we can use the [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) to discretize the SDE, this gives the well-known unadjusted Langevin algorithm (ULA),
\begin{align}
\mathbf{x}\_{i+1} \leftarrow \mathbf{x}\_{i} + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}\_i) + \sqrt{2\epsilon} \mathcal{N}(0, I), \quad i=0, 1, 2\cdots
\end{align}
ULA has been wide used in Bayesian inference and generative modelings as a sampling method. For example, training Bayesian neural networks, energy-based models, or the earliest version of diffusion models which uses annealing techniques for the Langevin dynamics. Sampling via the ULA relies on the knowledge of the density function of the target distribution, this can be an obstacle especially in which the target distribution can only represented by some data points, e.g., image generation and simulation-based inference. In Figure 1, Langevin dynamics is not available because the target $$p(\mathbf{x})$$ is just a collection of data points. 




In [score-based diffusion models](https://yang-song.net/blog/2021/score/#probability-flow-ode), [Song](https://yang-song.net/) et al. showed that each SDE has an associated probability flow ODE sharing the same marginal $$q_t$$, see Eq. (13) in Song, et. al., 2021. Simply using this result, we can convert the Langevin SDE to the associated ODE,
\begin{align}
\mathrm{d} \mathbf{x}\_t = \Big [\nabla\_\mathbf{x} \log p(\mathbf{x}\_t)- \nabla\_\mathbf{x} \log q_t(\mathbf{x}\_t)\Big]\mathrm{d}t.
\end{align}
Similarly, using the Euler method to discretize the ODE gives 
\begin{align}
\mathbf{x}\_{i+1} \leftarrow \mathbf{x}\_{i} + \epsilon \big[\nabla_{\mathbf{x}} \log {p(\mathbf{x}\_i)}- \nabla_{\mathbf{x}} \log {q_{i}({\mathbf{x}\_i})}\big], \quad i=0, 1, 2\cdots
\end{align}
It is worth noting the vector field of the probability flow ODE is the gradient of the logrithm density ratio $$\log \big[p(\mathbf{x}_i) / q_{i}({\mathbf{x}_i})\big]$$.



#### Estimating the vector field with the binary classification
Recall that in , training a discrminator to distinguish samples from $$$$
\begin{align}
\max\_D \quad \mathbb{E}\_{p} \big[\log D(\mathbf{x})\big]+ \mathbb{E}\_{q_i} \big[\log (1-D(\mathbf{x}))\big]
\end{align}
The optimal discrminator is given by 
\begin{align}
D^{\*}(\mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x}) + q\_i(\mathbf{x})}, \quad \sigma^{-1}\big(D^{\*}(\mathbf{x})\big)  =\log \frac{p(\mathbf{x}\_i)}{q_{i}({\mathbf{x}\_i})}
\end{align}
where $$\sigma^{-1}$$ is the inverse of the Sigmoid activation, and $$\sigma^{-1}\big(D^{*}(\mathbf{x})\big)$$ is the logit output of the binary classifier.
This gives us a strategy for sampling via the probability flow ODE which is similar to GANs, we can train a discrminator to obtain the log density ratio between $$p$$ and $$q_i$$, such that the vector field is given by 
\begin{align}
\nabla_{\mathbf{x}} \log {p(\mathbf{x}\_i)}- \nabla_{\mathbf{x}} \log {q_{i}({\mathbf{x}\_i})} = \nabla_{\mathbf{x}}  \sigma^{-1}\big(D^{\*}(\mathbf{x})\big)
\end{align}

## Modifying the code of GANs
 Removing the Generator in Generative Adversarial Nets

```diff
data_dim = 2
# loss
criterion = nn.BCEWithLogitsLoss()

# build network
D_logit = Discriminator().to(device)
-G = Generator().to(device)
+xs = make_8gaussians(n_samples=500).to(device)
+xs.requires_grad = True

# optimizer
D_optimizer = optim.Adam(D_logit.parameters(), lr = 0.001)
-G_optimizer = optim.Adam(G.parameters(), lr = 0.001)
+G_optimizer = optim.SGD([xs], lr = 0.001)
```
The first step is to remove the generator `G(z)`, instead we initialize $$500$$ particles `xs` and put them in to the `SGD` optimizer which corresponds to the forward Euler scehme in . One can also use the `G_optimizer = optim.Adam([xs], lr = 0.002)` to incorporate the momentum of gradients.

```diff
#==============Train the discriminator===============#
D_optimizer.zero_grad()
# eval real loss on p
x_real, y_real = x.view(-1, data_dim), torch.ones(batch_size, 1).to(device)
real_loss = criterion(D_logit(x_real), y_real)

# eval fake loss on q_i
-z = torch.randn(batch_size, z_dim).to(device)
-x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)
# randomly select some particles to train the discriminator
+idx = np.random.choice(xs.shape[0], batch_size, replace=False)
+x_fake, y_fake = xs[idx], torch.zeros(batch_size, 1).to(device)
fake_loss = criterion(D_logit(x_fake), y_fake)

# gradient backprop & optimize ONLY D's parameters
D_loss = real_loss + fake_loss
D_loss.backward()
D_optimizer.step()

#==============Update particles===============#
G_optimizer.zero_grad()
-z = torch.randn(batch_size, z_dim).to(device)
-x_fake, y_fake = G(z), torch.ones(batch_size, 1).to(device)
-G_loss = criterion(D_logit(x_fake), y_fake) # non-saturate trick
# update all particles
+logit_xs = D_logit(xs)
# allow for the batch gradient for each particle
+G_loss = -torch.sum(logit_xs)
G_loss.backward()
G_optimizer.step()
```
### Why the KL Divergence?



The marginal distributions $$\{q_{t}\}_{t\geq 0}$$ can be viewed as the Wasserstein gradient flow minimizing the KL divergence. To this end, it seems to be confusing because here comes with both the KL divergence and the Wasserstein distance, which conflicts with the convention in which they are different measurements for the probability discrepancy, such as the GANs and Wasserstein GANs, one minimizes $f$-divergence and another minimizes the Wasserstein distance. But the indeed, they can be good partners to each. Wasserstein gradient flows refer to a curve in a metric space $$(q_t, W_2)$$ where each point is a probability distribution and the distance between any two points is the Wasserstein-2 distance. In this space we can still define the KL divergence $$\mathrm{KL}(q_t\Vert p)$$ as a function but its derivative is obtained under specific Wasserstein calculus rule given by $$\nabla_{W_2} \mathrm{KL}(q_t\Vert p) = \nabla_{\mathbf{x}} [\log (q_t/p)]$$.

remember this ODE decreases the KL divergence along the steepest direction by some Wasserstein dark magic.

But anyway, let's forget the Wasserstein stuffs.  ### Numerical Approaches
We can use forward-Euler method to discretize the probability flow ODE,

The Euler-Maruyama method to the Langevin dynamics gives the well known unadjusted Langevin algorithm (ULA)

The numerical discretization scheme of the probability flow ODE simply replaces the term  $$\sqrt{2\epsilon} \mathcal{N}(0, I)$$ with $$ -\epsilon \nabla_{\mathbf{x}} \log {q_{i}({\mathbf{x}_i})} $$. Althoug via stochastic calculus, the ODE and SDE have the same marginal laws governed by the Fokker-Planck equation, I still have no idea why the schochastic process can be assessed by a deterministic process in the Euclidean space $$\mathbf{x}\in \mathbb{R}^n$$. Just like what Albert Einstein once said, "God does not play dice".