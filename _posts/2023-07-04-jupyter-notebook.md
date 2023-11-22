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
This post provides a simple numerical approach using [PyTorch](https://pytorch.org/) to simulate the probability flow ordinary differential equation (ODE) of the Langevin dynamics. The implementation is super simple, one just needs to slightly modify a code of Generative Adversarial Nets. Such implementation can be understood as ''non-parametric GANs'', which is an alternative view on GAN via the probability flow ODE, more details can be found in my paper ''[MonoFlow: Rethinking Divergence GANs via the Perspective of Wasserstein Gradient Flows](https://arxiv.org/abs/2302.01075)'' , or another excellant paper ''[Unifying GANs and Score-Based Diffusion as Generative Particle Models](https://arxiv.org/abs/2305.16150)'' by [Jean-Yves Franceschi](https://jyfranceschi.fr/). Briefly speaking, GANs can work without generators as a direct particle flow method similar to diffusion models.

<figure style="text-align:center;">
  <img src="{{ site.baseurl }}/assets/img/blog_pic/particles.gif" alt="Prob flow ODE" class="img-fluid rounded z-depth-1" style="width: 40%; height: auto; margin-left: auto; margin-right: auto;">
  <figcaption>Figure 1. Transporting particles via the probability flow ODE.</figcaption>
</figure>

## 2. Langevin dynamics and its probability flow ODE
The Langevin dynamics reads a stochastic differential equation (SDE),
\begin{align}
\mathrm{d} \mathbf{x}\_t = \nabla\_\mathbf{x} \log p(\mathbf{x}\_t)\mathrm{d}t + \sqrt{2} \mathrm{d}\mathbf{w}\_t,
\end{align}
where $$\mathbf{w}_t$$ represents the Brownian motion. Under mild conditions, the marginal $$\sim q_t(\mathbf{x})$$ converges to the stationary distribution $$p\mathbf{x}$$ given a sufficently long time. In order to numerically simulate the Langevin dynamics, we can use the [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) to discretize the SDE, this gives the well-known unadjusted Langevin algorithm (ULA),
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
It is worth noting the vector field of the probability flow ODE is the gradient of the logrithm density ratio $$\log \big[p(\mathbf{x}_i) / q_{i}({\mathbf{x}_i})\big]$$. Next, we will discuss a method to obtain the log density ratio which is analogue to training the discriminator in GANs.



#### Estimating the vector field with the binary classification
Recall that in GANs, we train a discriminator to solve the following binary classification problem.
\begin{align}
\max\_D \quad \mathbb{E}\_{p} \big[\log D(\mathbf{x})\big]+ \mathbb{E}\_{q_i} \big[\log (1-D(\mathbf{x}))\big]
\end{align}
The optimal discrminator is given by (see Proposition 4 in Goodfellow et. al. 2014)
\begin{align}
D^{\*}(\mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x}) + q\_i(\mathbf{x})}.
\end{align}
Since the last layer of the discriminator $$D^{*}(\mathbf{x})$$ is the Sigmoid activation function $$\sigma(\cdot)$$, inversing the Sigmoid activation gives the log density,
\begin{align}
\sigma^{-1}\big(D^{\*}(\mathbf{x})\big)  =\log \frac{p(\mathbf{x}\_i)}{q_{i}({\mathbf{x}\_i})},
\end{align}
$$\sigma^{-1}\big(D^{*}(\mathbf{x})\big)$$ is usually called the logit output of a binary classifier.

This gives us a strategy for sampling via the probability flow ODE which is similar to training GANs, 
- Training the discriminator with a few steps of gradient update.
- Update particles using Eq. ()

### 3. Modifying the code of GANs
- The first step is to remove the generator `G(z)`, instead we initialize $$500$$ particles `xs` and put them into the `SGD` optimizer which corresponds to the forward Euler discretization of the ODE. One can also use `optim.Adam([xs], lr = 0.002)` to incorporate the momentum of gradients. Note that we use the loss function criterion `nn.BCEWithLogitsLoss()`, this criterion directly works with the logit output of a binary classifer.

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

- The next step is to modify the training procedure of GANs. There is no difference on training the discriminator, we just replace fake samples with a minibatch of `xs`. The `G_loss` is now changed to `-torch.sum(D_logit(xs))` with the purpose of computing the per particle gradient. This is because there is no explicit way to perform batch gradient operation in PyTorch, we can instead sum all per particle forward pass together to generate a scalar and backpropogating this scalar would give each particle its own gradient.

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
# allow for the batch gradient for each particle
+G_loss = -torch.sum(D_logit(xs))
G_loss.backward()
G_optimizer.step()
```
So, that's it! We have deleted 7 lines and added 6 lines. Now we can run the code to simulate the probability flow ODE.  The complete Jupyter notebook for reproducing the experiment can be found here.

<div class="row mt-3">
<div class="col-sm mt-3 mt-md-0">
  <img src="{{ site.baseurl }}/assets/img/blog_pic/s_sgd.gif" class="img-fluid rounded z-depth-1" style="width: 90%; height: auto;">
  <figcaption style="text-align: center; margin-top: 10px;"> Using the SGD optimizer.</figcaption>
</div>
<div class="col-sm mt-3 mt-md-0">
<img src="{{ site.baseurl }}/assets/img/blog_pic/s_adam.gif" class="img-fluid rounded z-depth-1" style="width: 90%; height: auto;">
  <figcaption style="text-align: center; margin-top: 10px;"> Using the Adam optimizer. </figcaption>
</div>
<div>


