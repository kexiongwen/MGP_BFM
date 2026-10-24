# Factor Switching

Rotation, sign, and permutation post-processing of factor loadings.

- [1. Factor Model](#1-factor-model)
- [2. Identifiability Problems](#2-identifiability-problems)
- [3. Method](#3-method)
- [4. Computational Strategies](#4-computational-strategies)
- [References](#references)

---

## 1. Factor Model

Notation and model setting used throughout.

### 1.1 Model setting

Let $`Y_i=(Y_{i1},\ldots,Y_{ip})^\top\in\mathbb{R}^p`$ denote the $`i`$-th observation in a random sample of $`p`$-dimensional responses, for $`i=1,\ldots,n`$. Let $`\mathcal{N}_p(\mu,\boldsymbol{\Sigma})`$ denote the $`p`$-dimensional normal distribution with mean vector $`\mu=(\mu_1,\ldots,\mu_p)^\top\in\mathbb{R}^p`$ and covariance matrix $`\boldsymbol{\Sigma}`$, and let $`\mathbf{I}_p`$ denote the $`p\times p`$ identity matrix.

In the standard Factor Analysis (FA) model — also known as the *strict factor structure* — each response $`Y_i`$ is modeled as a linear combination of a latent factor vector $`F_i\in\mathbb{R}^q`$ plus idiosyncratic noise:

```math
Y_i = \mu + \boldsymbol{\Lambda} F_i + \varepsilon_i,\quad i = 1,\ldots,n,
\qquad\text{(1)}
```

where the number of factors $`q`$ is fixed and known, with $`0<q<p`$. Here, $`\boldsymbol{\Lambda}=(\lambda_{rj})\in\mathbb{R}^{p\times q}`$ is the matrix of factor loadings, and $`\mu=(\mu_1,\ldots,\mu_p)^\top`$ is the marginal mean of $`Y_i`$. The unobserved factor vectors $`F_i=(F_{i1},\ldots,F_{iq})^\top`$ lie in a lower-dimensional space ($`q<p`$) and consist of uncorrelated, unit-variance features:

```math
F_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}_q(0_q,\mathbf{I}_q),\quad i=1,\ldots,n,
\qquad\text{(2)}
```

where $`0_q:=(0,\ldots,0)^\top\in\mathbb{R}^q`$.

The error terms $`\varepsilon_i`$ (commonly referred to as *uniquenesses* or *idiosyncratic errors*) are independent of $`F_i`$, so $`\mathrm{Cov}(F_{ij},\varepsilon_{ir})=0`$ for all $`j=1,\ldots,q`$ and $`r=1,\ldots,p`$. They follow a normal distribution:

```math
\varepsilon_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}_p(0_p,\boldsymbol{\Sigma}),\quad i=1,\ldots,n,
\qquad\text{(3)}
```

with mutually independent components $`\varepsilon_{i1},\ldots,\varepsilon_{ip}`$, so that

```math
\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2,\ldots,\sigma_p^2).
\qquad\text{(4)}
```

Conditional on $`F_i`$, the responses are independent across observations with distribution

```math
Y_i \mid F_i \sim \mathcal{N}_p(\mu+\boldsymbol{\Lambda} F_i,\boldsymbol{\Sigma}),
\quad \text{independent for } i=1,\ldots,n,
\qquad\text{(5)}
```

and marginalizing out $`F_i`$ yields the unconditional distribution

```math
Y_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}_p(\mu,\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}),
\quad i=1,\ldots,n.
\qquad\text{(6)}
```

Without loss of generality, one may assume $`\mu=0_p`$, though this is not required in what follows.

### 1.2 Assumptions at a glance

1. **Latent linear structure (1).** Each $`p`$-dimensional response is a linear function of $`q<p`$ latent factors plus additive error, $`Y_i=\mu+\boldsymbol{\Lambda}F_i+\varepsilon_i`$, with $`q`$ fixed and known.
2. **Factor distribution (2).** $`F_i\stackrel{\mathrm{iid}}{\sim}\mathcal{N}_q(0_q,\mathbf{I}_q)`$: latent factors are uncorrelated, homoscedastic, and standardized (zero mean, unit variance).
3. **Error distribution (3).** $`\varepsilon_i\stackrel{\mathrm{iid}}{\sim}\mathcal{N}_p(0_p,\boldsymbol{\Sigma})`$, independent of the latent factors.
4. **Diagonal error covariance (4).** $`\boldsymbol{\Sigma}=\mathrm{diag}(\sigma_1^2,\ldots,\sigma_p^2)`$: idiosyncratic errors are mutually independent, which defines a *strict* factor structure.
5. **Conditional law (5).** Given $`F_i`$, the observations $`Y_i`$ are independent and normally distributed with mean $`\mu+\boldsymbol{\Lambda}F_i`$ and covariance $`\boldsymbol{\Sigma}`$.
6. **Marginal law (6).** Integrating out $`F_i`$ gives $`Y_i\stackrel{\mathrm{iid}}{\sim}\mathcal{N}_p(\mu,\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma})`$, so $`\mathrm{Cov}(Y_i)=\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}`$. Thus, the latent factors account for all dependence among the manifest variables — the defining feature of factor analysis.

---

## 2. Identifiability Problems

The model of Section 1 is subject to two distinct identifiability issues: one concerning the error covariance $`\boldsymbol{\Sigma}`$ (the *uniqueness problem*, Section 2.1) and one concerning the factor loadings $`\boldsymbol{\Lambda}`$ (*orthogonal invariance*, Section 2.2).

### 2.1 Uniqueness problem: identifiability of $`\boldsymbol{\Sigma}`$

With $`q`$ factors, the structured covariance matrix $`\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}`$ contains $`p+pq-\tfrac{1}{2}q(q-1)`$ free parameters (accounting for rotational symmetry), whereas an unconstrained $`p\times p`$ covariance matrix has $`\tfrac{1}{2}p(p+1)`$ free parameters. Imposing a strict factor structure therefore reduces the number of free covariance parameters by

```math
\frac{1}{2}p(p+1)-\left[p+pq-\frac{1}{2}q(q-1)\right]
=\frac{1}{2}\left[(p-q)^2-(p+q)\right].
\qquad\text{(7)}
```

This reduction is strictly positive when $`q<\phi(p)`$, where

```math
\phi(p):=\frac{2p+1-\sqrt{8p+1}}{2}
\qquad\text{(8)}
```

is the *Ledermann bound*. Under standard genericity and rank conditions, the strict inequality $`q<\phi(p)`$ is sufficient for $`\boldsymbol{\Sigma}`$ to be almost surely unique. Throughout, we assume $`q\le\phi(p)`$; the boundary case $`q=\phi(p)`$ is not covered by this strict-inequality sufficient condition.

**Isotropic error model.** Identifiability need not fail when $`q>\phi(p)`$ if additional structure is imposed. In particular, assuming isotropic errors $`\boldsymbol{\Sigma}=\sigma^2\mathbf{I}_p`$ in Equation (4) yields the probabilistic principal component analysis (PPCA) model, in which $`q`$ can be as large as $`p-1`$.

**Heywood cases.** A practical complication in estimating $`\boldsymbol{\Sigma}`$ is the occurrence of *Heywood cases*. Unconstrained likelihood maximization can produce negative estimates for one or more idiosyncratic variances $`\sigma_r^2`$, while optimization constrained to $`\sigma_r^2\ge 0`$ can hit the boundary at zero. A zero residual variance implies that the corresponding manifest variable is completely explained by the latent factors.

In a Bayesian setting, an inverse-gamma prior with positive shape and scale parameters is supported on $`(0,\infty)`$, so a proper posterior automatically excludes nonpositive variances. Even so, the posterior may place substantial mass — or exhibit a mode — near zero. Tuning the inverse-gamma hyperparameters can suppress posterior mass near zero, but because the prior density remains strictly positive on all of $`(0,\infty)`$, it cannot enforce a hard lower bound. If a strict bound $`\sigma_r^2\geq\delta>0`$ is required, it must be imposed via prior truncation or an alternative prior supported on $`[\delta,\infty)`$.

### 2.2 Orthogonal invariance: identifiability of $`\boldsymbol{\Lambda}`$

Even when $`\boldsymbol{\Sigma}`$ is identifiable, the factor loadings $`\boldsymbol{\Lambda}`$ remain non-identifiable under orthogonal transformations — the central problem addressed in this document. A square matrix $`\mathbf{O}\in\mathbb{R}^{q\times q}`$ is *orthogonal* if and only if $`\mathbf{O}^{\top}=\mathbf{O}^{-1}`$, which implies $`\det(\mathbf{O})\in\{-1,1\}`$. Geometrically, orthogonal matrices represent proper rotations ($`\det(\mathbf{O})=1`$) or improper rotations / reflections ($`\det(\mathbf{O})=-1`$).

For any $`q\times q`$ orthogonal matrix $`\mathbf{O}`$, define the rotated factors $`\widetilde{F}_i=\mathbf{O}F_i\sim\mathcal{N}_q(0_q,\mathbf{I}_q)`$. The transformed model

```math
Y_i=\mu+\boldsymbol{\Lambda}\mathbf{O}^{\top}\widetilde{F}_i+\varepsilon_i
\qquad\text{(9)}
```

induces the exact same marginal distribution for $`Y_i`$ as Equation (6) because $`(\boldsymbol{\Lambda}\mathbf{O}^\top)(\boldsymbol{\Lambda}\mathbf{O}^\top)^\top=\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top`$. The marginal likelihood is therefore invariant under $`\boldsymbol{\Lambda}\mapsto\boldsymbol{\Lambda}\mathbf O^\top`$.

A proper posterior inherits this invariance whenever the prior on $`\boldsymbol{\Lambda}`$ (with other parameters held fixed) is itself orthogonally invariant; when latent factors are sampled jointly, the invariance acts via $`(\boldsymbol{\Lambda},F_i)\mapsto(\boldsymbol{\Lambda}\mathbf O^\top,\mathbf O F_i)`$. For $`q\geq2`$, this continuous symmetry generates orbits of observationally equivalent parameter values, so any posterior maximum on a nontrivial orbit is a ridge rather than an isolated mode. Using a non-invariant prior can break the symmetry of the posterior, but it does not alter the rotational invariance of the likelihood.

**Unequal factor variances.** The orthogonal invariance above stems from assumption (2), which sets $`\mathrm{Cov}(F_i)=\mathbf{I}_q`$. If instead $`F_i\sim\mathcal{N}_q(0_q,\boldsymbol{\Psi})`$ for a fixed positive definite diagonal matrix $`\boldsymbol{\Psi}`$ with pairwise distinct diagonal entries, the marginal covariance becomes $`\boldsymbol{\Lambda}\boldsymbol{\Psi}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}`$. The only orthogonal matrices satisfying $`\mathbf O\boldsymbol{\Psi}\mathbf O^\top=\boldsymbol{\Psi}`$ are diagonal sign matrices. Thus, fixing distinct factor variances eliminates continuous orthogonal symmetries of the factor distribution, leaving only sign ambiguity (and simultaneous column-and-variance permutations unless the diagonal of $`\boldsymbol{\Psi}`$ is ordered).

However, fixing unequal factor variances does *not* identify unrestricted loadings. For any $`q\times q`$ orthogonal matrix $`\mathbf U`$, the generally nonorthogonal matrix

```math
\mathbf B=\boldsymbol{\Psi}^{1/2}\mathbf U\boldsymbol{\Psi}^{-1/2}
\quad\text{satisfies}\quad
\mathbf B\boldsymbol{\Psi}\mathbf B^\top=\boldsymbol{\Psi}.
```

Replacing $`(\boldsymbol{\Lambda},F_i)`$ with $`(\boldsymbol{\Lambda}\mathbf B,\mathbf B^{-1}F_i)`$ leaves the marginal covariance $`\boldsymbol{\Lambda}\boldsymbol{\Psi}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}`$ unchanged. For $`q\geq2`$, this still forms a continuous family of equivalent parameterizations; only for $`q=1`$ does it reduce to sign changes.

**Post-processing perspective.** For the standardized-factor model (2), the approach taken here draws unconstrained loadings during MCMC sampling and resolves their orientation *ex post* by post-processing the posterior sample. This selects a coherent representation for posterior summaries without imposing identifying constraints on the underlying statistical model. All orthogonal Rotation-Sign-Permutation (RSP) transformations below are formulated for the standardized model (2), rather than for general unequal $`\boldsymbol{\Psi}`$.

---

## 3. Method

### 3.1 Notation and definitions

Let $`\mathcal{T}_q`$ denote the symmetric group of all $`q!`$ permutations of $`\{1,\ldots,q\}`$. Under the *column-mapping convention* of this subsection, a permutation $`\nu=(\nu_1,\ldots,\nu_q)\in\mathcal{T}_q`$ sends the $`j`$-th column of $`\boldsymbol{\Lambda}`$ to the $`\nu_j`$-th column of the permuted matrix $`\dot{\boldsymbol{\Lambda}}`$, for $`j=1,\ldots,q`$. Equivalently, $`\dot{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{P}`$, where $`\mathbf{P}`$ is the $`q\times q`$ *permutation matrix* with entries

```math
(\mathbf{P})_{ij}=
\begin{cases}
1, & \text{if } \nu_i=j,\\
0, & \text{otherwise},
\end{cases}
\quad i,j=1,\ldots,q.
```

Every permutation matrix is orthogonal. For example, when $`q=3`$ and $`\nu=(3,1,2)\in\mathcal{T}_3`$, columns 1, 2, and 3 of $`\boldsymbol{\Lambda}`$ move to positions 3, 1, and 2, respectively:

```math
\mathbf{P}=
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix},
\qquad
\dot{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{P}=
\begin{pmatrix}
\lambda_{11} & \lambda_{12} & \lambda_{13}\\
\vdots & \vdots & \vdots \\
\lambda_{p1} & \lambda_{p2} & \lambda_{p3}
\end{pmatrix}
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
\lambda_{12} & \lambda_{13} & \lambda_{11}\\
\vdots & \vdots & \vdots \\
\lambda_{p2} & \lambda_{p3} & \lambda_{p1}
\end{pmatrix}.
```

A *signed permutation matrix* is a square matrix with exactly one nonzero entry in each row and each column, that entry being $`+1`$ or $`-1`$. Under the column-mapping convention, any $`q\times q`$ signed permutation matrix $`\mathbf{Q}`$ factors as

```math
\mathbf{Q}=\mathbf{S}\mathbf{P},
\qquad\text{(10)}
```

where $`\mathbf{S}=\mathrm{diag}(s_1,\ldots,s_q)`$ is a diagonal *sign matrix* with $`s_j\in\{-1,1\}`$ and $`\mathbf{P}`$ is a permutation matrix. (The assignment convention introduced in Section 3.2.2 uses the transpose factorization $`\mathbf{Q}=\mathbf{P}^\top\mathbf{S}`$.) For instance, flipping the signs of the first two columns and then applying $`\nu=(3,1,2)`$ gives

```math
\mathbf{Q}=\mathbf{S}\mathbf{P}=
\begin{pmatrix}
-1 & 0 & 0\\
0 & -1 & 0\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
0 & 0 & -1\\
-1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix},
```

which maps $`\boldsymbol{\Lambda}`$ to the sign-permuted matrix

```math
\mathring{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{Q}
=\boldsymbol{\Lambda}\mathbf{S}\mathbf{P}=
\begin{pmatrix}
-\lambda_{12} & \lambda_{13} & -\lambda_{11}\\
\vdots & \vdots & \vdots \\
-\lambda_{p2} & \lambda_{p3} & -\lambda_{p1}
\end{pmatrix}.
```

The table below summarizes the notation used for loading matrices, transformations, and optimization quantities throughout Section 3:

| Symbol | Meaning |
|---|---|
| $`\widetilde{\boldsymbol{\Lambda}}`$ | Varimax-rotated loading matrix |
| $`\dot{\boldsymbol{\Lambda}}`$ | Column-permuted loading matrix |
| $`\mathring{\boldsymbol{\Lambda}}`$ | Sign-permuted (aligned) loading matrix |
| $`\boldsymbol{\Lambda}^\star`$ | Reference matrix of factor loadings |
| $`\mathbf{O}`$ | Orthogonal matrix; $`\mathbf{O}^{(t)}`$ is the varimax rotation for draw $`t`$ |
| $`\mathbf{P}`$ | Column-mapping permutation matrix of Section 3.1 |
| $`\mathbf{S}`$ | Diagonal sign matrix, $`\mathrm{diag}(s_1,\ldots,s_q)`$ with $`s_j\in\{-1,1\}`$ |
| $`\mathbf{Q}`$ | Signed permutation matrix: $`\mathbf Q=\mathbf S\mathbf P`$ (column mapping) or $`\mathbf Q=\mathbf P^\top\mathbf S`$ (assignment) |
| $`\mathbf{R}^{(t)}`$ | Total orthogonal transform for draw $`t`$, $`\mathbf R^{(t)}=\mathbf O^{(t)}\mathbf Q^{(t)}`$ |
| $`\mathcal{T}_q`$ | Set of all $`q!`$ permutations of $`\{1,\ldots,q\}`$ |
| $`\mathcal{Q}_q`$ | Set of all $`q\times q`$ signed permutation matrices |
| $`s`$ | Sign vector $`(s_1,\ldots,s_q)\in\{-1,1\}^q`$ |
| $`\nu`$ | Permutation vector $`(\nu_1,\ldots,\nu_q)\in\mathcal{T}_q`$ (read as an assignment in Section 3.2; see note below) |
| $`\Psi`$ | Global objective function of the signed-permutation stage |
| $`\mathcal{L}^{(t)}_{s,\nu}`$ | Per-draw squared Frobenius loss in the signed-permutation stage |

> **Assignment convention note:** Starting in Section 3.2.2, $`\nu`$ is read as an *assignment*: reference column $`j`$ is matched to column $`\nu_j`$ of the rotated draw.

### 3.2 Rotation-Sign-Permutation (RSP) post-processing algorithm

The RSP algorithm aligns an MCMC sample of factor loadings $`\{\boldsymbol{\Lambda}^{(t)}\}_{t=1}^T`$ in two stages:

1. **Per-draw varimax rotation:** each draw is rotated independently to a canonical sparse orientation.
2. **Signed-permutation alignment:** all rotated draws are aligned jointly to a common reference matrix via alternating minimization over signed permutations and the reference.

These post-processing stages substantially reduce rotational and label switching, though they do not guarantee a unique global orientation or complete removal of rotational discrepancies.

#### 3.2.1 Varimax rotation

Given a $`p\times q`$ loading matrix $`\boldsymbol{\Lambda}`$, the varimax criterion seeks a $`q\times q`$ orthogonal matrix $`\boldsymbol{\Phi}`$ that maximizes the sum of within-column variances of the squared rotated loadings $`\widetilde{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\boldsymbol{\Phi}`$:

```math
\begin{aligned}
\text{maximize}\quad & \frac{1}{4}\sum_{j=1}^{q}\left[\left(\sum_{r=1}^{p}\widetilde\lambda^4_{rj}\right)-\frac{1}{p}\left(\sum_{r=1}^{p}\widetilde\lambda^2_{rj}\right)^2\right]\\
\text{subject to}\quad & \boldsymbol{\Phi}^\top \boldsymbol{\Phi} = \mathbf{I}_q,\\
\text{where}\quad & \widetilde\lambda_{rj}=\sum_{k=1}^{q}\lambda_{rk}\phi_{kj},\quad r=1,\ldots,p;\ j=1,\ldots,q.
\end{aligned}
\qquad\text{(11)}
```

Classical varimax algorithms maximize (11) by cyclically rotating pairs of columns, whereas modern matrix formulations update all $`q`$ columns simultaneously via polar / singular value decompositions.

For each MCMC draw $`\boldsymbol{\Lambda}^{(t)}=(\lambda_{rj}^{(t)})`$, $`t=1,\ldots,T`$, a single-start matrix-form varimax iteration is run independently with relative stagnation parameter $`\varepsilon=10^{-5}`$ (see Section 4.1), yielding the rotated draw $`\widetilde{\boldsymbol{\Lambda}}^{(t)}=(\widetilde\lambda_{rj}^{(t)})`$. Note that $`\varepsilon`$ governs relative stagnation of the step size rather than optimality accuracy for (11).

If two draws differ purely by an orthogonal transformation, varimax rotation reduces their discrepancy to a signed permutation *provided* that (i) both runs attain a global maximum of (11), and (ii) the global maximizer is unique up to signed permutations. Neither condition holds universally: distinct local maxima or non-equivalent global maxima can leave residual rotational differences that signed permutations cannot remove. Furthermore, posterior draws also exhibit genuine sampling variability beyond orthogonal rotation, so exact equality across aligned draws is not expected.

#### 3.2.2 Signed permutations towards a reference

In the second stage, each varimax-rotated draw $`\widetilde{\boldsymbol{\Lambda}}^{(t)}`$ is matched to a reference loading matrix $`\boldsymbol{\Lambda}^{\star}`$ by minimizing the squared Frobenius distance over the discrete set $`\mathcal{Q}_q`$ of $`q\times q`$ signed permutation matrices. Treating $`\boldsymbol{\Lambda}^{\star}`$ as fixed for the moment (relaxed in Section 3.2.3), the problem is

```math
\begin{aligned}
\text{minimize}\quad & \sum_{t=1}^{T}\left\|\widetilde{\boldsymbol{\Lambda}}^{(t)}\mathbf{Q}^{(t)}-\boldsymbol{\Lambda}^{\star}\right\|_F^2\\
\text{subject to}\quad & \mathbf{Q}^{(t)}\in\mathcal{Q}_q,\quad t=1,\ldots,T,
\end{aligned}
\qquad\text{(12)}
```

where $`\|\mathbf{A}\|_F=\sqrt{\sum_i\sum_j a_{ij}^2}`$ denotes the Frobenius norm. (If $`\mathcal{Q}_q`$ were relaxed to the full orthogonal group $`\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}_q`$, (12) would reduce to the classical Orthogonal Procrustes problem, solved analytically by $`\mathbf{Q}^{(t)}=\mathbf{V}\mathbf{U}^\top`$ from the SVD $`\boldsymbol{\Lambda}^{\star\top}\widetilde{\boldsymbol{\Lambda}}^{(t)}=\mathbf{U}\mathbf{D}\mathbf{V}^\top`$, allowing improper rotations with $`\det(\mathbf{Q}^{(t)})=-1`$. Here, however, we restrict $`\mathbf{Q}^{(t)}`$ to the discrete group $`\mathcal{Q}_q`$.)

Under the *assignment parameterization*, a signed permutation is represented by a sign vector $`s=(s_1,\ldots,s_q)\in\{-1,1\}^q`$ and a permutation vector $`\nu=(\nu_1,\ldots,\nu_q)\in\mathcal{T}_q`$: reference column $`j`$ is matched to draw column $`\nu_j`$ multiplied by $`s_j`$. The corresponding matrix $`\mathbf Q\in\mathcal{Q}_q`$ has entries $`(\mathbf Q)_{\nu_j,j}=s_j`$ and zeros elsewhere. In terms of the column-mapping matrices $`\mathbf P`$ and $`\mathbf S`$ of Section 3.1 constructed from the same $`(s,\nu)`$, we have $`\mathbf Q=\mathbf P^\top\mathbf S=(\mathbf S\mathbf P)^\top`$; the multiplication order matters because diagonal sign matrices and permutation matrices do not commute in general. Under this parameterization,

```math
\left\|\widetilde{\boldsymbol{\Lambda}}\mathbf{Q}-\boldsymbol{\Lambda}^{\star}\right\|_F^2
=\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j\widetilde\lambda_{r\nu_j}-\lambda^{\star}_{rj}\right)^2,
```

so problem (12) takes the coordinate form

```math
\begin{aligned}
\min_{\boldsymbol{\Lambda}^\star,\,s,\,\nu}\ \Psi(\boldsymbol{\Lambda}^{\star},s,\nu)
&=\sum_{t=1}^{T}\mathcal{L}_{s^{(t)},\nu^{(t)}}^{(t)}
=\sum_{t=1}^{T}\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j^{(t)}\widetilde\lambda_{r\nu_j^{(t)}}^{(t)}-\lambda^{\star}_{rj}\right)^2\\
\text{subject to}\quad & s_j^{(t)}\in\{-1,1\},\quad t=1,\ldots,T;\ j=1,\ldots,q,\\
& \nu^{(t)}\in\mathcal{T}_q,\quad t=1,\ldots,T,
\end{aligned}
\qquad\text{(13)}
```

with per-draw loss

```math
\mathcal{L}_{s,\nu}^{(t)}:=\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j\widetilde\lambda_{r\nu_j}^{(t)}-\lambda^{\star}_{rj}\right)^2.
\qquad\text{(14)}
```

**Index convention.** The assignment convention above is used throughout the remainder of the document and is the inverse of the column-mapping convention of Section 3.1. Both conventions parameterize the same set $`\mathcal{Q}_q`$, so the minimum value of $`\Psi`$ and the aligned loading matrices $`\mathring{\boldsymbol{\Lambda}}^{(t)}`$ are identical; only the correspondence between a stored pair $`(s,\nu)`$ and the matrix $`\mathbf{Q}`$ changes.

Because the rotated sample $`\{\widetilde{\boldsymbol{\Lambda}}^{(t)}\}_{t=1}^{T}`$ is fixed during Stage 2, its dependence is suppressed in the notation $`\Psi(\boldsymbol{\Lambda}^{\star},s,\nu)`$. In practice, the reference matrix $`\boldsymbol{\Lambda}^{\star}=(\lambda^{\star}_{rj})`$ is unknown and is estimated jointly with $`(s,\nu)`$ by alternating minimization, analogous to relabeling algorithms for label switching in Bayesian mixture models.

#### 3.2.3 The recursive algorithm

The RSP algorithm minimizes $`\Psi(\boldsymbol{\Lambda}^{\star},s,\nu)`$ by alternating two coordinate-descent updates:

- **Sign-Permutation (SP) step:** minimize $`\Psi`$ with respect to $`\{(s^{(t)},\nu^{(t)})\}_{t=1}^T`$ for a fixed reference matrix $`\boldsymbol{\Lambda}^{\star}`$.
- **Reference-Loading Matrix Estimation (RLME) step:** minimize $`\Psi`$ with respect to $`\boldsymbol{\Lambda}^{\star}`$ for fixed signed permutations.

Although both coordinate subproblems are solved globally and in closed form / polynomial time, the alternating minimization is nonconvex over the joint space and is only guaranteed to converge to a coordinate-wise stationary point rather than a global minimum of (13).

---

**Algorithm 1: Varimax-RSP Post-Processing (Current Implementation)**

- **Input:** MCMC sample of factor loading matrices $`\{\boldsymbol{\Lambda}^{(t)}\}_{t=1}^T`$.
- **Output:** Aligned loading sample $`\{\mathring{\boldsymbol{\Lambda}}^{(t)}\}_{t=1}^T`$, final reference matrix $`\mathbf M`$, signed-permutation indices, optional orthogonal transforms, and convergence diagnostics.

1. **Step 1 — Per-draw varimax rotation.** For each $`t=1,\ldots,T`$, apply the matrix-form varimax iteration of Section 4.1 to $`\boldsymbol{\Lambda}^{(t)}`$ to obtain the rotated loadings $`\widetilde{\boldsymbol{\Lambda}}^{(t)}`$ and rotation matrix $`\mathbf{O}^{(t)}`$ (global optimality for (11) is not certified).
2. **Step 2 — Reference initialization.** If a user-supplied reference matrix is provided, set $`\mathbf M_0`$ to it. Otherwise, initialize $`\mathbf M_0`$ according to the selected mode:
   - **`mean`:** set $`\mathbf M_0=T^{-1}\sum_{t=1}^{T}\widetilde{\boldsymbol{\Lambda}}^{(t)}`$. This matches the original R implementation and is equivalent to initializing all signed permutations to the identity followed by one RLME update.
   - **`auto` (default) or `draw`:** select the highest-norm draw $`t_0=\mathop{\mathrm{argmax}}_t\|\widetilde{\boldsymbol{\Lambda}}^{(t)}\|_F`$. Using $`\widetilde{\boldsymbol{\Lambda}}^{(t_0)}`$ as a temporary target, perform one full SP–RLME sweep across all draws and set $`\mathbf M_0`$ to the resulting aligned sample mean (not the single draw $`\widetilde{\boldsymbol{\Lambda}}^{(t_0)}`$ itself). In `auto` mode, a near-zero mean-cancellation diagnostic is also evaluated for verbose logging; this diagnostic is purely informational and never alters the initialization path. If all draws are identically zero, the mean reference is used.
3. **Step 3 — Alternating SP–RLME sweeps.** For outer iteration $`k=1,\ldots,K`$ (default $`K=\texttt{max\_iter}=100`$):
   - **3.1 SP step.** Given the current reference $`\mathbf M_{k-1}`$, solve the per-draw problem independently for each $`t=1,\ldots,T`$:
     ```math
     \left(s_k^{(t)},\nu_k^{(t)}\right)=\mathop{\mathrm{argmin}}_{s\in\{-1,1\}^q,\,\nu\in\mathcal{T}_q}\mathcal{L}_{s,\nu}^{(t)}(\mathbf M_{k-1}).
     \qquad\text{(15)}
     ```
     Form the aligned draws $`\mathring{\boldsymbol{\Lambda}}_k^{(t)}`$ and evaluate the pre-update objective $`f_k^{\mathrm{pre}}=\sum_{t=1}^{T}\|\mathring{\boldsymbol{\Lambda}}_k^{(t)}-\mathbf M_{k-1}\|_F^2`$.
   - **3.2 RLME step.** Update the reference matrix to the mean of the newly aligned draws and compute the post-update objective:
     ```math
     \mathbf M_k=\frac{1}{T}\sum_{t=1}^{T}\mathring{\boldsymbol{\Lambda}}_k^{(t)},
     \qquad
     f_k^{\mathrm{post}}=\sum_{t=1}^{T}\|\mathring{\boldsymbol{\Lambda}}_k^{(t)}-\mathbf M_k\|_F^2.
     ```
     Entry wise, $`\lambda_{k,rj}=T^{-1}\sum_{t=1}^T s_{k,j}^{(t)}\widetilde\lambda_{r\nu_{k,j}^{(t)}}^{(t)}`$, which is the unique global minimizer of $`\Psi`$ with respect to the reference for fixed signed permutations.
   - **3.3 Stopping rules and status codes.** Note that the RLME update is always executed before termination on the final sweep. Let $`\theta=\texttt{threshold}`$ (default $`10^{-6}`$), let $`\varepsilon_{\mathrm{dtype}}`$ denote the working-precision machine epsilon, and define the numerical tolerances
     ```math
     \rho=\max\left\{10^{-9},\,10\varepsilon_{\mathrm{dtype}}\sqrt{Tpq}\right\},
     \qquad
     \eta=100\varepsilon_{\mathrm{dtype}}^2 Tpq\left(\max_{t,r,j}|\lambda_{rj}^{(t)}|\right)^2,
     ```
     where $`\eta`$ uses the scale of the raw input loadings as a heuristic anomaly threshold (rather than a strict rounding-error bound):
     - **Status 4 (`nonfinite`):** At every sweep, if $`f_k^{\mathrm{pre}}`$, $`f_k^{\mathrm{post}}`$, $`\mathring{\boldsymbol{\Lambda}}_k`$, or $`\mathbf M_k`$ contains nonfinite values, terminate immediately with `status_code = 4` (taking precedence over all rules below). Final returned arrays and requested transforms are checked as well, with affected fields listed in `diagnostics.nonfinite_fields`. (A nonfinite `loss_initial` prior to alignment does not by itself trigger Status 4 if subsequent aligned quantities are finite.)
     - **Status 3 (`anomaly`):** Otherwise, for $`k\geq2`$, if the pre-update loss increases beyond tolerance, $`f_k^{\mathrm{pre}}-f_{k-1}^{\mathrm{pre}}>\max\{\rho|f_{k-1}^{\mathrm{pre}}|,\eta\}`$, terminate with `status_code = 3`.
     - **Status 1 (`converged`):** Otherwise, for $`k\geq2`$, if the pre-update improvement satisfies $`f_{k-1}^{\mathrm{pre}}-f_k^{\mathrm{pre}}<\theta Tpq`$, terminate with `status_code = 1`.
     - **Status 2 (`max_iter`):** If no stopping rule triggers by sweep $`K`$, terminate after sweep $`K`$ with `status_code = 2`.

---

The preliminary sweep used during `auto` / `draw` initialization in Step 2 is not counted in `n_iter` and is excluded from the recorded loss histories. Likewise, `converged` and `status_code` reflect only the outer SP–RLME loop; they do not certify optimality of the Step 1 varimax rotations or global optimality of (13).

**Floating-point safeguards.** To prevent overflow and underflow across wide dynamic ranges, residuals, sample means, and Frobenius norms are evaluated using binary exponent (`frexp`/`ldexp`) scaling. In the SP step, each pairwise cost retains its own exponent while preserving exact relative assignment weights; when extreme dynamic ranges occur within a draw, additional assignment solves with a feasible-cost cap are used. While these safeguards eliminate documented overflow and cancellation regressions, they do not make floating-point evaluation identical to exact rational arithmetic, and losses exceeding the output data type's representable range may still overflow or underflow to zero.

**Returned fields and indexing conventions.** The implementation returns a structured result with the following fields:

| Field | Description |
|---|---|
| `aligned_loadings` | Final aligned loading draws $`\mathring{\boldsymbol{\Lambda}}^{(t)}`$ |
| `mean_loadings` | Final reference loading matrix $`\mathbf M`$ |
| `signs`, `permutations` | Final signs $`s_j^{(t)}\in\{-1,1\}`$ and zero-based assignment indices (adding $`1`$ yields the $`1`$-based mathematical index $`\nu_j^{(t)}`$) |
| `loss_initial` | Initial unaligned loss $`\sum_{t=1}^T\|\widetilde{\boldsymbol{\Lambda}}^{(t)}-\mathbf M_0\|_F^2`$ |
| `loss_final` | Final post-update loss $`f^{\mathrm{post}}`$ |
| `loss_history`, `loss_history_post` | Arrays of pre- and post-update losses $`f_k^{\mathrm{pre}}`$ and $`f_k^{\mathrm{post}}`$ (first `n_iter` entries are populated) |
| `n_iter`, `converged`, `status_code` | Outer SP–RLME sweep count and termination diagnostics |
| `varimax_n_iter` | Per-draw `int32` iteration counts from Step 1 (`None` if varimax was skipped; a value of `1000` indicates the iteration limit was reached; a nonzero draw whose gradient is identically zero exits after its first update with count `1`) |
| `transforms` | Optional total orthogonal transforms $`\mathbf R^{(t)}`$ (included when `return_transforms=True`) |
| `diagnostics`, `metadata` | Numerical diagnostics and caller-supplied MCMC metadata (the latter passed through `metadata=` to `rsp_exact`; hashable scalars/sequences only). Diagnostics record the requested and the resolved assignment backend (`assignment_backend_requested` / `assignment_backend`), the requested `polar` path and `varimax_svd_fallbacks` |

**Co-transforming latent factor scores.** Let $`\mathbf{O}^{(t)}`$ denote the Step 1 varimax rotation matrix for draw $`t`$ (denoted $`\boldsymbol{\Phi}`$ in (11)), and let $`\mathbf Q^{(t)}`$ denote the Step 3 assignment signed-permutation matrix with entries $`(\mathbf Q^{(t)})_{\nu_j^{(t)},j}=s_j^{(t)}`$ (equivalently, $`\mathbf Q^{(t)}=(\mathbf P^{(t)})^\top\mathbf S^{(t)}`$ in terms of the column-mapping matrices of Section 3.1). Define the total orthogonal transform $`\mathbf R^{(t)}=\mathbf O^{(t)}\mathbf Q^{(t)}`$. The aligned loadings and consistently transformed latent factor vectors are given by

```math
\mathring{\boldsymbol{\Lambda}}^{(t)}
=\boldsymbol{\Lambda}^{(t)}\mathbf R^{(t)}
=\boldsymbol{\Lambda}^{(t)}\mathbf O^{(t)}\mathbf Q^{(t)},\quad t=1,\ldots,T,
\qquad\text{(16)}
```

```math
\mathring{F}_i^{(t)}
=(\mathbf R^{(t)})^\top F_i^{(t)}
=(\mathbf Q^{(t)})^\top(\mathbf O^{(t)})^\top F_i^{(t)},
\quad i=1,\ldots,n;\ t=1,\ldots,T.
\qquad\text{(17)}
```

When factor scores are stored as row vectors $`f_i^{(t)}=(F_i^{(t)})^\top`$, Equation (17) becomes $`\mathring f_i^{(t)}=f_i^{(t)}\mathbf R^{(t)}`$. These transformations preserve the common linear predictor identically: $`\mathring{\boldsymbol{\Lambda}}^{(t)}\mathring F_i^{(t)}=\boldsymbol{\Lambda}^{(t)}F_i^{(t)}`$. Notice that $`\mathbf R^{(t)}`$ depends only on the loading draws and is returned when `return_transforms=True`.

Finally, the SP–RLME stage (Steps 2–3) can be run as a stand-alone alignment procedure without Step 1 varimax rotation — for example, to align loadings across multiple MCMC chains. Computational strategies for solving the discrete SP subproblem (15) are detailed in Section 4.2.

---

## 4. Computational Strategies

### 4.1 Iterative optimization of the varimax criterion

To seek a maximizer of the varimax objective (11), each MCMC draw is processed independently by a single-start matrix-form varimax iteration that updates all $`q`$ factors simultaneously. The implementation operates on unnormalized loadings (`normalize = FALSE`), without Kaiser row normalization. Let $`\mathcal V(\mathbf O)`$ denote the objective function in (11) for $`\mathbf O\in\mathbb R^{q\times q}`$ satisfying $`\mathbf O^\top\mathbf O=\mathbf I_q`$.

#### Matrix gradient and orthogonal update

Starting from $`\mathbf O_0=\mathbf I_q`$, let $`\mathbf O_k`$ denote the orthogonal rotation at iteration $`k`$. Define the rotated loading matrix $`\mathbf Z_k`$, the column sums of squared loadings $`c_{kj}`$, and the Euclidean matrix gradient $`\mathbf B_k`$ by

```math
\begin{aligned}
\mathbf Z_k&=\boldsymbol{\Lambda}\mathbf O_k,\\
c_{kj}&=\sum_{r=1}^{p}z_{rkj}^2,\qquad
\mathbf C_k=\mathop{\mathrm{diag}}\nolimits(c_{k1},\ldots,c_{kq}),\\
\mathbf B_k
&=\boldsymbol{\Lambda}^{\top}
\left(\mathbf Z_k^{\odot 3}-\frac{1}{p}\mathbf Z_k\mathbf C_k\right)
=\nabla\mathcal V(\mathbf O_k),
\end{aligned}
\qquad\text{(18)}
```

where $`\mathbf Z_k^{\odot 3}`$ denotes the elementwise cube of $`\mathbf Z_k`$. Because of the prefactor $`1/4`$ in (11), $`\mathbf B_k`$ is the exact Euclidean gradient of $`\mathcal V`$ at $`\mathbf O_k`$ (omitting the $`1/4`$ simply scales the gradient by $`4`$, which leaves the polar update (19) unchanged).

The next iterate $`\mathbf O_{k+1}`$ is obtained by maximizing the first-order linear approximation of $`\mathcal V`$ over the orthogonal group:

```math
\mathbf O_{k+1}
=\mathop{\mathrm{argmax}}_{\mathbf O^\top\mathbf O=\mathbf I_q}
\langle\mathbf B_k,\mathbf O\rangle_F
=\mathop{\mathrm{argmax}}_{\mathbf O^\top\mathbf O=\mathbf I_q}
\mathop{\mathrm{tr}}\nolimits(\mathbf B_k^\top\mathbf O).
\qquad\text{(19)}
```

From the singular value decomposition $`\mathbf B_k=\mathbf U_k\mathbf D_k\mathbf V_k^\top`$, where $`\mathbf D_k`$ is diagonal with nonnegative singular values, von Neumann's trace inequality yields $`\mathop{\mathrm{tr}}(\mathbf B_k^\top\mathbf O)\le\mathop{\mathrm{tr}}(\mathbf D_k)`$ for all orthogonal $`\mathbf O`$, with equality at

```math
\mathbf B_k=\mathbf U_k\mathbf D_k\mathbf V_k^\top,
\qquad
\mathbf O_{k+1}=\mathbf U_k\mathbf V_k^\top,
\qquad
d_k:=\mathop{\mathrm{tr}}\nolimits(\mathbf O_{k+1}^\top\mathbf B_k)
=\mathop{\mathrm{tr}}\nolimits(\mathbf D_k).
\qquad\text{(20)}
```

Note that $`\mathbf O_{k+1}=\mathbf U_k\mathbf V_k^\top`$ directly *replaces* the total rotation matrix rather than multiplying $`\mathbf O_k`$.

#### Polar decomposition and the QDWH alternative

Equivalently, the update in (20) is the orthogonal factor of the polar decomposition of $`\mathbf B_k`$:

```math
\mathbf B_k=\mathbf W_k\mathbf H_k,
\qquad
\mathbf W_k=\mathbf U_k\mathbf V_k^\top,
\qquad
d_k=\langle\mathbf W_k,\mathbf B_k\rangle_F=\mathop{\mathrm{tr}}\nolimits(\mathbf D_k),
\qquad\text{(21)}
```

where $`\mathbf W_k`$ is orthogonal and $`\mathbf H_k`$ is symmetric positive semidefinite (for nonsingular $`\mathbf B_k`$, $`\mathbf W_k=\mathbf B_k(\mathbf B_k^\top\mathbf B_k)^{-1/2}`$). Two numerical backends are available to compute $`\mathbf W_k`$:

1. **SVD route (`polar="svd"`):** compute the full SVD of $`\mathbf B_k`$ and set $`\mathbf W_k=\mathbf U_k\mathbf V_k^\top`$. This reproduces the reference iteration of R's `stats::varimax`.
2. **QDWH route (`polar="qdwh"`, default):** compute the orthogonal polar factor directly via the QR-based dynamically weighted Halley (QDWH) iteration, avoiding an explicit SVD, and evaluate $`d_k=\langle\mathbf W_k,\mathbf B_k\rangle_F`$.

If $`\mathbf B_k`$ is identically zero or nonfinite, the QDWH backend leaves the current rotation unchanged for that step. It also verifies convergence of the inner QDWH loop, finiteness, and orthogonality of the resulting factor (with default orthogonality tolerance $`10^{-3}`$ in `float32` and $`10^{-9}`$ in `float64`). Any draw failing these numerical checks is automatically recomputed via the SVD route, whereas accepted draws are kept as-is. These checks validate the polar decomposition step, not optimality of the varimax objective.

#### Scale invariance and numerical scaling

Because the varimax criterion is homogeneous of degree four, its maximizers are invariant under positive scalar rescaling of $`\boldsymbol{\Lambda}`$. To prevent overflow or underflow in the cubic and quartic intermediates of (18), each draw is rescaled prior to iteration:

```math
\boldsymbol{\Lambda}_{\alpha}=\frac{\boldsymbol{\Lambda}}{\alpha},
\qquad \alpha=\alpha_1\alpha_2>0,
\qquad
\mathbf B_{\alpha,k}=\alpha^{-4}\mathbf B_k,
\qquad
\mathop{\mathrm{polar}}\nolimits(\mathbf B_{\alpha,k})=\mathop{\mathrm{polar}}\nolimits(\mathbf B_k)
\quad(\det\mathbf B_k\neq0),
\qquad\text{(22)}
```

where $`\mathbf B_{\alpha,k}`$ is computed from $`\boldsymbol{\Lambda}_{\alpha}`$. Positive scaling leaves the unique orthogonal polar factor invariant when $`\mathbf B_k`$ is nonsingular, and preserves the set of polar factors (as well as any fixed SVD/polar branch) when $`\mathbf B_k`$ is singular. Once the iteration terminates, the final rotated loadings are formed on the original scale as $`\boldsymbol{\Lambda}\mathbf O_{\mathrm{final}}`$.

For any draw with $`a=\max_{r,j}|\lambda_{rj}|>0`$, the scale factor is chosen as an exact power of two: setting $`e=\lfloor\log_2 a\rfloor`$, $`e_1=\lfloor e/2\rfloor`$, and $`e_2=e-e_1`$, the loading matrix is divided successively by $`\alpha_1=2^{e_1}`$ and $`\alpha_2=2^{e_2}`$ (for $`a=0`$, $`\alpha_1=\alpha_2=1`$). In binary floating-point arithmetic, scaling normal numbers by powers of two within the representable exponent range is exact and introduces no rounding error; splitting the exponent into two factors prevents overflow of the scale factor itself at extreme magnitudes (though subnormal rounding cannot be avoided universally). This scalar preconditioning is purely numerical and is unrelated to Kaiser row normalization.

#### Stopping rule and output

The scalar $`d_k=\mathop{\mathrm{tr}}(\mathbf D_k)`$ in (20)–(21) is the nuclear norm of the gradient $`\mathbf B_k`$ rather than the varimax objective $`\mathcal V(\mathbf O_k)`$. Following `stats::varimax`, the iteration monitors relative stagnation of $`d_k`$ and terminates as soon as

```math
d_k < d_{k-1}(1+\varepsilon),
\qquad \varepsilon=10^{-5},
\qquad\text{(23)}
```

or upon reaching `1000` updates. Initializing $`d_0=0`$ guarantees that at least one update is performed for every nonzero draw with $`q\geq2`$. Upon termination with $`\mathbf O_{\mathrm{final}}`$, the routine returns $`\widetilde{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf O_{\mathrm{final}}`$ and $`\mathbf O_{\mathrm{final}}`$. For $`q<2`$ or an identically zero draw, the routine returns the input unchanged with $`\mathbf O_{\mathrm{final}}=\mathbf I_q`$ and an iteration count of zero. A nonzero draw whose gradient is identically zero — for example identical loading rows, where the varimax gradient vanishes for every rotation — exits after that first update with an iteration count of one instead of exhausting the budget: the strict rule (23) can never terminate it because $`d_k=0`$ makes $`d_k<d_{k-1}(1+\varepsilon)`$ false at every step.

For $`q\geq2`$, stopping rule (23) does not certify a local or global maximum — or even stationarity — of (11): the sequence may stagnate near a saddle point or exhaust the 1000-step budget, and $`\varepsilon=10^{-5}`$ bounds neither the objective gap nor the rotation error. Moreover, even exact global maximizers of (11) need not be unique up to signed permutations. Consequently, the sufficient conditions noted in Section 3.2.1 are not guaranteed by Step 1.

### 4.2 Solving the sign-permutation problem

For a fixed reference matrix, the global objective $`\Psi`$ in (13) is additively separable across draws, reducing the SP step (15) to $`T`$ independent discrete minimizations over $`2^q q!`$ signed permutations. The key computational insight is that optimal column signs can be eliminated analytically in closed form, reducing each draw's minimization to a single $`q\times q`$ linear assignment problem. This analytical elimination route (`sp_path="elim"`) is the default; full sign enumeration (`sp_path="enum"`) is retained for validation and legacy compatibility.

Suppressing the draw index $`t`$, write $`\mathbf A=(a_{rj})\in\mathbb{R}^{p\times q}`$ for the loading matrix being aligned and $`\mathbf M=(m_{rj})\in\mathbb{R}^{p\times q}`$ for the current reference matrix $`\boldsymbol{\Lambda}^{\star}`$. Under the assignment convention of Section 3.2.2, reference column $`j`$ is matched to draw column $`\nu_j`$.

#### Analytical elimination of the signs

For any candidate pairing of reference column $`j`$ with draw column $`k`$, the squared Euclidean residuals under the two possible signs $`s_j=+1`$ and $`s_j=-1`$ are

```math
c_{jk}^{(+)}
=\sum_{r=1}^{p}(a_{rk}-m_{rj})^2,
\qquad
c_{jk}^{(-)}
=\sum_{r=1}^{p}(a_{rk}+m_{rj})^2.
\qquad\text{(24)}
```

The minimum pairwise cost and the optimal sign for pair $`(j,k)`$ are therefore known immediately:

```math
c_{jk}=\min\left\{c_{jk}^{(+)},c_{jk}^{(-)}\right\},
\qquad
\eta_{jk}=
\begin{cases}
+1,&c_{jk}^{(+)}\leq c_{jk}^{(-)},\\
-1,&c_{jk}^{(+)}>c_{jk}^{(-)}.
\end{cases}
\qquad\text{(25)}
```

Ties ($`c_{jk}^{(+)}=c_{jk}^{(-)}`$) are broken deterministically in favor of $`\eta_{jk}=+1`$.

Because the choice of sign $`s_j`$ for column $`j`$ depends only on the matched pair $`(j,\nu_j)`$ and does not interact with other columns, minimization over $`s\in\{-1,1\}^q`$ commutes with minimization over $`\nu\in\mathcal T_q`$:

```math
\boxed{
\min_{\substack{s\in\{-1,+1\}^q\\\nu\in\mathcal T_q}}
\mathcal L_{s,\nu}
=
\min_{\nu\in\mathcal T_q}
\sum_{j=1}^{q}c_{j,\nu_j}.
}
\qquad\text{(26)}
```

This reduces the $`2^q q!`$-state joint search to a $`q!`$-state permutation search over the precomputed $`q\times q`$ cost matrix $`\mathbf C=(c_{jk})`$.

#### Reduction to a linear assignment problem

Equation (26) is a standard square linear assignment problem on the cost matrix $`\mathbf C=(c_{jk})`$, whose rows index reference columns and whose columns index draw columns. Given an optimal assignment $`\nu^\star`$, the optimal signs, aligned loadings, and per-draw loss are recovered directly via

```math
\nu^\star=\mathop{\mathrm{argmin}}_{\nu\in\mathcal T_q}
\sum_{j=1}^{q}c_{j,\nu_j},
\qquad
s_j^\star=\eta_{j,\nu_j^\star},
\qquad
\mathring a_{rj}=s_j^\star a_{r\nu_j^\star},
\qquad
\mathcal L^\star=\sum_{j=1}^{q}c_{j,\nu_j^\star}.
\qquad\text{(27)}
```

Thus, each draw requires only a single assignment solve per outer SP sweep, followed by a column gather and sign multiplication.

Three assignment backends are supported:

1. **Exhaustive permutation search:** evaluates all $`q!`$ permutations directly and returns the lexicographically first minimizer under ties. Used by default for small $`q`$ ($`q\leq 7`$).
2. **Hungarian algorithm (Optax):** polynomial-time solver with $`O(q^3)`$ complexity, used by default for $`q>7`$.
3. **SciPy solver (`assignment_backend="scipy"`):** host-side `scipy.optimize.linear_sum_assignment` backend available for validation (selected explicitly, not as an automatic fallback).

The dispatch threshold between exhaustive search and the Hungarian algorithm (default $`q=7`$) is configurable. Forming the $`q\times q`$ cost matrix requires $`O(pq^2)`$ operations per draw, and batch chunking affects only memory layout and execution parallelism without altering the optimization results.

#### Equivalent inner-product formulation and direct residual evaluation

Algebraically, expanding the squared residuals in (24) in terms of the column inner products $`g_{jk}`$ gives

```math
g_{jk}=\sum_{r=1}^{p}m_{rj}a_{rk},
\qquad
c_{jk}^{(+)}-c_{jk}^{(-)}=-4g_{jk},
\qquad
c_{jk}=\sum_{r=1}^{p}a_{rk}^2+\sum_{r=1}^{p}m_{rj}^2-2|g_{jk}|.
\qquad\text{(28)}
```

Hence $`\eta_{jk}=\mathrm{sign}(g_{jk})`$ (with $`\eta_{jk}=+1`$ when $`g_{jk}=0`$). Moreover, because any bijection $`\nu\in\mathcal T_q`$ visits every column of $`\mathbf A`$ and every column of $`\mathbf M`$ once, the squared column norms sum to constants across all permutations:

```math
\min_{\nu\in\mathcal T_q}\sum_{j=1}^{q}c_{j,\nu_j}
=
\|\mathbf A\|_F^2+\|\mathbf M\|_F^2
-2\max_{\nu\in\mathcal T_q}\sum_{j=1}^{q}|g_{j,\nu_j}|.
\qquad\text{(29)}
```

Although maximizing $`\sum_j|g_{j,\nu_j}|`$ in (29) is algebraically equivalent to minimizing $`\sum_j c_{j,\nu_j}`$, the implementation intentionally evaluates $`c_{jk}^{(+)}`$ and $`c_{jk}^{(-)}`$ directly from the pairwise differences $`(a_{rk}\mp m_{rj})^2`$ in (24) rather than from the expanded form (28). Direct residual evaluation avoids catastrophic floating-point cancellation when $`\|\mathbf A\|_F^2+\|\mathbf M\|_F^2`$ and $`2\sum_j|g_{j,\nu_j}|`$ are large and nearly equal.

#### Exhaustive sign enumeration as a validation route

To validate the elimination solver and reproduce the exact tie-breaking order of the reference R implementation, `sp_path="enum"` enumerates all $`2^q`$ sign vectors explicitly. For each fixed $`s\in\{-1,+1\}^q`$, define the sign-conditioned costs and assignment value

```math
c_{jk}^{(s)}=\sum_{r=1}^{p}(a_{rk}-s_jm_{rj})^2,
\qquad
\mathcal L^\star(s)=
\min_{\nu\in\mathcal T_q}\sum_{j=1}^{q}c_{j,\nu_j}^{(s)},
\qquad
\mathcal L^\star=\min_{s\in\{-1,+1\}^q}\mathcal L^\star(s).
\qquad\text{(30)}
```

This route solves $`2^q`$ linear assignment problems per draw (versus one in the elimination route) and retains the first sign vector that achieves the minimum under strict comparison (`<`). Both routes solve the exact same fixed-reference SP problem (15) and achieve the same minimum loss $`\mathcal L^\star`$, differing at most in tie-breaking among equally optimal signed permutations.

Finally, while both routes solve each SP subproblem globally for a fixed reference $`\mathbf M`$, the outer RSP loop alternates SP steps with RLME mean updates. Exactness of the inner SP solver therefore does not guarantee that the outer alternating algorithm attains a global minimum of the joint objective (13).

## References

1. Papastamoulis, P., & Ntzoufras, I. (2022). On the identifiability of Bayesian factor analytic models. *Statistics and Computing*, **32**(2), 23.
2. Rey-Sáez, R., & Revuelta, J. (2026). An Efficient Rotation-Sign-Permutation Algorithm to Solve Rotational Indeterminacy in Bayesian Exploratory Factor Analysis. *Bayesian Analysis*, **1**(1), 1–23.
