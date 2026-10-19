# Movie Rating Prediction with Bayesian Matrix Factorization

Bayesian matrix factorization (BMF) implemented **from scratch with a Gibbs sampler** in NumPy. It predicts user ratings on MovieLens and is compared against standard classifiers.

> Course project for **ESTR2020 Statistics for Engineers**, The Chinese University of Hong Kong (Spring 2024).

## Data
[MovieLens "small"](https://grouplens.org/datasets/movielens/): 100,836 ratings from 610 users on 9,742 movies. Ratings run from 0.5 to 5 in half-star steps. The matrix is 98.3% empty. Files: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`.

A random 70/30 split separates training and test ratings.

## Model

Each rating is modelled as $R_{ij} \sim \mathcal{N}(u_i^\top v_j, \sigma^2)$, with Gaussian priors $u_i, v_j \sim \mathcal{N}(0, I_K)$ on the K-dimensional user and movie factors.

**Gibbs sampling.** The conditionals are conjugate, so each sweep draws:
- every user vector $u_i \mid V, R \sim \mathcal{N}\big(\mu_i, (I + \tau \sum_{j \in \Omega_i} v_j v_j^\top)^{-1}\big)$
- every movie vector $v_j$ symmetrically, given U

Here τ = 2 is the noise precision and $\Omega_i$ is the set of movies user i rated.

**Prediction.** Run 1,000 iterations and discard the first 80% as burn-in. For each test pair, average the predictive density of each candidate rating (0, 0.5, …, 5) over the retained posterior samples. Predict the rating with the highest posterior predictive probability.

## Results (test RMSE, lower is better)

| Model | RMSE |
|---|---|
| **BMF (Gibbs), K = 20** | **1.113** |
| Logistic regression | 1.162 |
| Support vector classifier | 1.164 |
| k-nearest neighbours (k = 20) | 1.181 |
| Random forest (20 trees) | 1.253 |

Effect of latent dimension K: 25 → 1.119, **20 → 1.113**, 15 → 1.115, 10 → 1.115, 5 → 1.121.

Note: the baseline classifiers use only raw user and movie IDs as features. They are a lower bound, not tuned recommenders. The main point of the project is the sampler itself and the posterior predictive approach.

## Run

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook ESTR2020proj.ipynb
```

The pure-Python Gibbs loop is slow (1,000 sweeps over 610 users and 9,742 movies for each K). Vectorising the per-user and per-movie updates is the obvious next speed-up.

## Tech stack
Python · NumPy · pandas · scikit-learn · Matplotlib
