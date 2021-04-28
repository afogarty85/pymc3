# sk learn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import theano
import xarray as xr
from sklearn.datasets import make_regression
import scipy.stats as stats

features, target, true_coef = make_regression(n_samples=100,
                                  bias=1.0,
                                  n_informative=3,
                                  n_features=3,
                                  noise=2.5,
                                  random_state=1,
                                  coef=True)


# coords
coords = {
          # dim 1: len of df
          'obs_id': np.arange(len(target)),
          # dim 2: feature cats.
          'features': ['treatment', 'cov1', 'cov2']
    }

# specify model
with pm.Model(coords=coords) as sk_lm:
    '''
    Bayesian Linear Model
    '''
    # data
    feature_data = pm.Data('feature_data', features, dims=('obs_id', 'features'))

    # priors
    alpha = pm.Normal('alpha', mu=0, sigma=1)  # sigma is the std dev.
    betas = pm.Normal('betas', mu=[40, 90, 50], sigma=5, dims='features')

    # model error
    sigma = pm.Exponential("sigma", lam=1)

    # matrix-dot products
    m1 = pm.math.matrix_dot(feature_data, betas)

    # expected value of y
    mu = pm.Deterministic("mu", alpha + m1, dims='obs_id')

    # Likelihood: Normal
    y = pm.Normal("y",
                  mu=mu,
                  sigma=sigma,
                  observed=target,
                  dims='obs_id')

    # set sampler
    step = pm.NUTS([alpha, betas, sigma], target_accept=0.9)

    # Inference button (TM)!
    lm_trace = pm.sample(draws=1000,
                           step=step,
                           init='jitter+adapt_diag',
                           cores=4,
                           tune=500,  # burn in
                           return_inferencedata=False)

    # prior analysis
    prior_pc = pm.sample_prior_predictive()

    # posterior predictive
    ppc = pm.fast_sample_posterior_predictive(trace=lm_trace,
                                              random_seed=1,
                                              )
    # generate inference data
    lm_idata = az.from_pymc3(
                             trace=lm_trace,
                             prior=prior_pc,
                             posterior_predictive=ppc,
                             )

# graph of model
pm.model_to_graphviz(sk_lm)

# diagnostics: plot prior
with sk_lm:
    az.plot_ppc(data=lm_idata, num_pp_samples=100, group='prior');

# diagnostics: plot posterior
with sk_lm:
    fig, ax = plt.subplots(figsize=(12,8))
    az.plot_ppc(data=lm_idata, num_pp_samples=100, group='posterior', ax=ax);
    ax.axvline(np.mean(target), ls="--", color="r", label="True mean")
    ax.legend(fontsize=12);

# diagnostics: plot trace
with sk_lm:
    az.plot_trace(lm_idata,
                  coords={ 'features': ['treatment'] },
                  var_names=['~mu', '~alpha', '~sigma']);

# posterior 94% High Density Interval
with sk_lm:
    az.plot_posterior(lm_idata,
                  coords={ 'features': ['treatment'] },
                  var_names=['~mu', '~alpha', '~sigma']);

# diagnotics: plot r-hat
az.summary(lm_idata, kind='diagnostics', var_names=['~mu'])

# view coefs
az.summary(lm_idata, var_names=["betas"], kind='stats')


# Posterior Analysis
post = lm_idata.posterior
# extract the data used to build the model
const = lm_idata.constant_data

# y-hat
post['mu'].mean(dim=("chain", "draw")).values


# Counterfactual Plot: Hold Covariates Constant:
# treatment variable column
idx_pred = 0
# indices for the covariates
idx_covs = [1, 2]
# generate low and high values to vary
low, high = np.zeros(features.shape[1]), np.zeros(features.shape[1])
# vary X1 from 1 to 3
low[idx_pred], high[idx_pred] = 1, 3
# generate 25 evenly spaced observations
treatment_seq = np.linspace(start=low, stop=high, num=25)
# hold other vars at their mean; find mean
cov1_mu, cov2_mu = np.mean(features[:, 1]), np.mean(features[:, 2])
# add in the mu
treatment_seq[:, idx_covs] = cov1_mu, cov2_mu

# compute counterfactual probabilities:
with sk_lm:
    # set the new data
    pm.set_data({"feature_data": treatment_seq})
    # run posterior predictive sampling
    post_checks = pm.fast_sample_posterior_predictive(
        lm_trace)

# get y-hat
estimated_mu = post_checks['y'].mean(axis=0)

# plot
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(treatment_seq[:, idx_pred], estimated_mu[:, ])
az.plot_hpd(treatment_seq[:, idx_pred], post_checks['y'][:, :], ax=ax)
ax.set_xlabel(f"Treatment")
ax.set_ylabel(f"Expected value of Y")
ax.set_title("Other predictors held at mean")



# Predictions
# generate new data
new_data = np.random.randn(3, 3)

with sk_lm:
    # set it over the feature data
    pm.set_data({ "feature_data": new_data})
    # generate preds
    predictions = pm.fast_sample_posterior_predictive(lm_idata)
    preds = az.from_pymc3_predictions(predictions,
                              coords={ 'obs_id': [0, 1, 2] },
                              idata_orig=lm_idata,
                              inplace=False)

# get y-hat for new X data
preds.predictions['y'].median(dim=('chain', 'draw')).values

# view y-hat as 94% HDI
az.plot_posterior(preds, group="predictions");


# Leave One Out Cross Validation Estimation
with sk_lm:
    loo = az.loo(lm_idata, sk_lm)
loo




# generate expected value of y given treatment of interest; alpha + beta1*x1
# slice post on: (chain, draw, features)
mu = (post["alpha"] + (post['betas'][:, :, 0] * const['feature_data'][:, 0]))  # (4, 1000, 100)

# average change in y for each unit increase of x1
np.mean(mu.mean(dim=("chain", "draw"))).values  # 1.50

mu.mean(dim=("chain", "draw"))

# manually generate HDI information
hdi_data_eY = az.hdi(mu, input_core_dims=[["chain", "draw"]])
hdi_data_yhat = az.hdi(lm_idata.posterior_predictive['y'], input_core_dims=[["chain", "draw"]])

# Partial Derivative / Marginal Effect
# the predicted value of Y changes with a one-unit increase in a particular X
# with no interactions or curvilinear terms, then the marginal effects are the same as the coefficients

# start plot
_, ax = plt.subplots(figsize=(8,8))
# plot data
ax.plot(features[:, 0], target, "o", ms=4, alpha=0.7, label="Data")
# mean outcome; # E(y) for one unit change in alpha + b1*x1
ax.plot(features[:, 0], mu.mean(dim=("chain", "draw")), color="000000", label="Mean outcome", alpha=0.6)

# plot 94% probability for marginal effect
az.plot_hdi(x=features[:, 0],  # observed treatment
            hdi_data=hdi_data_eY,  # 94% probability marginal effect for mu
            ax=ax,
            hdi_prob=0.94,
            fill_kwargs={"alpha": 0.7, "label": "Mean outcome 94% HPD"})

# plot 94% HPD for posterior possible unobserved y values
az.plot_hdi(x=features[:, 0],  # observed treatment data
            hdi_data=hdi_data_yhat,  # set of possible unobserved y
            ax=ax,
            hdi_prob=0.94,
            fill_kwargs={"alpha": 0.7, "color": "#a1dab4", "label": "Outcome 94% HPD"})

ax.set_xlabel("Predictor")
ax.set_ylabel("Outcome")
ax.set_title("Posterior predictive checks")
ax.legend(ncol=2, fontsize=10, frameon=True);

# get hdis for parameters from trace
with sk_lm:
    trace_hdi = az.hdi(lm_idata)

# Plot High Density Interval for Treatment
_, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=coords['features'],
           y=lm_idata.posterior["betas"].median(dim=("chain", "draw")),
           color="C0",
           alpha=1,
           s=55)
ax.vlines(
    x=coords['features'],
    ymin=trace_hdi["betas"].sel({"hdi": "lower"}),
    ymax=trace_hdi["betas"].sel({"hdi": "higher"}),
    alpha=0.6,
    lw=5,
    color="C0",
)
ax.set_xlabel("Features")
ax.set_ylabel("Posterior Impact on Y")
ax.set_title("HDI of Features Impact on Y");




### POLR
# load data
treatment_ucb = pd.read_csv('C:\\Users\\Andrew\\Desktop\\treatment_ucb.csv')

# extract y
target = treatment_ucb['Intelligence']

# select useful vars
treatment = treatment_ucb['Treatment']
covariates = treatment_ucb[['Length',
                               'Gender', 'English', 'Race', 'isUS',
                               'Degree', 'age_bins', 'ReadSocialMedia',
                               'WriteSocialMedia', 'Intelligence.pretreat',
                               'Writing.pretreat', 'Interest.pretreat',
                               'Effective.pretreat']]

# onehot
treatment = pd.get_dummies(data=treatment, drop_first=True)
covariates = pd.get_dummies(data=covariates,
                            columns=[col for col in covariates.columns],
                            drop_first=True)

# coords
coords = {
          # dim 1: treatment var
          'treatment': treatment.columns,
          # dim 2: covariates
          'covariates': covariates.columns,
          # dim 3: len of df
          'obs_id': np.arange(len(target)),
          # dim 4: cutpoints
          'cut_id': np.arange(6)
    }

import scipy as sp
# log cumulative odds for priors testval
target.value_counts(normalize=True, sort=False).cumsum().apply(sp.special.logit)

# to arrays
treatment = treatment.values
covariates = covariates.values
target = target.values

# specify model
with pm.Model(coords=coords) as cumlink:
    '''
    Bayesian Ordered Logistic Model
    '''
    # data containers
    covariates_data = pm.Data('covariates_data', covariates, dims=('obs_id', 'covariates'))
    treatment_data = pm.Data('treatment_data', treatment, dims=('obs_id', 'treatment'))

    # priors; cutpoints replace the intercepts
    cutpoints = pm.Normal("cutpoints", mu=0, sigma=1.5, transform=pm.distributions.transforms.ordered, dims='cut_id', testval=np.arange(6) - 2.5)
    covariates_betas = pm.Normal("covariates_betas", mu=0, sigma=1, dims='covariates')
    treatment_betas = pm.Normal("treatment_betas", mu=0, sigma=1, dims='treatment')

    # matrix-dot products
    m1 = pm.math.matrix_dot(treatment_data, treatment_betas)
    m2 = pm.math.matrix_dot(covariates_data, covariates_betas)

    # expected value of y
    phi = pm.Deterministic("phi", m1 + m2)

    # Likelihood: OrderedLogistic
    y = pm.OrderedLogistic("y",
                           eta=phi,
                           # theano.tensor.sort(cutpoints) needed for ppc
                           cutpoints=theano.tensor.sort(cutpoints),
                           observed=target-1)

    # set step
    step = pm.NUTS([treatment_betas, covariates_betas, cutpoints], target_accept=0.9)

    # Inference button (TM)!
    polr_trace = pm.sample(draws=1250,
                           step=step,
                           init='jitter+adapt_diag',
                           cores=4,
                           tune=500,  # burn in
                           return_inferencedata=False)
    # prior analysis
    prior_pc = pm.sample_prior_predictive()

    # posterior predictive
    ppc = pm.fast_sample_posterior_predictive(trace=polr_trace,
                                              random_seed=1,
                                              )
    # inference data
    polr_idata = az.from_pymc3(
                             trace=polr_trace,
                             prior=prior_pc,
                             posterior_predictive=ppc,
                             )

# graph of model
pm.model_to_graphviz(cumlink)

# diagnostics: plot prior
with cumlink:
    az.plot_ppc(data=polr_idata, num_pp_samples=100, group='prior');

# diagnostics: plot posterior
with cumlink:
    fig, ax = plt.subplots(figsize=(12,8))
    az.plot_ppc(data=polr_idata, num_pp_samples=100, group='posterior', ax=ax);
    #ax.axvline(np.mean(target), ls="--", color="r", label="True mean")
    ax.legend(fontsize=12);

# diagnostics: plot trace
with cumlink:
    az.plot_trace(polr_idata, var_names=["treatment_betas"]);

# posterior 94% High Density Interval
with cumlink:
    az.plot_posterior(polr_idata, var_names=["treatment_betas"]);

# diagnotics: plot r-hat
az.summary(polr_idata, var_names=["~phi"], kind='diagnostics')

# view coefs
az.summary(polr_idata, var_names=['treatment_betas'], kind='stats')

# Posterior Analysis
post = polr_idata.posterior
# extract the data used to build the model
const = polr_idata.constant_data

from scipy.special import expit as logistic
# distance between cutpoints is related to the frequency of each category of y
# but almost never interpreted
logistic(post['cutpoints'].mean(dim=("chain", "draw")))


# get hdis from trace
with cumlink:
    trace_hdi = az.hdi(polr_idata)

# Plot High Density Interval for Treatment
_, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=['Phonological', 'Typographical'],
           y=post["treatment_betas"].median(dim=("chain", "draw")),
           color="C0",
           alpha=1,
           s=100)
ax.vlines(
    x=['Phonological', 'Typographical'],
    ymin=trace_hdi["treatment_betas"].sel({"hdi": "lower"}),
    ymax=trace_hdi["treatment_betas"].sel({"hdi": "higher"}),
    alpha=0.6,
    lw=5,
    color="C0",
)
ax.set_xlabel("Treatment Category")
ax.set_ylabel("Posterior Impact on Intelligence")
ax.set_title("HDI of Treatment Impact on Intelligence");

# model comparsion - not run
# df_comparative_waic = az.compare(dataset_dict={"ordered logit": polr_idata, "linear model": lm_idata}, ic='waic')

# visual comparison - not run
# az.plot_compare(df_comparative_waic, insample_dev=False, figsize=(10, 4));


# generate counterfactual contrasts
cases_df = pd.DataFrame(
    np.array([[0, 0], [1, 0], [0, 1]]),
    columns=["Phonological", "Typographical"])

# transform data to df
ppc_y = polr_idata.posterior_predictive['y'].stack(dim=('chain', 'draw')).T.values
response_df = pd.DataFrame(ppc_y)
response_df.index.name = "case"
response_df = (
    pd.concat([cases_df, response_df], axis=1)
    .set_index(["Phonological", "Typographical"])
    .sort_index()
)
# drop nan
c = response_df.index.names
response_df = response_df.reset_index().dropna().set_index(c)


# show implied histogram of simulated outcomes
_, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
bins, xticks, xlabels, colors = (
    np.arange(8) - 0.5,
    np.arange(7),
    np.arange(1, 8),
    ["k", "b"],
)

# 0, 0
axes[0].hist(
    [response_df.loc[0, 0].values.flatten(), response_df.loc[0, 0].values.flatten()],
    bins=bins,
    rwidth=0.5,
    color=colors,
    alpha=0.7,
)
axes[0].set_title("Phonological=0, Typographical=0")
axes[0].set_ylabel("frequency")
axes[0].legend(fontsize=10)

axes[1].hist(x=
    [response_df.loc[1, 0].values.flatten(), response_df.loc[0, 1].values.flatten()],
    bins=bins,
    rwidth=0.5,
    color=colors,
    alpha=0.7,
)
axes[1].set_title("Phonological=1, Typographical=0")

for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("response")
plt.tight_layout();




















#
