# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax==0.8.2",
#     "marimo",
# ]
# ///
import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first class is half philosophical and half technical.

    The philosophical part introduced the Bayesian (scientific research) workflow. Statistics is about correlation, all causality arguments came from the domain theory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The practical part did a Bayesian update on a Bernoulli distributed variable, using the "forking path of data" method.
    We started with a percentage of water on the planet experiment, and collected some data, and showed how the posterior updates with the stream of data. However it's not clear *how* the update happened, just the plots. To actually understand how it works we have to work with the Beta prior and the Bernoulli likelihood? These are for later lectures.

    Then we jumped to a second, much simpler experiment. Still percentage of water on the planet, but an equal prior, with only 5 possible values, 0, 1, 2, 3, 4 out of 4 portion are water, and with a smaller data set of 3 obervations, WLW. However it's also shown that more data, or more elaborate prior can be easily incorporated into the calculation, since the Bayesian update process is multiplicative.

    This is the grid approximation of the water percentage problem, we have discretized a continuous variable between 0 and 1 to a discrete one with only 5 possible values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we can count the number of ways that an observed sequence of data can happen, by sequentially multiplying new counts with existing ones.
    """)
    return


@app.cell
def _():
    portions_water = [0, 1, 2, 3, 4]
    portions_land = [4 - pw for pw in portions_water]
    obs = [1, 0, 1]
    counts = [1, 1, 1, 1, 1]
    for _ob in obs:  # all proposals are equally likely before observations
        if _ob == 1:
            for i, (c, p) in enumerate(zip(counts, portions_water)):
                counts[i] = c * p
        if _ob == 0:
            for i, (c, p) in enumerate(zip(counts, portions_land)):
                counts[i] = c * p
    counts  # final counts, after observing the data
    return counts, obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the counts are then normalized to probabilities
    """)
    return


@app.cell
def _(counts):
    probs = [c/sum(counts) for c in counts]
    probs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we can also work directly with proportions, and the likelihood function for Bernoulli data
    """)
    return


@app.cell
def _(obs):
    import jax.numpy as jnp
    props = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    posterior = jnp.ones(5)
    for _ob in obs:  # uniform prior
        likelihood = props ** _ob * (1 - props) ** (1 - _ob)  # streaming observations, online update
        posterior = posterior * likelihood
    posterior / sum(posterior)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## homework
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the homework problem, 10 participants, flipped the same coin, reported head or tail, and if reported head win a 10 euros cash prize. As it turned out 8 of them claimed the prize, and we want to know
    1. how many ways the observed data can be realized, if all participants are honest;
    1. how many ways the observed data can be realized, if only 5 of the participants are honest;
    1. the number of honest participants that maximize the number of ways the observed data can be realized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    so this is an inference problem about the number (or portion) of honest participants.
    However, note that the problem is asking an "if" question: if there are `h` honest participant, how many ways the data can be realized. "if" means conditioning on, so we are condiering likelihoods.

    we have 10 participants, 8 reported heads; so the other 2 must be honest, and the possible number of honest participants are [2, 10].
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **If all 10 participants are honest:** just choose 2 people for tails, and the other 8 for heads, the number of ways is:

    $$\binom{10}{2} \cdot \binom{8}{8} = \binom{10}{2}$$

    **If only 5 participants are honest:** choose the 5 honest ones, then choose 2 for tails, the other 3 for heads. The number of ways is:

    $$\binom{10}{5} \cdot \binom{5}{2} \cdot \binom{3}{3} \cdot \binom{5}{5}$$

    **General case with \(h\) honest participants:** choose the `h` honests, then choose 2 tails, the other \(h-2\) heads.

    Number of ways:

    $$\binom{10}{h} \cdot \binom{h}{2} \cdot \binom{h-2}{h-2} \cdot \binom{10-h}{10-h} = \binom{10}{h} \cdot \binom{h}{2}$$
    """)
    return


@app.cell
def _():
    from math import comb

    N = 10
    combos = {h: comb(N, h) * comb(h, 2) for h in range(2, N+1)}
    combos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we want to go further, we can probably build a full generative story?
    1. the coin fairness unknow, so p~Beta (all participants using identical coin)
    2. the coin toss outcome xn~Bernoulli(p)
    3. the honesty of each participant hn~Beta
    4. the outcome should be a mixture distribution, 1 if the coin toss outcome is 1 or participant not honest, 0 if toss outcome is 0 AND participant honest.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
