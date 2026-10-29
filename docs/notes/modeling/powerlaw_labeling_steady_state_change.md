# Power-law $F_{new}$ when the labeling changes the steady state

Preserved from `notebooks/06_sensitivity_analysis_old.ipynb` (deleted; see git
history), where it was worked out "after talking with Elad". It is the power-law
counterpart of the lognormal steady-state sensitivity in
[`notebooks/06b_lognormal_steady_state_sensitivity.py`](../../../notebooks/06b_lognormal_steady_state_sensitivity.py):
a vegetation transition at time $0$ (labeling time $t'$ later) changes the input
$J$ and possibly the parameters of the age distribution.

---

## 1. Stocks of a given age

The power-law model with parameters $a$ ($t_{min}$) and $\tau$ ($t_{max}$) has
the impulse response (the fraction of an input still present after time $t$)

$$ s(t) = \frac{a\,e^{-t/\tau}}{a+t}, $$

and the transit time

$$ T = \int_0^\infty s(t)\,dt = a\,e^{a/\tau}E_1(a/\tau). $$

At steady state with input $J$, the carbon of age $t$ is $J\,p_A(t)\,T = J\,s(t)$,
where

$$ p_A(t) = \frac{1}{E_1(a/\tau)} \frac{e^{-(a+t)/\tau}}{a+t}, \qquad
P_A(t) = 1 - \frac{E_1\left(\frac{a+t}{\tau}\right)}{E_1(a/\tau)} $$

are the age density and CDF (`PowerLawDisKin.pA` and `cdfA`). A first draft of
this note worked with $p_A$ directly, which drops the factor $J\,T$; that is
harmless only when $J$ and the parameters do not change at the transition.

## 2. Labeled and unlabeled stocks

At a time $t'$ after the transition, the unlabeled carbon is everything older
than $t'$, and the labeled carbon is everything younger:

$$ C_{unlabeled}(t') = J\int_{t'}^{\infty} s(t)\,dt = J\,a\,e^{a/\tau}E_1\left(\frac{a+t'}{\tau}\right), $$

$$ C_{labeled}(t') = J\int_0^{t'} s(t)\,dt = J\,a\,e^{a/\tau}\left[E_1(a/\tau) - E_1\left(\frac{a+t'}{\tau}\right)\right]. $$

(Divided by $J\,T$, these are $1 - P_A(t')$ and $P_A(t')$.)

## 3. Fraction of new carbon

With input $J_1$ and parameters $a_1, \tau_1$ before the transition, and $J_2$,
$a_2, \tau_2$ after it, the old carbon decays with the old parameters and the
new carbon accumulates with the new ones:

$$ f(t') = \frac{J_2\,a_2 e^{a_2/\tau_2}\left[E_1\left(\frac{a_2}{\tau_2}\right) - E_1\left(\frac{a_2+t'}{\tau_2}\right)\right]}
{J_1\,a_1 e^{a_1/\tau_1}E_1\left(\frac{a_1+t'}{\tau_1}\right) + J_2\,a_2 e^{a_2/\tau_2}\left[E_1\left(\frac{a_2}{\tau_2}\right) - E_1\left(\frac{a_2+t'}{\tau_2}\right)\right]}. $$

The notebook's final formula had the bracketed (labeled) and unbracketed
(unlabeled) terms swapped, which gives $f(0) \neq 0$; the version above has
$f(0) = 0$ and, with no change at the transition, reduces to $f(t') = P_A(t')$.

The draft also conjectured that this equals the convolution of the impulse
response with the step change in input, $G(t) = \int_0^t J(t')\,s(t-t')\,dt'$
with $J(t) = J_1$ before the transition and $J_2$ after it. That holds when the
parameters do not change; when they do, each input cohort decays with the
parameters in force when it entered, as above.
