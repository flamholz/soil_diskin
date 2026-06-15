# Closed-form derivation for the lognormal Diskin model (fixed input)

Companion to [`diskin_utils_fast.jl`](diskin_utils_fast.jl).

## 1. Model

The Diskin model represents soil carbon as a continuum of pools indexed by a
first-order decay rate $k > 0$. Each pool $C_k(t)$ evolves under

$$
\frac{\partial C_k}{\partial t} \\,=\\, I(t)\\,f(k) \\,-\\, k\\,C_k(t),
\qquad C_k(0)=0,
$$

where:

- $I(t)$ is the carbon input rate (mass per time),
- $f(k)$ is the input partitioning over rates, with $\int_0^\infty f(k)\\,dk = 1$,
- $k$ is the first-order decay constant of pool $k$.

For the lognormal variant, $f$ is the lognormal density

$$
f(k) \\,=\\, \frac{1}{k\\,\sigma\sqrt{2\pi}}\exp\\!\Bigl(-\frac{(\log k - \mu)^2}{2\sigma^2}\Bigr),
$$

with $\mu, \sigma$ determined from the transit time $\tau$ and mean age $\bar a$ via

$$
\sigma = \sqrt{\log(\bar a / \tau)}, \qquad
\mu = -\log\\!\sqrt{\tau^3/\bar a}.
$$

The total carbon stock is

$$
C(t) \\,=\\, \int_0^\infty C_k(t)\\,dk.
$$

## 2. Closed form for constant input

Assume $I(t) \equiv I$ (constant). The ODE for $C_k$ is linear first-order with
constant coefficients:

$$
\dot C_k + k\\,C_k \\,=\\, I\\,f(k).
$$

Multiplying by the integrating factor $e^{kt}$ gives
$\frac{d}{dt}\bigl(e^{kt}\\,C_k\bigr) = I\\,f(k)\\,e^{kt}$, and integrating from
$0$ to $t$ with $C_k(0)=0$:

$$
e^{kt}\\,C_k(t) \\,=\\, I\\,f(k)\\,\frac{e^{kt}-1}{k}.
$$

Hence

$$
\boxed{\\,C_k(t) \\,=\\, \frac{I\\,f(k)}{k}\\,\bigl(1 - e^{-kt}\bigr)\\,}
$$

This is the pool-resolved closed form: no ODE integration required.

## 3. Total carbon as a 1-D integral

Integrating over $k$:

$$
C(t) \\,=\\, I\\!\int_0^\infty \frac{f(k)}{k}\\,\bigl(1 - e^{-kt}\bigr)\\,dk.
$$

The integrand picks up an extra $1/k$ on top of $f(k)$, which is a problem if
we try to discretize $k$ uniformly: $f(k)/k$ is sharply peaked near $k\\!\to\\!0$
when $\mu$ is small. The cure is to integrate in **log-rate space**.

### Change of variables $u = \log k$

Let $u = \log k$, so $k = e^u$ and $dk = e^u\\,du$. The lognormal density has
the convenient property

$$
f(k)\\,dk \\,=\\, \varphi_{\mathcal{N}}(u;\\,\mu,\sigma)\\,du,
$$

where $\varphi_{\mathcal{N}}(\cdot;\mu,\sigma)$ is the Gaussian density of $u$ —
in other words, the log of a lognormal-$\\!(\mu,\sigma)$ variable is normal-$\\!(\mu,\sigma)$.

Therefore

$$
\frac{f(k)}{k}\\,dk
\\,=\\, \frac{1}{k}\\,\varphi_{\mathcal{N}}(u;\\,\mu,\sigma)\\,du
\\,=\\, e^{-u}\\,\varphi_{\mathcal{N}}(u;\\,\mu,\sigma)\\,du.
$$

Substituting:

$$
\boxed{\\,
C(t) \\,=\\, I\\!\int_{-\infty}^{\infty}
e^{-u}\\,\varphi_{\mathcal{N}}(u;\\,\mu,\sigma)\\,
\bigl(1 - e^{-e^{u}\\,t}\bigr)\\,du.
\\,}
$$

This is the form used in [`diskin_utils_fast.jl`](diskin_utils_fast.jl#L42):
one smooth 1-D integral per time point — no ODE solver.

## 4. Numerical considerations

### Domain truncation

The Gaussian factor $\varphi_{\mathcal{N}}(u;\mu,\sigma)$ decays like
$\exp\\!\bigl(-(u-\mu)^2/(2\sigma^2)\bigr)$, which kills the $e^{-u}$ factor at
$\pm\infty$. Combining the two log-weights,

$$
e^{-u}\\,\varphi_{\mathcal{N}}(u;\mu,\sigma)
\\,\propto\\, \exp\\!\Bigl(-u - \frac{(u-\mu)^2}{2\sigma^2}\Bigr),
$$

so the integrand peaks where $\frac{d}{du}\\!\bigl[-u-(u-\mu)^2/(2\sigma^2)\bigr] = 0$,
i.e. at

$$
u^\star \\,=\\, \mu - \sigma^2.
$$

The implementation truncates to $[\mu - \sigma^2 - 10\sigma,\ \mu + 10\sigma]$,
which is more than $10\sigma$ past the peak in both directions — well past
double precision.

### Numerical stability of $1 - e^{-kt}$

For $kt \ll 1$ the term $1 - e^{-kt}$ loses precision through catastrophic
cancellation. We use the standard library identity

$$
1 - e^{-x} \\,=\\, -\mathrm{expm1}(-x),
$$

which is accurate to machine precision down to $x \to 0$. This matters at the
small-$t$ end of the time grid and for the rapidly decaying pools.

### Adaptive quadrature

`QuadGK.quadgk` is adaptive Gauss–Kronrod; with `rtol=1e-10` over the truncated
interval it converges in a few dozen evaluations and self-consistency across
different tolerances reaches $\sim 10^{-9}$ relative — well below the $\sim 10^{-3}$
intrinsic accuracy of the original ODE-based implementation
(`DifferentialEquations.jl` default `reltol`).

## 5. Time-varying input (for reference)

When $I$ depends on time, the linear ODE still solves analytically via the
integrating-factor argument:

$$
C_k(t) \\,=\\, f(k)\\!\int_0^t I(s)\\,e^{-k(t-s)}\\,ds.
$$

Integrating over $k$ and swapping the order:

$$
C(t) \\,=\\, \int_0^t I(s)\\,G(t-s)\\,ds,
\qquad
G(\tau) \\,=\\, \int_0^\infty f(k)\\,e^{-k\tau}\\,dk,
$$

where $G$ is the Laplace transform of $f$ — also a smooth 1-D integral, which
can be precomputed once on the output time grid and then convolved with
$I(s)$.

To connect explicitly with the boxed pool-resolved formula in Section 2, set
$I(s) \equiv I$ in the expression above:

$$
\begin{aligned}
C_k(t)
&= f(k)\\!\int_0^t I\\,e^{-k(t-s)}\\,ds \\
&= I\\,f(k)\\!\int_0^t e^{-k(t-s)}\\,ds \\
&= \frac{I\\,f(k)}{k}\\,\bigl(1-e^{-kt}\bigr),
\end{aligned}
$$

which is exactly
$\boxed{\\,C_k(t) = \frac{I\\,f(k)}{k}\\,\bigl(1 - e^{-kt}\bigr)\\,}$.

Integrating this over $k$ then recovers Section 3:

$$
C(t)\\,=\\, I\\!\int_0^\infty \frac{f(k)}{k}\\,\bigl(1 - e^{-kt}\bigr)\\,dk.
$$
