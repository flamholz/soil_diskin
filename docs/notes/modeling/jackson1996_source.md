# Jackson et al. (1996): source verification

Checked 2026-09-23 against the [publisher page](https://link.springer.com/article/10.1007/BF00333714) and [author-hosted paper](https://jacksonlab2.sites.stanford.edu/sites/g/files/sbiybj20871/files/media/file/oecol96c.pdf). Table 1 and Figure 5 were visually checked in the PDF.

## Published distribution

The cumulative root fraction is $Y(d)=1-\beta^d$, with **depth in centimetres** (Methods, p. 390). Reported coefficients are:

| Population | $\beta$ | Location in paper |
|---|---:|---|
| Global pooled studies | 0.966 | p. 394 |
| Grasses | 0.952 | Figure 5, p. 394 |
| Temperate and tropical trees | 0.970 | Figure 5, pp. 394–395 |
| Shrubs | 0.978 | Figure 5, p. 394 |
| Crops | 0.961 | Table 1, p. 391 |

The global pooling excluded tundra. The tree coefficient excludes boreal forests; crops were comparative examples. Functional groups and biome coefficients differ. The fitted curve primarily describes root biomass, with some other root measures included. [Methods and results](https://jacksonlab2.sites.stanford.edu/sites/g/files/sbiybj20871/files/media/file/oecol96c.pdf#page=2).

## Model translation: our derivation and assumptions

Because $\beta^d=\exp(d\ln\beta)$, this is exactly the existing exponential family with

$$h=-1/\ln\beta.$$

The corresponding $h$ values are 28.909, 20.329, 32.831, 44.953, and 25.138 cm, respectively. Thus the experiment tests published coefficients and vegetation assignments, rather than a different mathematical family.

For a layer spanning $a$–$b$ cm, assign

$$w_{a,b}=\frac{\beta^a-\beta^b}{1-\beta^{100}},\qquad I_{a,b}=\mathrm{NPP}\,w_{a,b}.$$

This conditions the root distribution on 0–100 cm so the ten layer inputs sum to NPP. Truncation and interpreting a biomass distribution as the distribution of carbon input are **our modeling assumptions**, not estimates of input flux from the paper. Missing observations must not trigger renormalization over the remaining layers.

Assigning coarse Balesdent vegetation labels to these groups is also our assumption. Mixed or ambiguous vegetation requires an explicit fallback; generic forest labels cannot establish whether the published temperate/tropical tree population applies.
