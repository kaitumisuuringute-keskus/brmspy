# Google colab example

This example runs in Google Colab environment. Dependency installation takes <50s.

The notebook can be run [HERE](https://colab.research.google.com/drive/1uq6S3exG1zy-JX1lD69ehG4ZHOu9eXY6)

```python
!pip install "brmspy>=0.1.10"
```

```python
from brmspy import brms
```

```python
brms.install_brms(use_prebuilt_binaries=True)
```

```python
epilepsy = brms.get_brms_data("epilepsy")
model = brms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    warmup=500,
    iter=1000,
    chains=4
)
idata = model.idata
```

```python
import arviz as az
summary = az.summary(
    idata,
    hdi_prob=0.95,
    kind="stats",
    round_to=3
)

print("Posterior Summary")
print("="*60)
print(summary)
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig = az.plot_posterior(
    idata,
    var_names=['b_Intercept', 'b_zAge', 'b_zBase', 'b_Trt1', 'b_zBase:Trt1'],
    figsize=(12, 8),
    textsize=10
)
plt.suptitle('Posterior Distributions - Fixed Effects', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```