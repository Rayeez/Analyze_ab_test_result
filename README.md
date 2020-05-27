# Analyze_ab_test_result

A/B tests are very commonly performed by data analysts and data scientists. 

For this project, I will be working to understand the results of an A/B test run by an e-commerce website. My goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install 'import file name which used below'
```

## Packages used

```python

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

```

## Conclusion
Based on the work we can conclude that we reject alternative hypothesis because the conversion rate shows no spike in new_page hence we accept Null Hypothesis
