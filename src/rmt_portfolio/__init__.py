"""
rmt_portfolio
=============

Random Matrix Theory (RMT) denoising of covariance matrices for
portfolio construction: minimum-variance, risk parity, and hierarchical
risk parity (HRP), with rolling out-of-sample backtesting.

The package is organised in four layers:

- ``rmt_portfolio.data``      : download, clean and cache price panels
- ``rmt_portfolio.rmt``       : eigenvalue spectrum analysis and denoising
- ``rmt_portfolio.portfolio`` : portfolio optimisers / weight allocators
- ``rmt_portfolio.backtest``  : rolling out-of-sample engine and metrics

References
----------
.. [1] Marchenko, V. A. and Pastur, L. A. (1967). Distribution of eigenvalues
       for some sets of random matrices. *Mat. Sb.*, 72(4):507-536.
.. [2] Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). Noise
       dressing of financial correlation matrices. *Phys. Rev. Lett.*, 83:1467.
.. [3] Ledoit, O. and Wolf, M. (2004). Honey, I shrunk the sample covariance
       matrix. *Journal of Portfolio Management*, 30(4):110-119.
.. [4] Ledoit, O. and Wolf, M. (2020). Analytical nonlinear shrinkage of
       large-dimensional covariance matrices. *Annals of Statistics*, 48(5).
.. [5] Lopez de Prado, M. (2016). Building diversification through
       hierarchical clustering. *Journal of Portfolio Management*, 42(4).
.. [6] Bun, J., Bouchaud, J.-P. and Potters, M. (2017). Cleaning large
       correlation matrices: tools from random matrix theory.
       *Physics Reports*, 666:1-109.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Portfolio Research"

from rmt_portfolio.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
