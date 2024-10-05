# Read Me
## RS Rating
`rs_rating` is an open-source Python package that provides various metrics for evaluating the relative strength of stocks.

- **Documentation:** [yorkjong.github.io/rs_rating/](https://yorkjong.github.io/rs_rating/)
- **Source code:** [github.com/yorkjong/rs_rating](https://github.com/yorkjong/rs_rating)
- **Report issues:** [github.com/yorkjong/rs_rating/issues](https://github.com/yorkjong/rs_rating/issues)

## Getting Started on Colab

1. Click [ibd.ipynb](https://colab.research.google.com/github/yorkjong/rs_rating/blob/main/notebooks/ibd.ipynb) (or [rsm.ipynb](https://colab.research.google.com/github/yorkjong/rs_rating/blob/main/notebooks/rsm.ipynb)) to open it in Colab.
2. Sign in to your Google account if required.
3. Fill in the parameters of the form.
4. Manually click the ► button (which means "start run") to generate tables.
   * You will see the `[ ]` symbol at the beginning of a cell, which will change to the ► button when you hover over it.
   * The `rs_rating` package will be installed automatically if it has not been done yet.
   * After running a cell manually, it will auto-run if you change the selected parameter value.

## Dive Into It on Your Computer

1. **Install `rs_rating` from GitHub:**

    ```bash
    pip install git+https://github.com/yorkjong/rs_rating.git
    ```

2. **Run a rating code:**

    There are three main modules for stock ratings:

    - [ibd_rs.py](https://github.com/yorkjong/rs_rating/blob/main/rs_rating/ibd_rs.py) — IBD RS Rating
    - [ibd_fin.py](https://github.com/yorkjong/rs_rating/blob/main/rs_rating/ibd_fin.py) — IBD Financial Ratings (e.g., EPS Rating, Revenue Rating)
    - [rsm.py](https://github.com/yorkjong/rs_rating/blob/main/rs_rating/rsm.py) — Mansfield RS Rating

    To run a test and generate CSV tables for a given set of stocks, execute the following (use ibd_rs.py as an example):

    ```bash
    python -m rs_rating.ibd_rs
    ```

## Project Background

This project was initially based on the work from [skyte/relative-strength](https://github.com/skyte/relative-strength), which provided the foundation for the IBD RS Rating. The project has since been expanded to include additional rating methods such as IBD's financial ratings and Mansfield RS Rating.

