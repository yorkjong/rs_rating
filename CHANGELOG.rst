Change Log
==========

1.5
----------------
* refined relative_strength_3m
* Extracted groupby_industry, append_ratings from ibd_rs.rankings function
* applied groupby_industry, append_ratings to generate and display
  industry_df

1.0 [2024-10-04]
----------------
* Extracted from vistock project (https://github.com/yorkjong/vistock.git)
* Removed unnecessary dependency
* Added initial Sphinx documentation setup
  * `mkdir docs`
  * `cd docs`
  * `sphinx-quickstart`
* Generated API documentation for rs_rating using sphinx-apidoc
  * `sphinx-apidoc -o docs rs_rating`
