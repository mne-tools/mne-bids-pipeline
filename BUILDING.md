# Building a release

* Tag a new release with `git` if necessary.
* Create `sdist` distribution:

  ```shell
  pip install -q build
  python -m build  # will build sdist and wheel
  ```
