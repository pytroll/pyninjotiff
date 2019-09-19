Changelog
=========


v0.2.0 (2019-09-19)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 0.1.0 → 0.2.0. [Martin Raspaud]
- Merge pull request #18 from mraspaud/fix-user-home-path. [Martin
  Raspaud]

  Fix user home path
- Fix travis to improve coverage. [Martin Raspaud]
- Expand the config filename in case ~ is used. [Martin Raspaud]
- Merge pull request #17 from mraspaud/fix-python3-configparser. [Martin
  Raspaud]

  Fix python2-only configparser import
- Fix python2-only configparser import. [Martin Raspaud]
- Merge pull request #16 from mraspaud/fix-tests. [Martin Raspaud]

  Fix test dependencies
- Fix area definitions in the tests. [Martin Raspaud]
- Add pyresample to setup dependencies. [Martin Raspaud]
- Add pyproj to setup dependencies. [Martin Raspaud]
- Fix dask array dependencies. [Martin Raspaud]
- Fix test dependencies. [Martin Raspaud]
- Fix .travis.yml file. [Martin Raspaud]
- Merge pull request #14 from pytroll/feature-python3. [Martin Raspaud]

  Support for python3 and unittests
- Fix P test. [Martin Raspaud]
- Add draft test for P mode. [Martin Raspaud]
- Use _FillValue to mask integer arrays. [Martin Raspaud]
- Add trollimage to test dependencies. [Martin Raspaud]
- Add codecov to travis. [Martin Raspaud]
- Fix channel in vis tests. [Martin Raspaud]
- Fix stickler line length. [Martin Raspaud]
- Fixing style errors. [stickler-ci]
- Add tests. [Martin Raspaud]
- Fix scaling bw images. [Martin Raspaud]
- Fix style. [Martin Raspaud]
- Fixing style errors. [stickler-ci]
- Start supporting python3. [Martin Raspaud]
- Merge pull request #13 from pytroll/add-stickler-config. [Martin
  Raspaud]

  Adding .stickler.yml configuration file
- Adding .stickler.yml. [stickler-ci]
- Merge pull request #9 from pytroll/develop. [David Hoese]

  Merge the develop branch in to master
- Merge pull request #3 from goodsonr/compatability-python3. [Martin
  Raspaud]

  change all occurences of xrange to range for compatability with Python3
- change all occurences of xrange to range for compatability with
  Python3. [ron goodson]
- Add zero seconds option to zero the seconds of the DateID. [Martin
  Raspaud]
- Fix package description. [Martin Raspaud]
- Merge pull request #5 from loreclem/master. [David Hoese]

  WIP|PCW: first attempt to make pyninjotiff xarray compatible.
- Merge pull request #2 from vgiuffrida/master. [lorenzo clementi]

   fix not assigned fill_value and a config file loader issue
- fix not assigned fill_value and config file loader. [root]
- fix fill_value and config file loader. [root]
- Merge pull request #1 from vgiuffrida/master. [lorenzo clementi]

  Add new parameter to configure the ninjotiff config  file to use
- Add ninjotiff configuration file loading. [root]
- Typos corrected and removed is_masked. [cll]
- Bugfix (is_masked computed twice) [cll]
- WIP Improvements here and there. [cll]
- Using finalize instead of fill_or_alpha. [cll]
- It now can handle also RGB images. [cll]
- WIP: first attempt to make pyninjotiff xarray compatible. For the
  moment, only the 'L' case (1 band) has been upgraded. Still to be
  verified. [cll]


v0.1.0 (2017-10-16)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 0.0.1 → 0.1.0. [Martin Raspaud]
- Add housekeeping files. [Martin Raspaud]
- Merge pull request #2 from goodsonr/fix-user-corner-of-pixel-coords.
  [Martin Raspaud]

  Fix - use corner of pixel coords
- Style fixes. [goodsonr]
- Add files via upload. [goodsonr]
- Navigation Fix. [goodsonr]

  Use corners of pixels for navigation instead of center of pixel
  Change to use utility routines to get projection semi-major / semi-minor axis
- Merge pull request #1 from loreclem/master. [Martin Raspaud]

  NinjoTiff minimal example with satpy
- NinjoTiff minimal example with satpy. [lorenzo clementi]
- Bugfix. [Martin Raspaud]
- First commit, copy files from mpop. [Martin Raspaud]
- Initial commit. [Martin Raspaud]



