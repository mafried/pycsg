CSG Library for Python

See sample code in main.py.

# Generator

Generates random CSG trees.

## Usage

See sample code in `test.py`. `n_primitives` determines the number of primitives per tree. Different configuration options can be set by providing a configuration file. See `generator/config.ini`.

`[primitives]`: Turn certain primitives on or off.

`[operations]`: Turn certain operations on or off. Intersection is not implemented.

`[size]`: Choose min and max size per primitive. Works only with even numbers. Please note that the generator was developed to provide a dataset for [pc2csg](https://gitlab2.cip.ifi.lmu.de/bauersv/pc2csg). CSG objects need to fit into a 64x64x64 grid. Therefore min and max sizes might need to be adjusted to guarantee that the desired amount of primitives fits into the grid. The generator was developed to work well with up to 6 primitives per tree.
