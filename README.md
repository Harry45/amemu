# amemu

Accelerating the computation of two-point statistics is an important research area in Cosmology. Here, we use Gaussian Processes to emulate the linear and nonlinear matter power spectrum. The main steps can be summarised as
<br>
1. Generate 500 Latin Hypercube (LH) samples. 
2. Scale the LH samples according to the prior distribution of the cosmological parameters. 
3. Compute the linear matter power spectra at these points. 
4. Train and store the GPs.
5. Make predictions at test points.

The inputs to the emulator are the cosmological parameters and redshift.

## Installing the code
```
conda create --name amemu python=3.9
conda activate amemu
pip install -i https://test.pypi.org/simple/ amemu==0.0.18
```

`pip` does not seem to keep the order. In this case:

```
conda create --name amemu python=3.9
conda activate amemu
git clone https://github.com/Harry45/amemu.git
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ amemu==0.0.18
```

## Example
Use the example notebook: `final.ipynb` as a test.