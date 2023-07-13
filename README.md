# amemu

Accelerating the computation of two-point statistics is an important research area in Cosmology. Here, we use Gaussian Processes to emulate the linear and nonlinear matter power spectrum. The main steps can be summarised as
<br>
1. Generate 500 Latin Hypercube (LH) samples. 
2. Scale the LH samples according to the prior distribution of the cosmological parameters. 
3. Compute the linear matter power spectra at these points. 
4. Train and store the GPs.
5. Make predictions at test points.

The inputs to the emulator are the cosmological parameters and redshift.

```
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install torch
pip install notebook
pip install fast-pt
pip install swig
pip install pyccl
pip install classy
```