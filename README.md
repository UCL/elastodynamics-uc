# README 
This repository contains the software, data and instructions to reproduce the numerical experiments in the paper 
> Unique continuation for the Lamé system using stabilized finite element methods
> 
> * authors: Erik Burman, Janosch Preuss
> * University College London 

# How to run / install

We describe two options to setup the software for running the experiments. 

* downloading a `docker image` from `Zenodo` or `Docker Hub` which contains all dependencies and tools to run the application,
* or installing everything manually on your own local machine. 

We recommend the first option as it is quick and convenient. The second option provides higher flexibility but may be more complicated. 
Please contact <j.preuss@ucl.ac.uk> if problems occur. 

## Pulling the docker image from Docker Hub 
* Please install the `docker` platform for your distribution as described [here](https://docs.docker.com/get-docker/).
* After installation the `Docker daemon` has to be started. This can either be done on boot or manually. In most Linux 
distributions the command for the latter is either `sudo systemctl start docker` or `sudo service docker start`.
* Pull the docker image using the command `docker pull janosch2888/elastodynamics-uc:v1`. 
* Run the image with `sudo docker run -it janosch2888/elastodynamics-uc:v1 bash `.
* Proceed further as described in [How to reproduce](#repro).

## Downloading the docker image from Zenodo
* For this option the first two steps are the same as above. 
* Assuming that `elastodynamics-repro.tar` is the filename of the downloaded image, please load the image with `sudo docker load < elastodynamics-repro.tar`.
* Run the image with `sudo docker run -it janosch2888/elastodynamics-uc:v1 bash `.
* Proceed further as described in [How to reproduce](#repro).

## Manual installation
You first need to compile a specific version of the finite element software `FEniCSx` on your machine. To this end, please compile `FEniCSx` from source 
as described [here](https://docs.fenicsproject.org/dolfinx/main/python/installation.html#source) and `git checkout` the following commits: 

* xtl: `e697c91e2a3ac571d120d2b093fb3b250d060a7d`
* xtensor: `d1499d900733cd089bd868ca1ca5fdce01e89b97` 
* pugixml: `a0e064336317c9347a91224112af9933598714e9`
* basix: `75e880990af503717f1b894f14376b57057721dd`
* ffcx: `2d080bfae4cce524cca3034b3254709c2ee31c3b` 
* ufl: `677358ab5e9b84f32ff586e6ddb87f74f29a9c76`
* dolfinx: `2a22e39e2ee35d21238e8bd92fac883c03d6fa6b` 

Please also install [h5py](http://www.h5py.org) version 3.6.0 and [meshio](https://github.com/nschloe/meshio) version 5.3.4 (we recommend using `pip` for this purpose). 
For compiling the figures you will also need a recent `latex` distribution installed on your machine.
Now we are ready to clone the repository using 

`git clone https://github.com/UCL/elastodynamics-uc.git` 

and proceed as described in [How to reproduce](#repro).

# <a name="repro"></a> How to reproduce
The `python` scripts for runnings the numerical experiments are located in the folder `scripts`. 
To run an experiments we change to this folder and run the corresponding file.
After execution has finished the produced data will be available in the folder `data`.
To generate the plots as shown in the article from the produced data we change to the folder `plots` 
and compile the corresponding `latex` file. 
Below we decribe the above process for each of the figures in the article in detail.

## Fig. 2 

Change to directory `scripts`. Run 
    
    python3 stab_param_sweep.py
 
Afterwards, three new data files will be available in the folder data called "Convex-Oscillatory-StabSweep-__X__-kk6.dat" where
__X__ describes which stabilization parameter, i.e. "gamma_jump", "gamma_GLS" or "alpha" has been varied. The value of this 
parameter is given in the first column of the respective files. The columns "l2-order__j__" contain the L2-norm error over the 
measurement domain for polynomial order __j__ in [1,2,3]. The columns "cond-order__j__" contain the condition number of the system matrix.
To generate the Fig.2 change to the directory `plots` and execute 

    latexmk -pdf Convex-Oscillatory-Stab-Sweep-L2error-k6.tex

## Fig 3 
Change to directory `scripts`. Run 

    python3 convex_perturb.py 6 "True"

The first argument passed to the script is the wavenumber and the second argument translates into a boolean giving the command 
to compute the condition number. After the script has finished, three files called "Convex-Oscillatory-ill-posed-k6-order__j__.dat"
for polynomial orders __j__ in [1,2,3] will be available in the `data` folder. The column `ndof` in these files contains the number of 
degrees of freedom while the column `cond` gives the corresponding condition number.

To generate the condition number plot switch to directory `plots` and run

    latexmk -pdf Cond-k6-convex.tex


## Fig 4 
Change to directory `scripts`. Run 

    python3 convex_perturb.py 1 "False" 0
    python3 convex_perturb.py 6 "False" 0
    python3 convex_perturb.py 6 "False" 1
    python3 convex_perturb.py 6 "False" 2 

The first argument passed to the script is the wavenumber and the second argument translates into a boolean giving the command skip 
computation of the condition number. The third argument is the parameter theta (which describes the strength of the data perturbation).
After execution the above commands several new files called "Convex-Oscillatory-ill-posed-k1-order__j__-theta__i__.dat" will have been 
created in the folder `data`. Here, the parameter __j__ describes the polynomial order of the FEM and the parameter __i__ the value 
of the perturbation parameter theta. The data in all these files is structured in the same way. The column `L2-error-u-uh-B` contains the 
relative L2-errors in the measurement domain, the column `ndof` gives the number of degrees of freedom and the column `hmax` the mesh width.

To generate the Fig. 4 switch to the directory `plots` and execute 

    latexmk -pdf Convex-Oscillatory-ill-posed-perturb.tex

## Fig 5 
Change to directory `scripts`. Run 

    python3 khscaling.py 

Afterwards, two files called "Convex-Oscillatory-kh-scaling-well-posed-tuned-gamma-CIP-primal-0.dat" and "Convex-Oscillatory-kh-scaling-ill-posed-tuned-gamma-CIP-primal-0.dat" will be available in the `data` folder containing the results for the well-posed and ill-posed case, respectively. 
The first column in these files is the wavenumber `k` and the two other columns `k-norm-order__j__` contain the relative L2-error on the target domain for 
polynomial order __j__ in [1,2]. 

To generate Fig. 5 switch to the `plots` folder and compile 

    latexmk -pdf Convex-Oscillatory-kh-scaling-knorm.tex

## Fig 6 
Change to directory `scripts`. Run 

    python3 khscaling.py "small"
    python3 khscaling.py "tiny" 

Afterwards four files called "Convex-Oscillatory-kh-scaling-__X__-tuned-gamma-CIP-primal-0p0001.dat" and "Convex-Oscillatory-kh-scaling-__X__-tuned-gamma-CIP-primal-0p000001.dat" will have been created in the `data` folder which contain the results for stabilisation parameter 10^-4 (left and middle plot) and 10^-5 (right plot). Here __X__ distinguishes the well-posed and ill-posed case. The data in these files is structured in the same way as in those already described in the reproduction instructions for Fig 5. 

To generate Fig. 6, switch to the `plots` folder and compile 

    latexmk -pdf latexmk -pdf Convex-Oscillatory-kh-scaling-knorm-CIP-primal_pres.tex

## Fig 8 
Change to directory `scripts`. Run 

    python3 split_geom.py

Afterwards, six files of the form "SplitGeom-ill-posed-k1-order__j__-__X__.dat" will be available in the `data` folder. Here __j__ in [1,2,3] 
distinguishes the polynomial degrees and __X__ can is either empty or __X__ "divknown". The latter gives the data for the case in which the 
divergence in Omega is added as an additional constraint into the minimization problem (corresponding to the dashed lines in the figure). The data
in all these files is structured in the same way: the first column "ndof" gives the number of degrees of freedom and the second column the mesh width.
The third column "L2-error-u-uh-B-minus" contains the relative errors in the part of the measurement domain which is contained in the convex hull of 
the data, while the fourth column "L2-error-u-uh-B-plus" gives the relative error in the part of the measurement domain that is located outside of the
convex hull.

To generate Fig.8 switch to the directory `scripts` and run 

    latexmk -pdf Split-domain-ill-posed-k1-div-allOmega.tex

## Fig 9 
Change to directory `scripts`. Run 

    python3 jump_plane.py

Afterwards, six files of the form "jump-mup__a__-mum__b__-ill-posed-k4-order__j__.dat" will be available in the folder `data`. 
Here __a__,__b__ in [1,2] distinguish the value of the shear modulus mup = mu plus and mum = mu minus. As always __j__ stands for 
the polynomial order of the FEM. The data in all these files is structured in the same way. The columns `ndof` and `h` give
the number of degrees of freedom, respectively. The relative errors in Omega minus are given in the column `L2-error-u-uh-B-minus` 
whereas the relative errors in Omega plus can be found in the column `L2-error-u-uh-B-plus`. 

To generate Fig.8 switch to the directory `scripts` and run 

    latexmk -pdf Jump-Split-domain-k4.tex

## Fig 10 
Change to directory `scripts`. Run 

    python3 jump_incl_data_bottom.py 4 1 1
    python3 jump_incl_data_bottom.py 1 1 1
    python3 jump_incl_data_bottom.py 1 2 1
    python3 jump_incl_data_bottom.py 1 .5 1

Here, the first argument passed to the script is the wavenumber __k__. The second argument is the shear in the interior __a__ 
and the third the shear in the exterior __b__. The script generates files of the form "jump-incl-DataBottom-mup__a__-mum__b__-ill-posed-k__k__-order3.dat"
which will be saved to the folder `data`. The data in all these files is stored in the same format. The first column `ndof` contains the number of degrees of 
freedom and the second column `h` the meshwidth. The fourth column `L2-error-u-uh-B-minus` contains the relative L2-error in Bminus and the fith column 
`L2-error-u-uh-B-plus` the relative L2-error in Bplus.

To generate Fig. 10 switch to the folder `plots` and run 

    latexmk -pdf Bottom-data-incl-k4.tex 
    latexmk -pdf Bottom-data-incl-k1.tex



