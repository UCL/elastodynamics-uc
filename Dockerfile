FROM dolfinx/dolfinx:stable

USER root
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y vim 
RUN pip install meshio h5py
RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y --no-install-recommends texlive-full  

WORKDIR /home/app        
RUN git clone https://github.com/UCL/elastodynamics-uc.git

WORKDIR /home/app/elastodynamics-uc

