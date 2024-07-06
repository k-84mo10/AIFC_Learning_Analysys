#!bin/bash/
singularity shell --nv --bind /mnt/data-raid/${whoami}/:/mnt/data-raid/${whoami}/ g2_dx.sif
