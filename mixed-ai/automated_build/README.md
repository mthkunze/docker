### Please do not use these `Dockerfiles` directly

These files are intended for Docker Hub's automated build process. 
The automated build for `Dockerfile.gpu` could fail due to timeout restrictions.
In case of this, `Dockerfile.gpu` you have to split the Dockerfile into two parts: `Dockerfile.gpu1` and `Dockerfile.gpu2`. 
These files are used to build the final `godeep-docker:gpu` that you can pull directly from Docker Hub using `docker pull mthkunze/godeep-docker:gpu`.
Link your prefered Dockerfile for automated builds...
