## All-in-one Docker image for Deep Learning
Here are Dockerfiles to get you up and running with a fully functional deep learning machine. It contains all the popular deep learning frameworks with CPU and GPU support (CUDA and cuDNN included). The CPU version should work on Linux, Windows and OS X. The GPU version will, however, only work on Linux machines. See [OS support](#what-operating-systems-are-supported) for details

## Specs
This is what you get out of the box when you create a container with the provided image/Dockerfile:
* Ubuntu 16.04
* [CUDA 9.1](https://developer.nvidia.com/cuda-toolkit) (GPU version only)
* [cuDNN v5](https://developer.nvidia.com/cudnn) (GPU version only)
* [Tensorflow](https://www.tensorflow.org/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Keras](http://keras.io/)
* [Lasagne](http://lasagne.readthedocs.io/en/latest/)
* [Torch](http://torch.ch/) (includes nn, cutorch, cunn and cuDNN bindings)
* [iPython/Jupyter Notebook](http://jupyter.org/) (including iTorch kernel)
* [Numpy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [Pandas](http://pandas.pydata.org/), [Scikit Learn](http://scikit-learn.org/), [Matplotlib](http://matplotlib.org/)
* [OpenCV](http://opencv.org/)
* A few common libraries used for deep learning

## Option 1: Automated Build
1. **GPU Version Only**: Install Nvidia drivers on your machine either from [Nvidia](http://www.nvidia.com/Download/index.aspx?lang=en-us) directly or follow the instructions [here](https://github.com/mthkunze/godeep-setup#nvidia-drivers). Note that you _don't_ have to install CUDA or cuDNN. These are included in the Docker container.

2. **GPU Version Only**: Install nvidia-docker: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker), following the instructions [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation). This will install a replacement for the docker CLI. It takes care of setting up the Nvidia host driver environment inside the Docker containers and a few other things.

**GPU Version**
An automated build for the GPU image could get problems due to timeout restrictions in Docker's automated build process.

#### Option 2: Build the Docker image locally
Alternatively, you can build the images locally. Also, since the GPU version is not available in Docker Hub at the moment, you'll have to follow this if you want to GPU version. Note that this will take an hour or two depending on your machine since it compiles a few libraries from scratch.

```bash
git clone https://github.com/mthkunze/godeep-docker.git
cd godeep-docker
```

**CPU Version**
```bash
docker build -f Dockerfile.cpu .
```

**GPU Version**
```bash
docker build -f Dockerfile.gpu .
```
This will build a Docker image named `godeep-docker` and tagged either `cpu` or `gpu` depending on the tag your specify. Also note that the appropriate `Dockerfile.<architecture>` has to be used.

## Running the Docker image as a Container
Once we've built the image, we have all the frameworks we need installed in it.

**CPU Version**
```bash
docker run -it -p 8888:8888 -p 6006:6006 -v /godeepdocker:/root/godeepdocker mthkunze/godeep-docker:cpu bash
```

**GPU Version**
```bash
nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /godeepdocker:/root/godeepdocker mthkunze/godeep-docker:gpu bash
```
Note the use of `nvidia-docker` rather than just `docker`

| Parameter      | Explanation |
|----------------|-------------|
|`-it`             | This creates an interactive terminal you can use to iteract with your container |
|`-p 8888:8888 -p 6006:6006`    | This exposes the ports inside the container so they can be accessed from the host. The format is `-p <host-port>:<container-port>`. The default iPython Notebook runs on port 8888 and Tensorboard on 6006 |
|`-v /godeepdocker:/root/godeepdocker/` | This shares the folder `/godeepdocker` on your host machine to `/root/godeepdocker/` inside your container. Any data written to this folder by the container will be persistent. You can modify this to anything of the format `-v /local/shared/folder:/shared/folder/in/container/`. See [Docker container persistence](#docker-container-persistence)
|`mthkunze/godeep-docker:cpu`   | This the image that you want to run. The format is `image:tag`. In our case, we use the image `godeep-docker` and tag `gpu` or `cpu` to spin up the appropriate image |
|`bash`       | This provides the default command when the container is started. Even if this was not provided, bash is the default command and just starts a Bash session. You can modify this to be whatever you'd like to be executed when your container starts. For example, you can execute `docker run -it -p 8888:8888 -p 6006:6006 mthkunze/godeep-docker:cpu jupyter notebook`. This will execute the command `jupyter notebook` and starts your Jupyter Notebook for you when the container starts

## Some common scenarios
### Jupyter Notebooks
The container comes pre-installed with iPython and iTorch Notebooks, and you can use these to work with the deep learning frameworks. If you spin up the docker container with `docker-run -p <host-port>:<container-port>` (as shown above in the [instructions](#running-the-docker-image-as-a-container)), you will have access to these ports on your host and can access them at `http://127.0.0.1:<host-port>`. The default iPython server uses port 8888 and Tensorboard uses port 6006. Since we expose both these ports when we run the container, we can access them both from the localhost.

However, you still need to start the Notebook inside the container to be able to access it from the host. You can either do this from the container terminal by executing `jupyter notebook` or you can pass this command in directly while spinning up your container using the `docker run -it -p 8888:8888 -p 6006:6006 mthkunze/godeep-docker:cpu jupyter notebook` CLI. The Jupyter Notebook has both Python (for TensorFlow, Caffe, Theano, Keras, Lasagne) and iTorch (for Torch) kernels.

Note: If you are setting the server on Windows, you will need to first determine the IP address of your Docker container. This command on the Docker command-line provides the IP address
```bash
docker-machine ip default
> <IP-address>
```
```default``` is the name of the container provided by default to the container you will spin.
On obtaining the IP-address, run the docker as per the [instructions](#running-the-docker-image-as-a-container) provided and start the Jupyter pc  as [described above](#jupyter-server). Then accessing ```http://<IP-address>:<host-port>``` on your host's browser should show you the pc.

### Data Sharing
See [Docker container persistence](#docker-container-persistence).
Consider this: You have a script that you've written on your host machine. You want to run this in the container and get the output data (say, a trained model) back into your host. The way to do this is using a [Shared Volume](#docker-container-persistence). By passing in the `-v /godeepdocker/:/root/godeepdocker` to the CLI, we are sharing the folder between the host and the container, with persistence. You could copy your script into `/godeepdocker` folder on the host, execute your script from inside the container (located at `/root/godeepdocker`) and write the results data back to the same folder. This data will be accessible even after you kill the container.

### Performance
Running the DL frameworks as Docker containers should have no performance impact during runtime.

### Docker container persistence
Docker container, added and deleted a few files and then kill the container. The next time you spin up a container using the same image, all your previous changes will be lost and you will be presented with a fresh instance of the image. This is great, because if you mess up your container, you can always kill it and start afresh again. It's bad if you don't know/understand this and forget to save your work before killing the container. There are a couple of ways to work around this:

1. **Commit**: If you make changes to the image itself (say, install a few new libraries), you can commit the changes and settings into a new image. Note that this will create a new image, which will take a few GBs space on your disk. In your next session, you can create a container from this new image. For details on commit, see [Docker's documentaion](https://docs.docker.com/engine/reference/commandline/commit/).

2. **Shared volume**: If you don't make changes to the image itself, but only create data (say, train a new Caffe model), then commiting the image each time is an overkill. In this case, it is easier to persist the data changes to a folder on your host OS using shared volumes. Simple put, the way this works is you share a folder from your host into the container. Any changes made to the contents of this folder from inside the container will persist, even after the container is killed. For more details, see Docker's docs on [Managing data in containers](https://docs.docker.com/engine/userguide/containers/dockervolumes/)

### How do I update/install new libraries?
You can do one of:

1. Modify the `Dockerfile` directly to install new or update your existing libraries. You will need to do a `docker build` after you do this. If you just want to update to a newer version of the DL framework(s), you can pass them as CLI parameter using the --build-arg tag ([see](-v /godeepdocker:/root/godeepdocker) for details). The framework versions are defined at the top of the `Dockerfile`. For example, `docker build -t mthkunze/godeep-docker:cpu -f Dockerfile.cpu --build-arg TENSORFLOW_VERSION=1.0.0 .`

2. You can log in to a container and install the frameworks interactively using the terminal. After you've made sure everything looks good, you can commit the new contains and store it as an image
