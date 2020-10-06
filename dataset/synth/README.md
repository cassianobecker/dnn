# Running MITK on Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
2. Download [`Dockerfile`](https://github.com/cassianobecker/dnn/blob/master/dataset/mitk/Dockerfile) to a project directory (e.g., `~/Developer/mitk`).
3. Download [`Ubuntu 18.04 executable (with Python support)`](https://github.com/MIC-DKFZ/MITK-Diffusion) to a subdirectory (e.g., `~/Developer/mitk/dnn`).
    - This subdirectory can be persisted across Docker containers.
4. `cd ~/Developer/mitk`
5. `docker build .`
    - The last line will say `Successfully built ID`. `ID` is the ID of the image.
6. `docker run -p 6080:80 -v /project_dir/dnn:/dnn -e USER=dnn -e PASSWORD=odf IMAGE_ID`
    - Replace
        - `/project_dir/dnn` with project directory from Step 2
        - `IMAGE_ID` with `IMAGE ID` of the most recent image from Step 5.
    - Explanation of options (just fyi)
        - `-p 6080:80` forwards port `80` of the container to `6080` of the host (i.e., your computer)
        - `-v /dir_host:/dir_container` mounts `/dir_host` in your computer to `/dir_container` in the container. You can use `-v myvol:/dir_container` to use a named volume managed by Docker.
        - `-e USER... -e PASSWORD` you need a non-root user to use MITK
7. In a browser, go to [`http://127.0.0.1:6080`](http://127.0.0.1:6080). You should see an Ubuntu desktop.
8. In the terminal (at Home > System Tools > LXTerminal),
    - `cd /dnn`
    - `chmod +x ./MITK-Diffusion_ubuntu-18.04_Python.run`
    - `./MITK-Diffusion_ubuntu-18.04_Python.run`
    - In the installer, install to `/dnn/MitkDiffusion`.
        - password for `sudo` is `odf`
    - `cd MitkDiffusion`
    - `./MitkDiffusion.sh`

# Using containers
1. To view containers, `docker ps -a`.
2. To start a container, `docker start container_name`.
    - See `container_name` in Step 1.
    - To attach process to terminal, use option `-a`.
3. To stop a container, `docker stop container_name`.
    - If process attached, just `ctrl-C`.
4. Save files to `/dnn` in the container to access in `/project_dir/dnn` in the host.
5. To create a new container, run Step 6 in **Getting started**.
    - When you use `-v /project_dir/dnn:/dnn`, there is no need to reinstall MITK.

# Resources
1. Dockerfile references [link](https://docs.docker.com/engine/reference/builder)
2. Running, starting, stopping docker containers [link](https://stackoverflow.com/a/41806119)
3. Ubuntu image with VNC [link](https://github.com/fcwu/docker-ubuntu-vnc-desktop)

