# OpenCVGPU-docker-build

### To create docker image to support OpenCV GPU on Nvidia Tesla K80 on Azure Cloud
1. Create Web-app into "myapp" folder
2. Build container by Dockerfile (you need to install docker)
- 2.1 Build part1 to create docker with enable gpu for K80 series cuda and cudnn version should be support
- 2.2 Build part2 to run app.py inside "myapp" folder
3. You can deploy to "Azure Container Instance" following link : https://learn.microsoft.com/en-us/azure/container-instances/container-instances-quickstart-portal
