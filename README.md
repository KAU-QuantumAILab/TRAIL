# AE-GNN

We highly recommed to use the docker containner, which environment is already uploaded thanks to Nvidia. 
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg/tags 

If you want to run the code locally, you need to install python library which is written on requirement.txt

The core library is written below:
torch_geometric==2.5.0
torch==2.0.1+cu118
torch-cluster==1.6.3+pt20cu118
torch-scatter==2.1.2+pt20cu118
torch-sparse==0.6.18+pt20cu118
torch-spline-conv==1.2.2+pt20cu118
pyg-lib==0.4.0+pt20cu118