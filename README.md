# AE-GNN

It is strongly recommended that the Docker container be utilized, as the requisite environment has already been uploaded via the Nvidia platform.   
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg/tags  

In order to execute the code on a local machine, it is necessary to install the Python library, which is described in below:  

torch_geometric==2.5.3  
torch==2.1.2  
torch-cluster==1.6.3  
torch-scatter==2.1.2  
torch-sparse==0.6.18  
torch-spline-conv==1.2.2  
pyg-lib==0.4.0  

The specific information about our experimentation environment is written in requirement.txt.  


## Example
If you want to evaluate this code(e.g. Cora dataset, 128 layer) just asecute the code as below:  

'''
python main.py --dataset Cora --layer 128 
'''

The particulars of setting are written on 'setting.py'  
