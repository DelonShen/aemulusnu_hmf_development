
import subprocess
import pickle

cosmos_f = open('../data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

def run_gp_fit(box):
    #run the fit job for this guy 
#     subprocess.run('./gp_fit_handler.sh %s'%(box), shell=True)
    cmd = 'nohup python -u gp_fit.py %s &> logs/2024-08-14-try2_gp_fit_%s'%(box, box)
    print(cmd)
    subprocess.run(cmd, shell=True)

weird_boxes = []
for box in cosmo_params:
    if(box in weird_boxes):
        continue
    print(box)
    run_gp_fit(box)
