
import subprocess
import pickle

cosmos_f = open('../data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

def run_gp_fit(box):
    #run the fit job for this guy 
    subprocess.run('./gp_fit_handler.sh %s'%(box), shell=True)
weird_boxes = ['Box63_1400', 'Box35_1400', 'Box_n50_38_1400', 'Box5_1400']
for box in cosmo_params:
    if(box in weird_boxes):
        continue
    run_gp_fit(box)
    print(box)
