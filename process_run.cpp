#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <time.h>

using namespace std;


int main(int argc, char** argv) {
    
    string curr_run_fname = "/oak/stanford/orgs/kipac/aemulus/aemulus_nu/" + string(argv[1]) + "/";
    cout << curr_run_fname << endl;
    string rockstar_dir = curr_run_fname + "output/rockstar/";
        
    ifstream f (rockstar_dir + "savelist.txt", ifstream::in);
    vector<string> savelist;
    string line;
    while (getline(f, line)) {
        stringstream ss(line);
        savelist.push_back(ss.str());
    }
    f.close();

    int N_snapshots = savelist.size();
    

    vector<vector<double>> mass_data(N_snapshots);
    
    string oup_fname = "/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/" + string(argv[1]) + "_M200b";
    string oup_pos_fname = "/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/" + string(argv[1]) + "_pos";


    ofstream oup_f (oup_fname);
    ofstream oup_pos_f (oup_pos_fname);
    clock_t MtStart = clock();
    for (int i = 0; i < N_snapshots; i++) {
        clock_t tStart = clock();

        cout << "Currently on " << i+1 << " of " << N_snapshots << endl;
        ifstream f(rockstar_dir + "out_" + to_string(i) + ".list");
        getline(f, line);
        stringstream ss(line.substr(1));
        vector<string> cols;
        string col;
        int i_M200b = -1;
        int i_X = -1;
        int i_Y = -1;
        int i_Z = -1; 

        int i_tmp = 0;
        while (ss >> col) {
            cols.push_back(col);
            if(col=="M200b"){
                i_M200b = i_tmp;
            }
            if(col=="X"){
                i_X = i_tmp;
            }
            if(col=="Y"){
                i_Y = i_tmp;
            }
            if(col=="Z"){
                i_Z = i_tmp;
            }
            i_tmp ++;
        }
        while (getline(f, line)) {
            if (line[0] == '#') {
                continue;
            }
            stringstream ss(line);
            
            for (int i = 0; i<cols.size(); i++) {
                string value;
                ss >> value;
                if(i==i_M200b){
                    oup_f << value << " ";
                }
                if(i==i_X){
                    oup_pos_f << value << " ";
                }
                if(i==i_Y){
                    oup_pos_f << value << " ";
                }
                if(i==i_Z){
                    oup_pos_f << value << " ";
                }
            }
            oup_pos_f << ", ";
        }
        oup_f << endl; 
        oup_pos_f << endl;
        printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        cout << endl;
        f.close();
        
    }
    oup_f.close();
    printf("Total time taken: %.2fs\n", (double)(clock() - MtStart)/CLOCKS_PER_SEC);
    cout << endl;
}
