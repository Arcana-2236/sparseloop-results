import yaml, inspect, os, sys, subprocess, pprint, shutil
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

OVERWRITE = True

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)

# paths to important input specs
problem_template_path = os.path.join(this_directory, "..", "input_specs", "prob.yaml")
arch_path = os.path.join(this_directory, "..", "input_specs", "architecture.yaml")
component_path = os.path.join(this_directory, "..", "input_specs", "compound_components.yaml")
mapping_path = os.path.join(this_directory, "..", "input_specs", "Os-mapping.yaml")
sparse_opt_path = os.path.join(this_directory, "..",  "input_specs", "sparse-opt.yaml")



def run_timeloop(job_name, input_dict, base_dir, ert_path, art_path):
    output_dir = os.path.join(base_dir +"/outputs")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print("Found existing results: ", output_dir)
            return
        else:
            print("Found and overwrite existing results: ", output_dir)

    # reuse generated ERT and ART files
    shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    ert_file_path = os.path.join(base_dir, "ERT.yaml")
    art_file_path = os.path.join(base_dir, "ART.yaml")
    logfile_path = os.path.join(output_dir, "timeloop.log")
    
    yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
    os.chdir(output_dir)
    subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    p.communicate(timeout=None) 

def main():
    
    ert_path = os.path.join(this_directory, "..",  "input_specs", "ERT.yaml")
    art_path = os.path.join(this_directory, "..",  "input_specs", "ART.yaml")
    print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "..", "outputs")
    stats = {}
    
    for A_density in [0.2]:
        for B_density in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            
            if A_density not in stats:
                stats[A_density] = {}
            
            new_problem = deepcopy(problem_template)

            new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
            if A_density <= 0.125:
                new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
            else:
                new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.125

            new_mapping = deepcopy(mapping)
            for item in new_mapping["mapping"]:
        
                if item["target"] == "Buffer" and item["type"] == "spatial":
                    item["factors"] = "M=1 N=1 K=16"

                if item["target"] == "Buffer" and item["type"] == "temporal":
                    item["factors"] = "K=1 N=1 M=1"

                if item["target"] == "GLB" and item["type"] == "temporal":
                    k_factor = 2048 / 16
                    # print("k_factor", k_factor)
                    item["factors"] = "K=%d N=4 M=4" % (k_factor)
                if item["target"] == "DRAM" and item["type"] == "temporal":
                    n_factor = 2048 / (8 * 8)
                    m_factor = 2048 / (16 * 4)
                    # print("n_factor", n_factor)
                    # print("m_factor", m_factor)
                    item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
                    # print("item[factors]", item["factors"])
        
            aggregated_input = {}
            aggregated_input.update(arch)
            aggregated_input.update(new_problem)            
            aggregated_input.update(components)
            aggregated_input.update(new_mapping)
            aggregated_input.update(sparse_opt)
            
            job_name  = "B_" + str(B_density) + "-A_" + str(A_density)
            
            run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)
            
if __name__ == "__main__":
    main()
        
