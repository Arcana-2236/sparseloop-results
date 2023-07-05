import yaml, os, inspect, shutil, subprocess
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from parse_timeloop_output import parse_timeloop_stats
from postprocess import calc_psumb_access_rmstc
OVERWRITE = True

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
print("this_file_path  ", this_file_path)
this_directory = os.path.join(os.path.dirname(this_file_path), "..")
print("this_directory  ", this_directory)

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

def get_mat_shape(run_type, op, shape):
    if (run_type == 'inference' or run_type == 'fprop'):
        if (op == 'gemm'):
            m = shape[0]
            n = shape[1]
            k = shape[2]
        elif (op == 'conv'):
            m = shape[4]
            n = shape[1] * shape[2] * shape[0]
            k = shape[5] * shape[6] * shape[3]
    elif (run_type == 'wgrad'):
        if (op == 'gemm'):
            m = shape[1]
            n = shape[2]
            k = shape[0]
        elif (op == 'conv'):
            m = shape[3] * shape[5] * shape[6]
            n = shape[4]
            k = shape[1] * shape[2] * shape[0]
    elif (run_type == 'dgrad'):
        if (op == 'gemm'):
            m = shape[2]
            n = shape[0]
            k = shape[1]
        elif (op == 'conv'):
            m = shape[0] * shape[1] * shape[2]
            n = shape[3]
            k = shape[5] * shape[6] * shape[4]
    return m, n, k

def run_dtc(m, n, k, A_density, B_density, job_name):
    # print("m ", m, ",n ", n, ",k ", k)
    dtc_directory = os.path.join(this_directory, "dense_tc")
    # print("dtc  ", dtc_directory)

    # paths to important input specs
    problem_template_path = os.path.join(dtc_directory, "input_specs", "prob.yaml")
    arch_path = os.path.join(dtc_directory, "input_specs", "architecture.yaml")
    component_path = os.path.join(dtc_directory, "input_specs", "compound_components.yaml")
    mapping_path = os.path.join(dtc_directory, "input_specs", "AS-mapping.yaml")
    # sparse_opt_path = os.path.join(dtc_directory,  "input_specs", "sparse-opt.yaml")

    ert_path = os.path.join(dtc_directory,  "input_specs", "ERT.yaml")
    art_path = os.path.join(dtc_directory,  "input_specs", "ART.yaml")
    # print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    # sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(dtc_directory, "outputs")
    stats = {}

    new_arch = deepcopy(arch)
    required_size = (64 + 64) * k + 64 * 64
    # m = 64 & n = 64, mn+mk+nk
    print("require size: ", required_size)
    if required_size > 528384:
        # ['data_storage_depth'] * ['data_storage_width'] / ['datawidth']
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['data_storage_depth'] = math.ceil(required_size / 16)
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['metadata_storage_depth'] = math.ceil(required_size / 32)


    new_problem = deepcopy(problem_template)

    new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
    new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
    new_problem["problem"]["instance"]["M"] = m
    new_problem["problem"]["instance"]["N"] = n
    new_problem["problem"]["instance"]["K"] = k

    new_mapping = deepcopy(mapping)
    for item in new_mapping["mapping"]:
        if item["target"] == "GLB" and item["type"] == "temporal":
            k_factor = k / (8 * 2)
            # print("k_factor", k_factor)
            item["factors"] = "K=%d N=8 M=4" % (k_factor)
        if item["target"] == "DRAM" and item["type"] == "temporal":
            if A_density == 1.0:
                n_factor = n / (8 * 8)
                m_factor = m / (16 * 4)
                # print("n_factor", n_factor)
                # print("m_factor", m_factor)
                item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
                # print("item[factors]", item["factors"])
        
    aggregated_input = {}
    aggregated_input.update(new_arch)
    aggregated_input.update(new_problem)            
    aggregated_input.update(components)
    aggregated_input.update(new_mapping)
    # aggregated_input.update(sparse_opt)
            
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)

    parse_filename = os.path.join(output_base_dir, job_name) + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    return output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['RF']['network_energy'])

def run_nvstc(m, n, k, A_density, B_density, job_name):
    # if A is 100% dense, use the same result as DENSE
    if A_density == 1.0:
        return run_dtc(m, n, k, A_density, B_density, job_name)
    
    nvstc_directory = os.path.join(this_directory, "nvstc")
    # print("nvstc  ", nvstc_directory)

    # paths to important input specs
    problem_template_path = os.path.join(nvstc_directory, "input_specs", "prob.yaml")
    arch_path = os.path.join(nvstc_directory, "input_specs", "architecture.yaml")
    component_path = os.path.join(nvstc_directory, "input_specs", "compound_components.yaml")
    mapping_path = os.path.join(nvstc_directory, "input_specs", "Os-mapping.yaml")
    sparse_opt_path = os.path.join(nvstc_directory,  "input_specs", "sparse-opt.yaml")

    ert_path = os.path.join(nvstc_directory,  "input_specs", "ERT.yaml")
    art_path = os.path.join(nvstc_directory,  "input_specs", "ART.yaml")
    # print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(nvstc_directory, "outputs")
    stats = {}

    new_arch = deepcopy(arch)
    required_size = (64 + 64) * k + 64 * 64
    # m = 64 & n = 64, mn+mk+nk
    print("require size: ", required_size)
    if required_size > 528384:
        # ['data_storage_depth'] * ['data_storage_width'] / ['datawidth']
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['data_storage_depth'] = math.ceil(required_size / 16)
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['metadata_storage_depth'] = math.ceil(required_size / 32)

             
    new_problem = deepcopy(problem_template)

    new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
    new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
    new_problem["problem"]["instance"]["M"] = m
    new_problem["problem"]["instance"]["N"] = n
    new_problem["problem"]["instance"]["K"] = k

    new_mapping = deepcopy(mapping)
    for item in new_mapping["mapping"]:
        if item["target"] == "GLB" and item["type"] == "temporal":
            k_factor = k / (16 * 2)
            # print("k_factor", k_factor)
            item["factors"] = "K=%d N=8 M=4" % (k_factor)
        if item["target"] == "DRAM" and item["type"] == "temporal":
            n_factor = n / (8 * 8)
            m_factor = m / (16 * 4)
            # print("n_factor", n_factor)
            # print("m_factor", m_factor)
            item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
            # print("item[factors]", item["factors"])
        
    aggregated_input = {}
    aggregated_input.update(new_arch)
    aggregated_input.update(new_problem)            
    aggregated_input.update(components)
    aggregated_input.update(new_mapping)
    aggregated_input.update(sparse_opt)
            
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)

    parse_filename = os.path.join(output_base_dir, job_name) + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    return output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy'])

def run_dstc(m, n, k, A_density, B_density, job_name):
    dstc_directory = os.path.join(this_directory, "dstc")
    # print("nvstc  ", nvstc_directory)

    # paths to important input specs
    problem_template_path = os.path.join(dstc_directory, "input_specs", "prob.yaml")
    arch_path = os.path.join(dstc_directory, "input_specs", "architecture.yaml")
    component_path = os.path.join(dstc_directory, "input_specs", "compound_components.yaml")
    mapping_path = os.path.join(dstc_directory, "input_specs", "Os-mapping.yaml")
    sparse_opt_path = os.path.join(dstc_directory,  "input_specs", "sparse-opt.yaml")

    ert_path = os.path.join(dstc_directory,  "input_specs", "ERT.yaml")
    art_path = os.path.join(dstc_directory,  "input_specs", "ART.yaml")
    # print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(dstc_directory, "outputs")
    stats = {}

    new_arch = deepcopy(arch)
    required_size = (64 + 64) * k + 64 * 64
    # m = 64 & n = 64, mn+mk+nk
    print("require size: ", required_size)
    if required_size > 528384:
        # ['data_storage_depth'] * ['data_storage_width'] / ['datawidth']
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['data_storage_depth'] = math.ceil(required_size / 16)
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['metadata_storage_depth'] = math.ceil(required_size / 32)

             
    new_problem = deepcopy(problem_template)

    new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
    new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
    new_problem["problem"]["instance"]["M"] = m
    new_problem["problem"]["instance"]["N"] = n
    new_problem["problem"]["instance"]["K"] = k

    new_mapping = deepcopy(mapping)
    for item in new_mapping["mapping"]:
        if item["target"] == "GLB" and item["type"] == "temporal":
            k_factor = k / 8
            # print("k_factor", k_factor)
            item["factors"] = "K=%d N=2 M=2" % (k_factor)
        if item["target"] == "DRAM" and item["type"] == "temporal":
            n_factor = n / (8 * 8)
            m_factor = m / (16 * 4)
            # print("n_factor", n_factor)
            # print("m_factor", m_factor)
            item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
            # print("item[factors]", item["factors"])
        
    aggregated_input = {}
    aggregated_input.update(new_arch)
    aggregated_input.update(new_problem)            
    aggregated_input.update(components)
    aggregated_input.update(new_mapping)
    aggregated_input.update(sparse_opt)
            
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)

    parse_filename = os.path.join(output_base_dir, job_name) + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    return output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy'])

def run_rmstc(m, n, k, A_density, B_density, job_name):
    rmstc_directory = os.path.join(this_directory, "rmstc")
    # print("nvstc  ", nvstc_directory)

    # paths to important input specs
    problem_template_path = os.path.join(rmstc_directory, "input_specs", "prob.yaml")
    arch_path = os.path.join(rmstc_directory, "input_specs", "architecture.yaml")
    component_path = os.path.join(rmstc_directory, "input_specs", "compound_components.yaml")
    mapping_path = os.path.join(rmstc_directory, "input_specs", "Os-mapping.yaml")
    sparse_opt_path = os.path.join(rmstc_directory,  "input_specs", "sparse-opt.yaml")

    ert_path = os.path.join(rmstc_directory,  "input_specs", "ERT.yaml")
    art_path = os.path.join(rmstc_directory,  "input_specs", "ART.yaml")
    # print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(rmstc_directory, "outputs")
    stats = {}

    new_arch = deepcopy(arch)
    required_size = (64 + 64) * k + 64 * 64
    # m = 64 & n = 64, mn+mk+nk
    print("require size: ", required_size)
    if required_size > 528384:
        # ['data_storage_depth'] * ['data_storage_width'] / ['datawidth']
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['data_storage_depth'] = math.ceil(required_size / 16)
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['metadata_storage_depth'] = math.ceil(required_size / 32)

             
    new_problem = deepcopy(problem_template)

    new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
    new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
    new_problem["problem"]["instance"]["M"] = m
    new_problem["problem"]["instance"]["N"] = n
    new_problem["problem"]["instance"]["K"] = k

    new_mapping = deepcopy(mapping)
    for item in new_mapping["mapping"]:
        
        if item["target"] == "Buffer" and item["type"] == "spatial":
            if A_density == 1.0:
                item["factors"] = "M=1 N=1 K=%d" % 2
            elif A_density == 0.5:
                item["factors"] = "M=1 N=1 K=%d" % 4

        if item["target"] == "GLB" and item["type"] == "temporal":
            if A_density == 1.0:
                k_factor = k / 16
            elif A_density == 0.5:
                k_factor = k / 32
            # print("k_factor", k_factor)
            item["factors"] = "K=%d N=4 M=4" % (k_factor)
        if item["target"] == "DRAM" and item["type"] == "temporal":
            n_factor = n / (8 * 8)
            m_factor = m / (16 * 4)
            # print("n_factor", n_factor)
            # print("m_factor", m_factor)
            item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
            # print("item[factors]", item["factors"])
        
    aggregated_input = {}
    aggregated_input.update(new_arch)
    aggregated_input.update(new_problem)            
    aggregated_input.update(components)
    aggregated_input.update(new_mapping)
    aggregated_input.update(sparse_opt)
            
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)

    parse_filename = os.path.join(output_base_dir, job_name) + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)

    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_rmstc(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * (1 - ratio) - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)

    return output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy'])

def run_rmstc_random(m, n, k, A_density, B_density, job_name):
    rmstc_directory = os.path.join(this_directory, "rmstc")

    # paths to important input specs
    problem_template_path = os.path.join(rmstc_directory, "input_specs", "prob.yaml")
    arch_path = os.path.join(rmstc_directory, "input_specs", "architecture.yaml")
    component_path = os.path.join(rmstc_directory, "input_specs", "compound_components.yaml")
    mapping_path = os.path.join(rmstc_directory, "input_specs", "Os-mapping.yaml")
    sparse_opt_path = os.path.join(rmstc_directory,  "input_specs", "sparse-opt.yaml")

    ert_path = os.path.join(rmstc_directory,  "input_specs", "ERT.yaml")
    art_path = os.path.join(rmstc_directory,  "input_specs", "ART.yaml")
    # print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(rmstc_directory, "outputs")
    stats = {}

    new_arch = deepcopy(arch)
    required_size = (64 + 64) * k + 64 * 64
    # m = 64 & n = 64, mn+mk+nk
    print("require size: ", required_size)
    if required_size > 528384:
        # ['data_storage_depth'] * ['data_storage_width'] / ['datawidth']
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['data_storage_depth'] = math.ceil(required_size / 16)
        new_arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['metadata_storage_depth'] = math.ceil(required_size / 32)

             
    new_problem = deepcopy(problem_template)

    new_problem["problem"]["instance"]["densities"]["B"]["density"] = B_density
    # all set as 16 choose 2
    if A_density <= 0.125:
        new_problem["problem"]["instance"]["densities"]["A"]["density"] = A_density
    else:
        new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.125
    new_problem["problem"]["instance"]["M"] = m
    new_problem["problem"]["instance"]["N"] = n
    new_problem["problem"]["instance"]["K"] = k

    new_mapping = deepcopy(mapping)
    for item in new_mapping["mapping"]:
        
        if item["target"] == "Buffer" and item["type"] == "spatial":
                item["factors"] = "M=1 N=1 K=16"

        if item["target"] == "Buffer" and item["type"] == "temporal":
                item["factors"] = "K=1 N=1 M=1"

        if item["target"] == "GLB" and item["type"] == "temporal":
            k_factor = k / 16
            # print("k_factor", k_factor)
            item["factors"] = "K=%d N=4 M=4" % (k_factor)
        if item["target"] == "DRAM" and item["type"] == "temporal":
            n_factor = n / (8 * 8)
            m_factor = m / (16 * 4)
            # print("n_factor", n_factor)
            # print("m_factor", m_factor)
            item["factors"] = "K=1 N=%d M=%d" % (n_factor, m_factor)
            # print("item[factors]", item["factors"])
        
    aggregated_input = {}
    aggregated_input.update(new_arch)
    aggregated_input.update(new_problem)            
    aggregated_input.update(components)
    aggregated_input.update(new_mapping)
    aggregated_input.update(sparse_opt)
            
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)

    parse_filename = os.path.join(output_base_dir, job_name) + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)

    if A_density > 0.125:
        # print(output)
        output = rmstc_random_postprocess(output, A_density)
    # print('A_density: ', A_density, '\n\n\n')

    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_rmstc(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * (1 - ratio) - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    
    return output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy'])

def rmstc_random_postprocess(output, A_density):
    # postprocess to deal with unstructure random sparsity_A
    result = output
    # DRAM-A access & network
    result['energy_breakdown_pJ']['DRAM']['storage_access_energy'][0] = output['energy_breakdown_pJ']['DRAM']['storage_access_energy'][0] * A_density / 0.125
    result['energy_breakdown_pJ']['DRAM']['network_energy'][0] = output['energy_breakdown_pJ']['DRAM']['network_energy'][0] * A_density / 0.125
    # GLB-A access & network
    result['energy_breakdown_pJ']['GLB']['storage_access_energy'][0] = output['energy_breakdown_pJ']['GLB']['storage_access_energy'][0] * A_density / 0.125
    result['energy_breakdown_pJ']['GLB']['network_energy'][0] = output['energy_breakdown_pJ']['GLB']['network_energy'][0] * A_density / 0.125
    
    x_step = math.ceil(A_density / 0.125)
    # LB-A access & network
    result['energy_breakdown_pJ']['LineBuffer']['storage_access_energy'][0] = output['energy_breakdown_pJ']['LineBuffer']['storage_access_energy'][0] * x_step
    result['energy_breakdown_pJ']['LineBuffer']['network_energy'][0] = output['energy_breakdown_pJ']['LineBuffer']['network_energy'][0] * x_step
    # Buf-Z access & network
    result['energy_breakdown_pJ']['Buffer']['storage_access_energy'][2] = output['energy_breakdown_pJ']['Buffer']['storage_access_energy'][2] * x_step
    result['energy_breakdown_pJ']['Buffer']['network_energy'][2] = output['energy_breakdown_pJ']['Buffer']['network_energy'][2] * x_step
    # MAC access
    result['energy_breakdown_pJ']['MAC']['energy'] = output['energy_breakdown_pJ']['MAC']['energy'] * x_step

    # LB-B access & network
    result['energy_breakdown_pJ']['LineBuffer']['storage_access_energy'][1] = output['energy_breakdown_pJ']['LineBuffer']['energy_per_access_per_instance'][0] * output['energy_breakdown_pJ']['GLB']['actual_accesses_per_instance'][1] * x_step
    result['energy_breakdown_pJ']['LineBuffer']['network_energy'][1] = output['energy_breakdown_pJ']['LineBuffer']['network_energy'][0] / output['energy_breakdown_pJ']['LineBuffer']['instances'][0] \
        / output['energy_breakdown_pJ']['LineBuffer']['accesses_per_instance'][0] * output['energy_breakdown_pJ']['GLB']['actual_accesses_per_instance'][1] * x_step

    for component in ['DRAM', 'GLB', 'Buffer', 'LineBuffer']:
        result['energy_breakdown_pJ'][component]['energy'] = np.nansum(result['energy_breakdown_pJ'][component]['storage_access_energy']) + np.nansum(result['energy_breakdown_pJ'][component]['network_energy']) + result['energy_breakdown_pJ'][component]['temporal_add_energy'] + result['energy_breakdown_pJ'][component]['spatial_add_energy'] + result['energy_breakdown_pJ'][component]['address_generation_energy']

    result['energy_pJ'] = sum([value['energy'] for key, value in result['energy_breakdown_pJ'].items()])

    return result


def run_dense_inference(workload_dict, workload_data):
    label = workload_dict['label']
    op = workload_dict['op']
    shape = workload_dict['shape']
    densities = workload_dict['densities']

    m, n, k = get_mat_shape('inference', op, shape)

    A_De = 1.0
    A_Sp = 0.5
    B_De = 1.0
    B_Sp = densities[0] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Inference DTC'] = run_dtc(m, n, k, A_De, B_De, label)
    
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Inference NV-STC'] = run_nvstc(m, n, k, A_De, B_De, label)
    
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Inference DS-STC'] = run_dstc(m, n, k, A_De, B_Sp, label)
    
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Inference RM-STC'] = run_rmstc(m, n, k, A_De, B_Sp, label)
    
    # A dense/B sparse
    # here rmstc+ performs the same as rmstc
    workload_data.loc[workload_data['Label'] == label, 'Dense Inference RM-STC+'] = run_rmstc(m, n, k, A_De, B_Sp, label)

    # print(workload_data)
    # print(a)
    # print(m, n, k)

def run_sparse_inference(workload_dict, workload_data):
    label = workload_dict['label']
    op = workload_dict['op']
    shape = workload_dict['shape']
    densities = workload_dict['densities']

    m, n, k = get_mat_shape('inference', op, shape)

    A_De = 1.0
    A_Sp = 0.5
    B_De = 1.0
    B_Sp = densities[1] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Inference DTC'] = run_dtc(m, n, k, A_De, B_De, label)
    
    # A sparse/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Inference NV-STC'] = run_nvstc(m, n, k, A_Sp, B_De, label)
    
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Inference DS-STC'] = run_dstc(m, n, k, A_Sp, B_Sp, label)
    
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Inference RM-STC'] = run_rmstc(m, n, k, A_Sp, B_Sp, label)

    # A dense/B sparse
    # here rmstc+ performs the same as rmstc
    workload_data.loc[workload_data['Label'] == label, 'Sparse Inference RM-STC+'] = run_rmstc(m, n, k, A_Sp, B_Sp, label)

def run_dense_training(workload_dict, workload_data):
    label = workload_dict['label']
    op = workload_dict['op']
    shape = workload_dict['shape']
    densities = workload_dict['densities']

    # step 1: fprop
    m, n, k = get_mat_shape('fprop', op, shape)

    A_De = 1.0
    A_Sp = 0.5
    B_De = 1.0
    B_Sp = densities[2] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DTC'] = run_dtc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training NV-STC'] = run_nvstc(m, n, k, A_De, B_De, label)   
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DS-STC'] = run_dstc(m, n, k, A_De, B_Sp, label)    
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC'] = run_rmstc(m, n, k, A_De, B_Sp, label)
    # A dense/B sparse
    # here rmstc+ performs the same as rmstc
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC+'] = run_rmstc(m, n, k, A_De, B_Sp, label)

    # step 2: wgrad
    m, n, k = get_mat_shape('wgrad', op, shape)

    A_De = 1.0
    A_Sp = densities[2] / 100
    B_De = 1.0
    B_Sp = densities[3] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DTC'] += run_dtc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training NV-STC'] += run_nvstc(m, n, k, A_De, B_De, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DS-STC'] += run_dstc(m, n, k, A_De, B_Sp, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC'] += run_rmstc(m, n, k, A_De, B_Sp, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC+'] += run_rmstc_random(m, n, k, A_Sp, B_Sp, label)

    # step 3: dgrad
    m, n, k = get_mat_shape('dgrad', op, shape)

    A_De = 1.0
    A_Sp = densities[3] / 100
    B_De = 1.0
    B_Sp = 0.5

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DTC'] += run_dtc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training NV-STC'] += run_nvstc(m, n, k, A_De, B_De, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training DS-STC'] += run_dstc(m, n, k, A_De, B_De, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC'] += run_rmstc(m, n, k, A_De, B_De, label)
    # A sparse/B dense
    workload_data.loc[workload_data['Label'] == label, 'Dense Training RM-STC+'] += run_rmstc_random(m, n, k, A_Sp, B_De, label)

def run_sparse_training(workload_dict, workload_data):
    label = workload_dict['label']
    op = workload_dict['op']
    shape = workload_dict['shape']
    densities = workload_dict['densities']

    # step 1: fprop
    m, n, k = get_mat_shape('fprop', op, shape)

    A_De = 1.0
    A_Sp = 0.5
    B_De = 1.0
    B_Sp = densities[4] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DTC'] = run_dtc(m, n, k, A_De, B_De, label)
    # A sparse/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training NV-STC'] = run_nvstc(m, n, k, A_Sp, B_De, label)   
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DS-STC'] = run_dstc(m, n, k, A_Sp, B_Sp, label)    
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC'] = run_rmstc(m, n, k, A_Sp, B_Sp, label)
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC+'] = run_rmstc(m, n, k, A_Sp, B_Sp, label)


    # step 2: wgrad
    m, n, k = get_mat_shape('wgrad', op, shape)

    A_De = 1.0
    A_Sp = densities[4] / 100
    B_De = 1.0
    B_Sp = densities[5] / 100

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DTC'] += run_dtc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training NV-STC'] += run_nvstc(m, n, k, A_De, B_De, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DS-STC'] += run_dstc(m, n, k, A_De, B_Sp, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC'] += run_rmstc(m, n, k, A_De, B_Sp, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC+'] += run_rmstc_random(m, n, k, A_Sp, B_Sp, label)

    # step 3: dgrad
    m, n, k = get_mat_shape('dgrad', op, shape)

    A_De = 1.0
    A_Sp = densities[5] / 100
    B_De = 1.0
    B_Sp = 0.5

    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DTC'] += run_dtc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training NV-STC'] += run_nvstc(m, n, k, A_De, B_De, label)
    # A dense/B dense
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training DS-STC'] += run_dstc(m, n, k, A_De, B_De, label)
    # A dense/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC'] += run_rmstc(m, n, k, A_De, B_Sp, label)
    # A sparse/B sparse
    workload_data.loc[workload_data['Label'] == label, 'Sparse Training RM-STC+'] += run_rmstc_random(m, n, k, A_Sp, B_Sp, label)


def run_dnn_workload(workload_dict, workload_data):
    run_dense_inference(workload_dict, workload_data)
    run_sparse_inference(workload_dict, workload_data)
    run_dense_training(workload_dict, workload_data)
    run_sparse_training(workload_dict, workload_data)

# read yaml file configuration
yaml_file = "DNN workload.yaml"
with open(yaml_file, "r") as f:
    yaml_data = yaml.safe_load(f)

workload_data = pd.read_csv('DNN workloads.csv')
print('shape:', workload_data.shape)
# process each DNN workload
for it in yaml_data:
    run_dnn_workload(it, workload_data)

    # print(shape[1])
# print(yaml_data[0])
print(workload_data)
os.chdir(os.path.dirname(this_file_path))   # back to current working directory
workload_data.to_csv('DNN workloads.csv', index=False, sep=',')