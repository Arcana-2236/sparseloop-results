import pandas as pd
import numpy as np
from parse_timeloop_output import parse_timeloop_stats
from postprocess import calc_psumb_access_rmstc, calc_psumb_access_gamma
import math
import os, inspect

sparse_gemm_data = pd.read_csv('sparse gemm.csv')
print(sparse_gemm_data.shape)  

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)

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

for index, row in sparse_gemm_data.iterrows():
    B_density = str(row['B'] / 100)
    A_density = "1.0"

    parse_filename = this_directory + "/../dense_tc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    # print(output['energy_breakdown_pJ'])
    sparse_gemm_data.loc[index, 'DTC'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['RF']['network_energy']))

    parse_filename = this_directory + "/../dstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    sparse_gemm_data.loc[index, 'DS-STC (A Dense)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../rmstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_rmstc(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * (1 - ratio) - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    sparse_gemm_data.loc[index, 'RM-STC (A Dense)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../gamma/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_gamma(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) + np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * ratio / 16 - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    # here /k means modify 基数
    sparse_gemm_data.loc[index, 'Gamma (A Dense)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))


    A_density = "0.5"
    parse_filename = this_directory + "/../nvstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    sparse_gemm_data.loc[index, 'NV-STC'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../dstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    sparse_gemm_data.loc[index, 'DS-STC (A 50%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../rmstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_rmstc(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * (1 - ratio) - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    sparse_gemm_data.loc[index, 'RM-STC (A 50%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../gamma/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_gamma(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) + np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * ratio / 32 - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    # here /k means modify 基数
    sparse_gemm_data.loc[index, 'Gamma (A 50%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))


    A_density = "0.2"
    parse_filename = this_directory + "/../dstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    sparse_gemm_data.loc[index, 'DS-STC (A 20%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../rmstc/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    output = rmstc_random_postprocess(output, 0.2)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_rmstc(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * (1 - ratio) - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    sparse_gemm_data.loc[index, 'RM-STC (A 20%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))

    parse_filename = this_directory + "/../gamma/outputs/" + "B_" + \
                            B_density + "-A_" + A_density + "/outputs/timeloop-model.map+stats.xml"
    output = parse_timeloop_stats(parse_filename)
    output = rmstc_random_postprocess(output, 0.2)
    actual_mac = output['actual_mac']
    ratio = calc_psumb_access_gamma(float(A_density), float(B_density))
    output['energy_pJ'] = output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) + np.nansum(output['energy_breakdown_pJ']['Buffer']['storage_access_energy']) * ratio / 16 - \
        output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] + output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] / math.sqrt(2)
    # here /16 means modify 基数
    sparse_gemm_data.loc[index, 'Gamma (A 20%)'] = (output['energy_pJ'] - np.nansum(output['energy_breakdown_pJ']['Buffer']['network_energy']))


sparse_gemm_data.to_csv("sparse gemm.csv", index=False, sep=',')
