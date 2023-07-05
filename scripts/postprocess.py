import numpy as np
import math

def cal_equivalent_scatter_network_energy(energy, scatter_factor1, scatter_factor2):
    equivalent_factor = scatter_factor2 / scatter_factor1
    new_energy = energy / equivalent_factor * math.sqrt(equivalent_factor)
    return new_energy

def modified_to_actual_energy(output):
    for arch, info in output['energy_breakdown_pJ'].items():
        if arch != 'MAC':
            # print(arch, 'energy:', output['energy_breakdown_pJ'][arch]['energy'])
            # print('storage_access_energy:', output['energy_breakdown_pJ'][arch]['storage_access_energy'])
            # print('network_energy:', output['energy_breakdown_pJ'][arch]['network_energy'])
            # print('actual_accesses_per_instance:', output['energy_breakdown_pJ'][arch]['actual_accesses_per_instance'])
            # print('accesses_per_instance:', output['energy_breakdown_pJ'][arch]['accesses_per_instance'])

            for i in range(len(output['energy_breakdown_pJ'][arch]['actual_accesses_per_instance'])):
                if output['energy_breakdown_pJ'][arch]['actual_accesses_per_instance'][i] != 0:
                    ratio = output['energy_breakdown_pJ'][arch]['actual_accesses_per_instance'][i] / \
                            output['energy_breakdown_pJ'][arch]['accesses_per_instance'][i]
                    # print(ratio)
                    output['energy_breakdown_pJ'][arch]['storage_access_energy'][i] *= ratio
                    output['energy_breakdown_pJ'][arch]['network_energy'][i] *= ratio
           
            output['energy_breakdown_pJ'][arch]['energy'] = np.nansum(output['energy_breakdown_pJ'][arch]['storage_access_energy']) + \
                    np.nansum(output['energy_breakdown_pJ'][arch]['network_energy']) + output['energy_breakdown_pJ'][arch]['temporal_add_energy'] + \
                    output['energy_breakdown_pJ'][arch]['spatial_add_energy'] + output['energy_breakdown_pJ'][arch]['address_generation_energy']
            # print('storage_access_energy:', output['energy_breakdown_pJ'][arch]['storage_access_energy'])
            # print('network_energy:', output['energy_breakdown_pJ'][arch]['network_energy'])
            # print('energy:', output['energy_breakdown_pJ'][arch]['energy'])
        # print(arch)
    
    output['energy_breakdown_pJ']['LineBuffer']['network_energy'][2] *= output['energy_breakdown_pJ']['Buffer']['actual_accesses_per_instance'][2] / \
                            output['energy_breakdown_pJ']['Buffer']['accesses_per_instance'][2] / math.sqrt(2)
    return output


def calc_psumb_access_rmstc(A_density, B_density):
    # print("calc psumb rmstc ", B_density)
    tile=16
    probs = {}
    
    for n1 in range(tile+1):
        prob_n1 = (B_density)**n1 * (1-B_density)**(tile-n1) * math.comb(tile, n1)
        for n2 in range(n1+1):
            for n3 in range(tile-n1+1):
                prob_2 = (B_density)**(n2+n3) * (1-B_density)**(tile-n2-n3) * math.comb(n1,n2) * math.comb(tile-n1, n3)
                value = n1+n3
                if value in probs:
                    probs[value] += prob_n1*prob_2
                else:
                    probs[value] = prob_n1*prob_2
    ret=0
    for k,v in probs.items():
        ret += k*v
    return ret/tile*A_density

def calc_psumb_access_gamma(A_density, B_density):
    # print("calc psumb gamma ", B_density)
    tile=16     # k
    probs = {}
    
    # both A and B are non-zero
    z_density = A_density * B_density

    # z累加求和时，只要有一个不为0，就需要读取一次z
    ratio = 1 - z_density ** tile
    
    
    return ratio

def modified_network_energy_by_correct_z_ingress(output, A_density, B_density, arch):
    if arch == 'RM-STC':
        ratio = calc_psumb_access_rmstc(A_density, B_density)
    elif arch == 'DS-STC':
        ratio = A_density * B_density
    # output['energy_breakdown_pJ']['Buffer']['storage_access_energy'][2] *= ratio
    output['energy_breakdown_pJ']['Buffer']['network_energy'][2] *= ratio
    return output