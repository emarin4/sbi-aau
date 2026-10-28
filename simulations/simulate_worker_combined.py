"Run particle-spray stream simulations for a batch of perturbers (subhalos), transform the resulting particle phase-space into AAU stream coordinates (phi1, phi2, vr, pm1, pm2, dist) and write the result out as an hdf5 graph dataset."

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.append("/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims")

import sims_funcs_refined as sfr
import Nbody_streams.nbody_streams as nbody
import agama
agama.setUnits(mass=1, length=1, velocity=1)
from scipy.stats import rv_histogram


###################################
##Define input/output directories##
###################################
RUN_TAG = "pert_sequential_aau"


perturber_dir = os.path.join("/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/perturbers", RUN_TAG)
output_dir = os.path.join("/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/sims", "sims_"+ RUN_TAG +"_seedchange_200part_ws5foot")
os.makedirs(output_dir, exist_ok=True)

########################
##Define the potential##
########################

BASE_PATH = "/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims"

potMW_path = os.path.join(BASE_PATH, "McMillan17_nora.ini")
potLMC_path = os.path.join(BASE_PATH, "LMC_nora.ini")
accMW_path = os.path.join(BASE_PATH, "accMW")
trajLMC_path = os.path.join(BASE_PATH, "trajLMC")

potMW = agama.Potential(file=potMW_path)
accMW = np.loadtxt(accMW_path)
trajLMC = np.loadtxt(trajLMC_path)

potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
potLMC  = agama.Potential(file=potLMC_path)
potLMCm = agama.Potential(potential=potLMC, center=trajLMC)

potTotal = agama.Potential(potMW, potLMCm, potacc)

##########################################
##Load unpert stream and stripping times##
##########################################

unpert_stream_path = os.path.join(BASE_PATH, "stream_unperturbed_810.pkl")
with open(unpert_stream_path, "rb") as f:
    streamdata = pickle.load(f)
meta = streamdata["metadata"]

#Num. stripping times = Num of particles
distrib_stripping_exact = np.load(os.path.join(BASE_PATH, "distrib_stripping_107.npy"))
#Stripping time distribution to be sampled
distrib_stripping_pool = np.load(os.path.join(BASE_PATH, "distrib_stripping_1016.npy"))

# Fit a continuous distribution to the stripping time pool once
_hist, _bin_edges = np.histogram(distrib_stripping_pool, bins=500, density=True)
stripping_rv = rv_histogram((_hist, _bin_edges))


#####################
##Load S5 footprint##
#####################
_fp = pd.read_csv(os.path.join(BASE_PATH, "aau_footprint.csv"))
_fp = _fp[_fp["group"] == "AAU"]
AAU_FOOTPRINT = _fp[["phi1", "phi2"]].values  # (29, 2)
FIELD_RADIUS = _fp["field_radius_deg"].values 


def simulate_stream_from_pert(pert_dict, meta, distrib_stripping):
    "Run the particle-spray sim for a single perturber, and transform to AAU stream coords"

    N_strip = meta["num_particles"] // 2 + 1
    distrib_stripping = np.sort(stripping_rv.rvs(size=N_strip))
    
    stream_perturb = nbody.fast_sims.create_particle_spray_stream(
        pot_host=potTotal,
        initmass=meta["prog_mass_Msun"],
        scaleradius=meta["prog_scaleradius_kpc"],
        prog_pot_kind='Plummer',
        sat_cen_present=meta["prog_wtoday"],
        num_particles= meta["num_particles"],
        time_end=0.0,
        time_total=meta["Age_stream_Gyr"],
        save_rate=1,
        time_stripping=distrib_stripping, #or use distrib_stripping_exact
        add_perturber=pert_dict,
        verbose=False,
        dissolve_progenitor = True,
        seed = None,
    )
  
    part_xv = stream_perturb["part_xv"]
    coords = sfr.galcen_to_aau_full(part_xv)

    phi1 = coords["phi1"]
    phi2 = coords["phi2"]
    dphi1 = phi1[:, None] - AAU_FOOTPRINT[:, 0]
    dphi2 = phi2[:, None] - AAU_FOOTPRINT[:, 1]
    in_footprint = (np.sqrt(dphi1**2 + dphi2**2) <= FIELD_RADIUS).any(axis=1)
    #print(f"  {in_footprint.sum()} / {len(in_footprint)} particles in footprint ({100*in_footprint.mean():.1f}%)")

    #No S5 footprint
    #feats = np.column_stack([coords[name] for name in sfr.NODE_FEATURE_NAMES])

    #Using S5 footprint
    feats = np.column_stack([coords[name][in_footprint] for name in sfr.NODE_FEATURE_NAMES])

    return feats

########
##Main##
########

PARAM_NAMES = ["mass", "scale_radius", "phi1_impact_today", "impact_time", 
    "impact_param", "v_para", "v_perp", "angle_pos", "angle_vel"]

def main():
    pert_file = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    with open(os.path.join(perturber_dir,pert_file), "rb") as f: 
        pert_batch = pickle.load(f)

    pert_subset = pert_batch[start_idx:end_idx]

    feats_list = []
    theta_rows = []

    for pert_ele in tqdm(pert_subset,desc=f"{pert_file}[{start_idx}:{end_idx}]"):
        pert_inputs = pert_ele["inputs"]
        pert_dict = pert_ele["perturber"]

        feats = simulate_stream_from_pert(pert_dict, meta, stripping_rv)
        feats_list.append(feats)

        theta_rows.append([pert_inputs[name] for name in PARAM_NAMES])

    theta = np.array(theta_rows)

    out_file = os.path.join(output_dir, f"{pert_file.replace('.pkl', '')}_streams_{start_idx:05d}_{end_idx:05d}.h5",)


    sfr.write_graph_dataset(
        path=out_file,
        theta=theta,
        feats_list=feats_list,
        param_names=PARAM_NAMES,
        headers={
            "pert_file": pert_file,
            "start_idx": start_idx,
            "end_idx": end_idx, },)

    print(f"Saved {out_file}  ({len(feats_list)} sims, "
        f"{sum(f.shape[0] for f in feats_list):,} total particles)")


if __name__ == "__main__":
    main()
