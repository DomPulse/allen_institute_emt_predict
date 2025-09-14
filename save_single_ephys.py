import arbor as A
from arbor import units as U
import sys
from arbor import mechanism as mech
import json
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

catalogue = A.allen_catalogue()
#print(list(catalogue.keys()))

folder = 'lee_lj'
cell_name = 'Het-mPFC-20X-4-A.CNG'
morph_path = r'/mnt/f/arbor_ubuntu/neuronal_model_491766131/reconstruction.swc'
morph_path = rf'/mnt/f/arbor_ubuntu/10k_mouse_pyr_morph/{folder}/CNG version/{cell_name}.swc'
fit_params_path = r'/mnt/f/arbor_ubuntu/neuronal_model_491766131/fit_parameters.json'
fit_params_path = r'/mnt/f/arbor_ubuntu/10k_randomized_jsons/random_genome_91.json'

labels = A.label_dict().add_swc_tags()
labels["midpoint"] = "(location 0 0.5)"

def load_allen_fit(fit):
    with open(fit) as fd:
        fit = json.load(fd)

    # cable parameters convenience class
    @dataclass
    class parameters:
        cm: Optional[U.quantity] = None
        temp: Optional[U.quantity] = None
        Vm: Optional[U.quantity] = None
        rL: Optional[U.quantity] = None

    param = defaultdict(parameters)
    mechs = defaultdict(dict)
    for block in fit["genome"]:
        mech = block["mechanism"] or "pas"
        region = block["section"]
        name = block["name"]
        value = float(block["value"])

        #print(mech, region, name, value)
        if name.endswith("_" + mech):
            name = name[: -(len(mech) + 1)]
        elif mech == "pas":
            # transform names and values
            if name == "cm":
                param[region].cm = value * U.uF / U.cm2
            elif name == "Ra":
                param[region].rL = value * U.Ohm * U.cm
            elif name == "Vm":
                param[region].Vm = value * U.mV
            elif name == "celsius":
                param[region].temp = value * U.Celsius
            else:
                raise Exception(f"Unknown key: {name}")
            continue
        else:
            raise Exception(f"Illegal combination {mech} {name}")
        mechs[(region, mech)][name] = value


    regs = list(param.items())
    mechs = [(r, m, vs) for (r, m), vs in mechs.items()]

    default = parameters(
        temp=float(fit["conditions"][0]["celsius"]) * U.Celsius,
        Vm=float(fit["conditions"][0]["v_init"]) * U.mV,
        rL=float(fit["passive"][0]["ra"]) * U.Ohm * U.cm,
    )

    ions = []
    for kv in fit["conditions"][0]["erev"]:
        region = kv["section"]
        for k, v in kv.items():
            if k == "section":
                continue
            ion = k[1:]
            ions.append((region, ion, float(v) * U.mV))

    return default, regs, ions, mechs, fit["fitting"][0]["junction_potential"]

def make_cell(swc, fit):
    # (1) Load the swc file passed into this function
    morphology = A.load_swc_neuron(swc).morphology

    # (2) Label the region tags found in the swc with the names used in the parameter fit file.
    # In addition, label the midpoint of the soma.
    labels = A.label_dict().add_swc_tags()
    labels["midpoint"] = "(location 0 0.5)"

    # (3) A function that parses the Allen parameter fit file into components
    dflt, regions, ions, mechanisms, offset = load_allen_fit(fit)

    # (4) Instantiate an empty decor.
    decor = A.decor()

    # (5) assign global electro-physiology parameters
    decor.set_property(
        tempK=dflt.temp,
        Vm=dflt.Vm,
        cm=1 * U.uF/U.cm2,
        rL=dflt.rL,
    )

    # (6) override regional electro-physiology parameters
    for region, vs in regions:

        decor.paint(f'"{region}"', rL=vs.rL)
        decor.paint(f'"{region}"', cm=vs.cm)

    # (7) set reversal potentials
    for region, ion, e in ions:
        decor.paint(f'"{region}"', ion=ion, rev_pot=e)
    decor.set_ion("ca", int_con=5e-5 * U.mM, ext_con=2.0 * U.mM, method="nernst/x=ca")

    # (8) assign ion dynamics
    for region, mech, values in mechanisms:
        nm = mech
        vs = {}
        sp = "/"
        for k, v in values.items():
            if mech == "pas" and k == "e":
                #print(region, mech, values)
                nm = f"{nm}{sp}{k}={v}"
                sp = ","
            else:
                vs[k] = v
        decor.paint(f'"{region}"', A.density(A.mechanism(nm, vs)))
        
    # (9) attach stimulus and detector
    decor.place('"midpoint"', A.iclamp(0.5 * U.s, 1.2 * U.s, -200 * U.pA), "ic0")
    decor.place('"midpoint"', A.iclamp(2.2 * U.s, 1.2 * U.s, -100 * U.pA), "ic1")
    decor.place('"midpoint"', A.iclamp(3.9 * U.s, 1.2 * U.s, -50 * U.pA), "ic2")
    decor.place('"midpoint"', A.iclamp(5.6 * U.s, 1.2 * U.s, 50 * U.pA), "ic3")
    decor.place('"midpoint"', A.iclamp(7.3 * U.s, 1.2 * U.s, 100 * U.pA), "ic4")
    decor.place('"midpoint"', A.iclamp(9.0 * U.s, 1.2 * U.s, 150 * U.pA), "ic5")
    decor.place('"midpoint"', A.iclamp(10.7 * U.s, 1.2 * U.s, 200 * U.pA), "ic6")
    decor.place('"midpoint"', A.threshold_detector(-20 * U.mV), "sd")

    # (10) discretisation strategy: max compartment length
    cvp = A.cv_policy_max_extent(20)

    #print(decor.paintings())   

    # (11) Create cell
    return A.cable_cell(morphology, decor, labels, cvp), offset
# ---- worker function ----
def run_simulation(task):
    morph_path, fit_params_path, folder_name, cell_name, counter = task
    try:
        # Build cell
        cell, offset = make_cell(morph_path, fit_params_path)
        model = A.single_cell_model(cell)

        # Probe
        model.probe("voltage", '"midpoint"', "Um", frequency=1 / (5 * U.us))

        # Catalogue
        model.properties.catalogue.extend(A.allen_catalogue())

        # Run simulation
        model.run(tfinal=12.5 * U.s, dt=5 * U.us)

        # Save results
        df = pd.DataFrame({
            "Simulator": "Arbor",
            "t/ms": model.traces[0].time,
            "U/mV": model.traces[0].value,
        })

        last_bit = fit_params_path.split('/')[-1].split('.')[0]
        csv_path = f"/mnt/f/arbor_ubuntu/{folder_name}_{cell_name}_{last_bit}.csv"
        df.to_csv(csv_path, index=False)

        print(f"[OK] saved {folder_name} {cell_name} {counter}")
    except Exception as e:
        print(f"[FAIL] {folder_name} {cell_name} {counter} -> {e}")


# ---- main section ----
if __name__ == "__main__":
    folder = r"/mnt/f/arbor_ubuntu/10k_mouse_pyr_morph"
    tasks = []
    template_cell = r'/mnt/f/arbor_ubuntu/10k_mouse_pyr_morph/kimura/CNG version/P7-control-for-TRAK2-MBD-14-003.CNG.swc'

    fit_params_path = f"/mnt/d/Neuro_Sci/morph_ephys_trans_stuff/cell_params_updated.json"
    #fit_params_path = f"/mnt/f/arbor_ubuntu/10k_randomized_jsons/random_genome_19.json"
    run_simulation((morph_path, fit_params_path, 'kimura', 'P7-control-for-TRAK2-MBD-14-003.CNG.swc', 0))
