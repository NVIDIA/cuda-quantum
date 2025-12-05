# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import cudaq
from cudaq.operators import *
from cudaq.dynamics import *
import numpy as np


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("density-matrix-cpu")
    yield
    cudaq.reset_target()


expected_result_ideal = [
    [
        0.0, -0.12533323353836368, -0.2486898871141352, -0.3681245526115405,
        -0.4817536740096591, -0.5877852521860436, -0.6845471058133562,
        -0.7705132426578025, -0.8443279253882351, -0.9048270523637306,
        -0.9510565162118616, -0.9822872506719114, -0.9980267284053214,
        -0.9980267284460382, -0.9822872507934287, -0.9510565164122886,
        -0.9048270526399483, -0.8443279257359464, -0.770513243071597,
        -0.6845471062867947, -0.587785252711761, -0.48175367457947504,
        -0.36812455321658843, -0.2486898877450009, -0.12533323418523093,
        -6.528038045024165e-10, 0.12533323288985043, 0.2486898864801398,
        0.3681245520020615, 0.4817536734343086, 0.5877852516538943,
        0.6845471053328007, 0.7705132422364205, 0.8443279250326712,
        0.9048270520795919, 0.9510565160036297, 0.9822872505428704,
        0.9980267283575053, 0.9980267284802026, 0.9822872509090343,
        0.9510565166075124, 0.9048270529117113, 0.8443279260799628,
        0.7705132434824419, 0.6845471067579881, 0.5877852532358723,
        0.4817536751482384, 0.3681245538210348, 0.24868988837559658,
        0.12533323483203213, 1.3056095640381904e-09, -0.12533323224133586,
        -0.24868988584614285, -0.36812455139258166, -0.48175367285895676,
        -0.5877852511217446, -0.6845471048522459, -0.7705132418150386,
        -0.8443279246771083, -0.904827051795455, -0.9510565157954002,
        -0.982287250413832, -0.9980267283096933, -0.99802672851437,
        -0.9822872510246431, -0.9510565168027394, -0.9048270531834771,
        -0.8443279264239824, -0.7705132438932886, -0.6845471072291843,
        -0.5877852537599859, -0.48175367571700345, -0.36812455442548186,
        -0.24868988900619315, -0.12533323547883332, -1.9584148261420076e-09,
        0.12533323159282161, 0.24868988521214694, 0.36812455078310297,
        0.4817536722836067, 0.5877852505895972, 0.6845471043716921,
        0.770513241393659, 0.8443279243215464, 0.9048270515113196,
        0.9510565155871714, 0.9822872502847939, 0.9980267282618817,
        0.9980267285485382, 0.9822872511402524, 0.9510565169979661,
        0.9048270534552433, 0.8443279267680006, 0.7705132443041349,
        0.6845471077003792, 0.5877852542840986, 0.4817536762857671,
        0.3681245550299276, 0.24868988963678965, 0.12533323612563457,
        2.6112205665958232e-09
    ],
    [
        1.0, 0.9921147013174793, 0.9685831611410892, 0.9297764859163264,
        0.8763066800932234, 0.8090169944505833, 0.7289686275274656,
        0.6374239898883076, 0.5358267951542017,
        0.4257792917766672, 0.30901699462244303, 0.18738131486730625,
        0.06279051984183398,
        -0.06279051919030254, -0.18738131422591298, -0.3090169940011685,
        -0.4257792911851783, -0.5358267946017032, -0.6374239893833964,
        -0.7289686270779989, -0.8090169940635555, -0.8763066797746584,
        -0.929776485671181, -0.9685831609731798, -0.9921147012294198,
        -0.9999999999931615, -0.9921147013925125, -0.9685831612968109,
        -0.9297764861502813, -0.8763066804017218, -0.8090169948287591,
        -0.7289686279693559, -0.6374239903869435, -0.5358267957017187,
        -0.4257792923644306, -0.3090169952411844, -0.18738131550726722,
        -0.06279052049292211, 0.06279051853835615, 0.18738131358338905,
        0.30901699337819977, 0.4257792905915905, 0.5358267940468564,
        0.6374239888760412, 0.7289686266261377, 0.8090169936743136,
        0.8763066794541746, 0.9297764854245096, 0.9685831608042106,
        0.9921147011408175, 0.9999999999863243, 0.9921147014675475,
        0.9685831614525354, 0.9297764863842384, 0.8763066807102223,
        0.8090169952069381, 0.7289686284112484, 0.6374239908855812,
        0.5358267962492378, 0.42577929295219696, 0.3090169958599277,
        0.18738131614722875, 0.06279052114401051, -0.06279051788640949,
        -0.18738131294086507, -0.3090169927552321, -0.4257792899980043,
        -0.5358267934920123, -0.6374239883686903, -0.7289686261742789,
        -0.8090169932850744, -0.8763066791336938, -0.929776485177841,
        -0.9685831606352447, -0.9921147010522189, -0.9999999999794899,
        -0.9921147015425854, -0.9685831616082613, -0.9297764866181967,
        -0.8763066810187238, -0.8090169955851174, -0.7289686288531416,
        -0.6374239913842179, -0.535826796796756, -0.42577929353996186,
        -0.3090169964786702, -0.1873813167871911, -0.06279052179509709,
        0.0627905172344635, 0.18738131229834232, 0.3090169921322653,
        0.42577928940441656, 0.5358267929371684, 0.6374239878613371,
        0.7289686257224195, 0.8090169928958344, 0.8763066788132121,
        0.9297764849311713, 0.9685831604662767, 0.9921147009636178,
        0.9999999999726529
    ]
]
expected_result_decay = [
    [
        0.0, -0.12408614703067208, -0.24376549742198256, -0.35724482788654943,
        -0.46286384228547217, -0.5591186272015368, -0.6446821858688034,
        -0.7184217856678693, -0.7794129097067345, -0.8269496605141179,
        -0.8605515226281641, -0.8799664499208268, -0.8851703019228946,
        -0.8763627103011329, -0.8539595111239997, -0.8185819298272922,
        -0.7710427531137778, -0.7123297647260717, -0.643586759540161,
        -0.5660924822515595, -0.481237862679671, -0.3905019391140252,
        -0.29542687398973233, -0.197592472436011, -0.0985906139235383,
        -4.788024514628536e-10, 0.09663838802968526, 0.18984475987096416,
        0.27822255134385954, 0.3604787225144521, 0.4354420244348714,
        0.5020789909837533, 0.5595074491091492, 0.6070073843306462,
        0.6440290431438844, 0.6701981997299499, 0.685318560366039,
        0.6893713244349595, 0.6825119652343054, 0.6650643362190816,
        0.6375122482444898, 0.6004887002305419, 0.5547629789290164,
        0.5012258726846284, 0.4408732688673196, 0.3747884247094761,
        0.3041232163891192, 0.23007868121838665, 0.15388517266916585,
        0.07678244772002879, 7.457849997283272e-10, -0.07526205192484081,
        -0.1478512473324847, -0.21667994057248793, -0.28074111113113714,
        -0.3391225894099802, -0.391019511183707, -0.4357448393879168,
        -0.47273782618069504, -0.5015703231016021, -0.5219508827890886,
        -0.5337266315374137, -0.5368829274105111, -0.5315408531336631,
        -0.5179526260306518, -0.4964950383738523, -0.467661070217407,
        -0.43204984268563623, -0.3903551024386612, -0.3433524473398343,
        -0.29188551896987625, -0.2368513993981871, -0.1791854574239967,
        -0.11984589329478465, -0.05979823071647732, -8.712270466691707e-10,
        0.05861414470404304, 0.11514666695372372, 0.1687505071739148,
        0.2186413969994738, 0.264108938032303, 0.30452630138290776,
        0.3393584220469614, 0.3681685891654906, 0.39062336038165363,
        0.4064957562614801, 0.41566671864267524, 0.41812484437343944,
        0.41396443277454853, 0.4033819108948188, 0.38667072485063086,
        0.36421480789384253, 0.33648075602493865, 0.30400885968584795,
        0.267403155099732, 0.22732067099018854, 0.18446005557531128,
        0.13954977480843148, 0.09333607579271852, 0.046570909146573906,
        9.046828442632604e-10
    ],
    [
        1.0, 0.9822429951093886, 0.9494039295565445, 0.9022974379821339,
        0.8419462037240212, 0.7695607700778446, 0.6865167995374319,
        0.5943301888667871, 0.4946304734700204, 0.3891329728220939,
        0.2796101395590745, 0.1678625781791523, 0.0556901952705402,
        -0.05513606803082649, -0.1629014887237789, -0.2659733916473964,
        -0.3628251785323418, -0.45205821449251643, -0.5324212704870548,
        -0.6028272646544449, -0.6623670929534069, -0.7103203894296494,
        -0.7461631078333113, -0.7695718684010229, -0.7804250655569975,
        -0.7788007832663065, -0.7649716140084476, -0.7393965240914617,
        -0.7027099516107173, -0.6557083631499816, -0.5993345307753555,
        -0.5346598215139062, -0.4628648169522486, -0.3852186005390212,
        -0.3030570644245393, -0.21776059610984366, -0.13073150778835646,
        -0.043371568120772974, 0.042940012549026685, 0.12686780660444785,
        0.20714028535104806, 0.2825685328605473, 0.3520632911876734,
        0.41465010217480824, 0.46948234561611946, 0.5158520105715215,
        0.5531980754708654, 0.5811124126835536, 0.5993431737958248,
        0.607795652289244, 0.6065306600162137, 0.5957604922125418,
        0.5758425921977142, 0.54727106085667, 0.5106661869880728,
        0.4667622022148836, 0.41639348801645176, 0.36047948225672216,
        0.30000854811917843, 0.23602107945658324, 0.16959212313577793,
        0.10181380099129034, 0.033777811554019654, -0.03344171507985805,
        -0.09880474683638713, -0.16132101617216016, -0.2200645944306897,
        -0.2741871666706516, -0.32292982411520665, -0.36563321828444545,
        -0.40174594970314975, -0.43083109433250716, -0.45257080205356115,
        -0.4667689331238204, -0.4733517300319542, -0.4723665530956563,
        -0.4639787380102828, -0.44846666191247114, -0.42621513095790314,
        -0.3977072265483482, -0.36351476884642947, -0.3242875748014229,
        -0.2807417033416024, -0.23364689248815096, -0.18381340178829897,
        -0.13207847858385124, -0.07929266821488712, -0.02630618635235299,
        0.02604443364345821, 0.07694921397860915, 0.12563693351447003,
        0.1713864782878839, 0.21353717995780536, 0.2514979997740693,
        0.2847554366237069, 0.3128800601631225, 0.33553159360823087,
        0.35246249503697874, 0.36352001066382605, 0.36864669808071443,
        0.3678794415397008
    ]
]


@pytest.mark.parametrize("init_state", ["array", "enum"])
def test_evolve(init_state):
    # Set random seed for shots-based observe test.
    cudaq.set_random_seed(13)

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    if init_state == "array":
        rho0 = cudaq.State.from_data(
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))
    elif init_state == "enum":
        rho0 = InitialState.ZERO

    # Schedule of time steps.
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (ideal).
    evolution_result = cudaq.evolve(hamiltonian,
                                    dimensions,
                                    schedule,
                                    rho0,
                                    observables=[spin.y(0),
                                                 spin.z(0)],
                                    collapse_operators=[],
                                    store_intermediate_results=cudaq.
                                    IntermediateResultSave.EXPECTATION_VALUE)

    schedule.reset()
    # Now, run the simulation with qubit decaying due to the presence of a collapse operator.
    evolution_result_decay = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE)

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    ideal_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result)
    ]
    decay_results = [
        get_result(0, evolution_result_decay),
        get_result(1, evolution_result_decay)
    ]
    np.testing.assert_allclose(ideal_results, expected_result_ideal, atol=0.01)
    np.testing.assert_allclose(decay_results, expected_result_decay, atol=0.01)

    # Test for `shots_count`
    schedule.reset()
    evolution_result_shots = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
        shots_count=2000)
    results_with_shots = [
        get_result(0, evolution_result_shots),
        get_result(1, evolution_result_shots)
    ]
    np.testing.assert_allclose(results_with_shots,
                               expected_result_ideal,
                               atol=0.1)


def test_evolve_async():
    # Set random seed for shots-based observe test.
    cudaq.set_random_seed(13)

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
    dimensions = {0: 2}

    # Initial state of the system (ground state).
    rho0 = cudaq.State.from_data(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))

    # Schedule of time steps.
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (ideal).
    evolution_result = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE).get()

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]
    ideal_results = [
        get_result(0, evolution_result),
        get_result(1, evolution_result)
    ]
    np.testing.assert_allclose(ideal_results, expected_result_ideal, atol=0.01)

    schedule.reset()
    # Now, run the simulation with qubit decaying due to the presence of a collapse operator.
    evolution_result_decay = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE).get()

    decay_results = [
        get_result(0, evolution_result_decay),
        get_result(1, evolution_result_decay)
    ]
    np.testing.assert_allclose(decay_results, expected_result_decay, atol=0.01)

    # Test for `shots_count`
    schedule.reset()
    evolution_result_shots = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
        shots_count=2000).get()
    results_with_shots = [
        get_result(0, evolution_result_shots),
        get_result(1, evolution_result_shots)
    ]
    np.testing.assert_allclose(results_with_shots,
                               expected_result_ideal,
                               atol=0.1)


def test_evolve_no_intermediate_results():
    """Test evolve with store_intermediate_results=NONE 
    to verify the else branch in evolve_single is working."""

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions
    dimensions = {0: 2}

    # Initial state
    rho0 = cudaq.State.from_data(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))

    # Schedule
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Test 1: NONE without observables
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        store_intermediate_results=cudaq.IntermediateResultSave.NONE)

    # NONE mode: only final state is saved, no intermediate states
    assert len(evolution_result.intermediate_states()) == 1

    # Test 2: NONE with observables
    schedule.reset()
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE)

    # Verify final expectation value is reasonable
    final_exp = evolution_result.expectation_values()
    assert final_exp is not None

    # Test 3: NONE with collapse_operators (tests the missing return bug)
    schedule.reset()
    evolution_result_decay = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE)

    # Results with decay should differ from ideal (noise should have effect)
    # This test would fail if the noise_model is ignored (the return bug)
    final_exp_decay = evolution_result_decay.expectation_values()
    assert final_exp_decay is not None
    # expectation_values() returns [[ObserveResult, ...]] - outer list is time steps,
    # inner list is observables. With NONE mode, there's only one time step (final).
    assert final_exp_decay[0][0].expectation() != final_exp[0][0].expectation()
    assert final_exp_decay[0][1].expectation() != final_exp[0][1].expectation()


def test_evolve_async_no_intermediate_results():
    """Test evolve_async with store_intermediate_results=NONE 
    to verify the else branch in evolve_single_async is working."""

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions
    dimensions = {0: 2}

    # Initial state
    rho0 = cudaq.State.from_data(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))

    # Schedule
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Test 1: NONE without observables
    evolution_result = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # NONE mode: only final state is saved, no intermediate states
    assert len(evolution_result.intermediate_states()) == 1

    # Test 2: NONE with observables
    schedule.reset()
    evolution_result = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # Verify final expectation value is reasonable
    final_exp = evolution_result.expectation_values()
    assert final_exp is not None

    # Test 3: NONE with collapse_operators (tests the missing return bug)
    schedule.reset()
    evolution_result_decay = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # Results with decay should differ from ideal (noise should have effect)
    # This test would fail if the noise_model is ignored (the return bug)
    final_exp_decay = evolution_result_decay.expectation_values()
    assert final_exp_decay is not None
    # expectation_values() returns [[ObserveResult, ...]] - outer list is time steps,
    # inner list is observables. With NONE mode, there's only one time step (final).
    assert final_exp_decay[0][0].expectation() != final_exp[0][0].expectation()
    assert final_exp_decay[0][1].expectation() != final_exp[0][1].expectation()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
