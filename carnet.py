from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent", "Starts"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]   
    ],
    evidence=["Gas", "Ignition", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Gas": ['Full', 'Empty'],
        "Ignition": ["Works", "Doesn't work"],
        "KeyPresent": ['yes', 'no']
    }
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_key = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent":['yes','no']},
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key)

car_infer = VariableElimination(car_model)

# print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

# Given that the car will not move, what is the probability that the battery is not working?
c1 = car_infer.query(variables=["Battery"],evidence={"Moves":"no"})

# Given that the radio is not working, what is the probability that the car will not start?
c2 = car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"})

# Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
c3_1 = car_infer.query(variables=["Radio"],evidence={"Battery":"Works"})
c3 = car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"})

# Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
c4_1 = car_infer.query(variables=["Ignition"],evidence={"Moves":"no"})
c4 = car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"})

# What is the probability that the car starts if the radio works and it has gas in it? Include each of your queries in carnet.py. Also, please add a main that executes your queries.
c5 = car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"})

# print("Probability of the battery given that the car will not move: \n", c1)
# print("Probability of the Car Starting given that the radio is not working: \n", c2)
# print("Probability of the radio given that the battery is working: \n", c3_1)
# print("Probability of the radio given that the battery is working and the car has gas in it: \n", c3)
# print("Probability of the ignition given that the car doesn't move: \n", c4_1)
# print("Probability of the ignition given that the car doesn't move and it doesn't have gas in it: \n", c4)
# print("Probability that the Car Starting if the radio works and it has gas in it: \n", c5)

c6 = car_infer.query(variables=["KeyPresent"],evidence={"Moves":"no"})

print("Probability of the Key given that the car does not move", c6)