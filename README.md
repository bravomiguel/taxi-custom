# taxi-custom

RL Taxi Problem by Sahil Dhingra and Miguel Bravo, modified from the Open AI Gym [Taxi-v3](http://gym.openai.com/envs/Taxi-v3/) environment.

## Description

There are three designated locations, P(ickup), D(ropoff), H (pothole). Grid world below:

```bash
+-------+
|P: | : |
| :H: : |
| : : : |
| | : | |   
| | : |D|
+-------+
```

When the episode starts, the taxi starts off at a random square and the passenger pickup location P is always in the top-left corner. The taxi drives to P, picks up the passenger, drives to the passenger's destination D (one of the other three corners), and then drops off the passenger. In addition, there are barriers, and a pothole H - appearing in one of locations (2,1), (2,2), (2,3), (2,4) - which the taxi needs to avoid. Once the passenger is dropped off, the episode ends. 

## Passenger location
- 0: P (0,0)
- 1: in taxi

## Destination D
- 0: (0,3)
- 1: (4,0)
- 2: (4,3)

## Pothole locations
- 0: (1,0)
- 1: (1,3)
- 2: (2,1)
- 3: (2,3)

## Actions
There are 6 discrete deterministic actions:
- 0: move south
- 1: move north
- 2: move east 
- 3: move west 
- 4: pickup passenger
- 5: dropoff passenger

## Rewards
There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally. There is also a reward of -8 for driving into the pothole.

## Rendering
- blue: passenger
- magenta: destination
- yellow: empty taxi
- green: full taxi
- letters (P, D, H): pick-up, destination and pothole locations

## Installation

```bash
cd taxi-custom
pip install -e .
```
