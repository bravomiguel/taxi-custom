# taxi-custom

RL Taxi Problem by Sahil Dhingra and Miguel Bravo, modified from the Open AI Gym [Taxi-v3](http://gym.openai.com/envs/Taxi-v3/) environment.

## Description

There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). Grid world below:

```bash
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| : : : : |   
| | : | : |
|Y| : |B: |
+---------+
```

When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends. 

So far this describes the original Taxi-v3 environment. Our modifications to this are:
* adding an additional row in the middle of the grid world (3rd row)
* randomly placing a 'pothole' anywhere in the middle two rows of the grid world (3rd and 4th rows) at the start of each episode, which the taxi should try to avoid.

## Observations
There are 6000 discrete states since there are 30 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), 4 possible destination locations, and 10 possible pothole locations. 
    
## Passenger locations
* 0: R(ed)    (0,0)
* 1: G(reen)  (0,4)
* 2: Y(ellow) (4,0)
* 3: B(lue)   (4,3)
* 4: in taxi
    
## Destinations
* 0: R(ed)    (0,0)
* 1: G(reen)  (0,4)
* 2: Y(ellow) (4,0)
* 3: B(lue)   (4,3)

## Pothole locations
* 0: (2,0)
* 1: (2,1)
* 2: (2,2)
* 3: (2,3)
* 4: (2,4)
* 5: (3,0)
* 6: (3,1)
* 7: (3,2)
* 8: (3,3)
* 9: (3,4)
        
## Actions
There are 6 discrete deterministic actions:
* 0: move south 
* 1: move north 
* 2: move east 
* 3: move west 
* 4: pickup passenger 
* 5: dropoff passenger 
    
## Rewards 
There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally. There is also a reward of -10 for driving into the pothole.
    
## Rendering:
* blue: passenger
* magenta: destination
* yellow: empty taxi
* green: full taxi
* letters R, G, Y and B: locations for passengers and destinations
* letter H: location of pothole
    
## State Space

State space is represented by: (taxi_row, taxi_col, passenger_location, destination, pothole_location)

## Installation

```bash
cd taxi-custom
pip install -e .
```
