# gravity_simulation
## About
This is a gravity simulation using the runge kutter integration method.
## Simulating
To start a new simulation, change `path` in `cnf.py` to where you want your simulations to be stored, `{}` is a placeholder for the simulation id. (WARNING: SIMULATION FILES CAN GET VERY BIG)
Then you can run the simulation using `simulate.py`.
## Rendering
To render the simulation, run `render.py`and enter the path of the simulation.
## Configuration
To create an own simulation environment, change `environment.py` to what you want it to be. X (shape=[?, 2]) is the array for the positions, V (shape=[?, 2]) is the array for the velocities, M (shape=[?, 2]) is the array for the masses of the particles, COLOR (shape=[?, 3]) is the array for the colors.

`cnf.py`has all the settings for simulating the environment.
`cnfrender.py` stores the settings for rendering a simulation.