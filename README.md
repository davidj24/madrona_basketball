# Madrona Simple Example

## Installation
This repository contains an implementation of a simple 2D basketball simulator environment written in the [Madrona Engine](https://madrona-engine.github.io).

This environment was built off of the Madrona Simple Example repository. https://github.com/shacklettbp/madrona_simple_example.git


## Build
First, make sure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies) (briefly, recent python and cmake, as well as Xcode or Visual Studio on MacOS or Windows respectively).

Next, fetch the repo (don't forget `--recursive`!):
```bash
git clone --recursive https://github.com/davidj24/madrona_basketball.git
cd madrona_basketball
```

Next, for Linux and MacOS: Run `cmake` and then `make` to build the simulator:
```bash
mkdir build
cd build
cmake ..
make -j # cores to build with
cd ..
```

Or on Windows, open the cloned repository in Visual Studio and build
the project using the integrated `cmake` functionality.

Now, setup the python components of the repository with `pip`:
```bash
pip install -e . # Add -Cpackages.madrona_simple_example.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```
(And finally, [install pytorch](https://pytorch.org/get-started/locally/)):


## Running Training
You can test the simulator as follows:
```bash
python3 scripts/ppo.py --viewer --num-envs 8192 # Simulate on GPU with a playback once every 100 iterations

python3 scripts/ppo.py --full-viewer --num-envs 1 --no-use-gpu # Simulate on CPu with one environment viewed in real time for debugging (extremely slow training)
```

<br>
Logs of training will be stored in logs/{model_name}. To play one back, run:

```bash
python3 scripts/viewer.py --playback-log "path to log you want to play"
```

<br>
<br>
<br>
<br>

File Description
================
All simulation related files are in src/
<br>
All training and visualization files in scripts/
<br>
More details available in sub-READMEs.
