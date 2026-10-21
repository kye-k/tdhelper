![alt text](https://github.com/kye-k/tdhelper/blob/main/tdhelper_logo.png?raw=true)

`tdhelper` (*trajectory data helper*) is a pythonic toolkit for working with single particle tracking (SPT) data. SPT data is usually defined by trajectory ID, x coordinate, y coordinate, and time. `tdhelper` can flexibly import various SPT data formats (TrackMate .CSV; PALMTracer .txt; and various other formats like .TRXYT), perform simple filtering (length, region of interest, time), perform simple calculations ($MSD$, $D_{app}$, step-angle, etc.) and create figures (super-resolved density maps, trajectory maps, KDE maps, etc.).

`tdhelper` can make your work reproducible by allowing for pipelining trajectory data between other python SPT analysis packages (e.g., NASTIC, saSPT, TRamWay).
`tdhelper` plots rely on matplotlib and seaborn; so you can be creative with how you want your figures to look!

As of now, `tdhelper` has very simple functions, contributions to enhance use and include more functions are more than welcome! The goal `tdhelper` is to have an open-source simple API and to make SPT analyses (from trajectories to figures) accessible and reproducible without the need of a GUI.

## Install 

```
git clone https://github.com/kye-k/tdhelper
cd tdhelper
pip install -e .
```
