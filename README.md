![alt text](https://github.com/kye-k/tdhelper/blob/main/tdhelper_logo.png?raw=true)

`tdhelper` is a tool used for working with single particle tracking (SPT) data. SPT data is usually defined by trajectory ID, x coordinate, y coordinate, and time. `tdhelper` can flexibly import various SPT data formats (TrackMate .CSV; PALMTracer .txt; and various other formats like .TRXYT), perform simple filtering (length, region of interest, time), perform simple calculations ($MSD$, $D_{app}$, step-angle, etc.) and create figures (super-resolved density maps, trajectory maps, KDE maps, etc.).

`tdhelper` can make your work reproducible by allowing for pipelining trajectory data between different other SPT-analysis packages.
`tdhelper` plots rely on matplotlib and seaborn; so you can be creative with how you want your figures to look!

## Install 

```
git clone https://github.com/kye-k/tdhelper
cd tdhelper
pip install -e .
```

