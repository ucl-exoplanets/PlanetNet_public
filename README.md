# PlanetNet 1.0


The PlanetNet code is designed to perform spectral clustering to train a two-stream convolutional neural 
network to identify spectral/spatial features. 

The code is part of the Nature Astronomy publication "Mapping Saturn using deep learning" (DOI: 10.1038/s41550-019-0753-8).

The data for the paper can be found here: https://osf.io/htgrn/  or DOI 10.17605/OSF.IO/HTGRN

The code is still under development and may not be the easiest to use in its current form. Future versions will improve on that. 

To convert the raw Cassini-VIMS ascii data found in /data/storm, please run 

```python
preprocess_vims.py
```

and change the data paths therein 

To train PlanetNet on the prepared data cubes, run 

```python
python PlanetNet_cnn_train.py 
```

and make sure you point the code to the right data paths within the file. 
In future versions the user interface will be revamped to make it easier to interact with the code. 


### License:

This work is licensed under the Creative Commons Attribution 4.0 International License.

We would like to draw your attention to Section 3 of the license and to:

 - retain identification of the creators by including the above listed references in future work and publications.
 - indicate if You modified the Licensed Material and retain an indication of any previous modifications

To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

