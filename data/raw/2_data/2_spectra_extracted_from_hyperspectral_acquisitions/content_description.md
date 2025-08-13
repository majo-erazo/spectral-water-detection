
## flume_mvx_reflectance.csv
**Brief description:**
In addition to the raw hyperspectral datacubes (in: 2_data/1_raw_mvx_hyperspectral_datacubes), we included in the dataset a spectral signature extracted from each data-cube. 
**Extraction of the reflectance spectra**
The extraction procedure followed four steps: 

 1. Conversion of the raw data cubes to reflectance by normalizing them with a dark and white reference.
 2. Cropping the datacubes to retain only the area and the wavelengths of interest.
 3. Selection of pixels based on a filter that eliminates the brightest 20% and darkest 20% of pixels, defining the pixel brightness as its average reflectance intensity.
 4. Calculation of the spectral signature using the average reflectance spectra of the remaining 60 percent of the pixels

These four steps made it possible to extract a representative, unidimensional spectral signature of the wastewater. Our previous manuscript provides more details on this spectral signature extraction procedure:

Lechevallier P, Villez K, Felsheim C, Rieckermann J. Towards non-contact pollution monitoring in sewers with hyperspectral imaging. Environ Sci: Water Res Technol . 2024 Feb 19'. https://pubs.rsc.org/en/content/articlelanding/2024/ew/d3ew00541k

We also provide a datacube processing example in "1_Codes" with the necessary Python functions.
**Data organization**
For each data-cube, identified by the timestamps for which it was captured, the extracted spectral signature is in the form of a reflectance spectra. This spectra ranges from 400 to 798 nm every 2 nm, and the reflectance is uniteless. 

In addition to the reflection intensities at each wavelength, this table contains the results of three validation tests applied to the data: 

 - Test 1: Verification that the flume was working at the given timestamp
 - Test 2: Verification that the MV.X hyperspectral camera was working properly at the given timestamp
 - Test 3: Verification that the reflectance intensity at the 600nm wavelength was within the expected range (i.e., between 0.05 and 0.25)

The last column of the table, name "valid_data", combines the three validation tests to determine when a datapoint is considered valid.

| index | timestamp_iso | ... | mvx_reflectanceXXX_| ... | data_validations_test iii | ... | valid_data |
|--|--|--|--|--|--|--|--|
| *integer* | *string* |  | *float* | | *0/1* | | *0/1* |
| Index of the measurement | Timestamp, in isoisoformat | |Reflectance calculated at the XXX wavelength | | Validation test iii | | 1 if the data is considered valid|

With:

 - XXX ranging from 400 to 798 nm, with a 2nm step
 - iii ranging from 1 to 3
