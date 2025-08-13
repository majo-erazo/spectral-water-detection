## File naming convention
**[sensor location]_ [sensor name]_[measured variable].csv**
With:

 - [sensor location]:
	 - catchment: in the catchment area
	 - hall: the experimental hall in which the flume is installed
	 - flume: the flume in which wastewater was circulated
 - [sensor name]:
	 - pluvio2s: for precipitation measurement
	 - isa: spectrometer
	 - isemax: ion selective electrode, phmeter, and thermometer
	 - scan: spectrometer and thermometer
	 - turbimax: turbidity sensor
	 - cs2: flowmeter
	 - e53: conductivity probe
	 - i3: levelmeter
 - [measured variable]: precipitation, absorbance, nh4: ammonium, ph, temperature, turbidity, flow, ec: electrical conductivity and level.

## Table organization
** Common table structure**
Each table is organized in the same structure:
|index|timestamp_iso|...|variable_i|...|validity_test_j|...|valid_data|
|--|--|--|--|--|--|--|--|
|*integer*|*string*|...|*float*|...|*0/1*|...|*0/1*|
|Index of the measurement|Measurement timestamp in isoformat|...|Value of variable_i|...|1 if validity test j is passed|...|1 if the data point is considered valid |
**Variable naming convention**
Each variable_i is named with the following convention:
**[sensor name] _ [measured variable] _ [unit]**
With:

 - [sensor name]: see section above
 - [measured variable]: see section above. For absorbance measurements, the name of the variable contains the wavelength of measurement. Ex: absorbance400 -> absorbance at 400 nm. Ex2: absorbance5025 -> Absorbance at 502.5 nm.
 - [unit]:
	 - mm: milimeter
	 - *none*: no unit
	 - v: volt
	 - degc: degree celcius
	 - fnu: Formazine nephelometric unit
	 - l_s: liter per second
	 - ms_cm: millisiemens pro centimeter
	 - _m: per meter

**Data validation**
Depending on the data characteristics, between 1 and 3 validation steps are applied:

 - Test 1: Verification that the flume was working at the given timestamp
 - Test 2: Verification that the sensor was working properly at the given timestamp
 - Test 3: Verification that the variable value is within a predefined range.

The last column of the table, name "valid_data", combines the three validation tests to determine when a datapoint is considered valid.