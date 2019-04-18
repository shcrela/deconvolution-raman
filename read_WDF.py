# -*- coding: latin-1 -*-
from __future__ import print_function
import numpy as np
import os
import time
import pandas as pd

def convert_time(t):
    '''Takes the Windows 64bit timestamp and converts it to human readable format
    
    Input:
        t: timestamp in W64 format (default for .wdf files)
    Output:
        string formatted to suit local settings
    
    Example:
        >>>time_of_spectrum_recording = [convert_time(x) for x in origins.iloc[:,4]]
        
        should give you the list with the times on which each specific spectrum was recorded    
        '''
    return time.strftime('%c', time.gmtime((t/1e7-11644473600)))


def read_WDF(filename, vebrose=False):
    '''
    Reads data from the binary .wdf file and returns it in form of five variables
    
    Example: 
        >>>measure_params, map_params, x_values, spectra, origins = read_WDF(filename)
    
    Input:
        filename: The path to the file, with the filename itself
    Output:
        measure_params: a dictionary containing measurement parameters
        map_params:  a dictionary containing map parameters
        x_values: a numpy array containing the raman shifts
        spectra: numpy array contaiing all the recorded spectra
        origins: pandas dataframe containing the conditions of each recording.
                Note that it has triple column names (label, data type, data units)
    '''
        
    
    DATA_TYPES = ['Arbitrary','Spectral','Intensity', 
                  'SpatialX','SpatialY','SpatialZ', 
                  'SpatialR','SpatialTheta','SpatialPhi', 
                  'Temperature','Pressure','Time', 
                  'Derived','Polarization','FocusTrack', 
                  'RampRate','Checksum','Flags', 
                  'ElapsedTime','Frequency', 
                  'MpWellSpatialX','MpWellSpatialY', 
                  'MpLocationIndex','MpWellReference', 
                  'PAFZActual','PAFZError','PAFSignalUsed', 
                  'ExposureTime','EndMarker']
    
    DATA_UNITS = ['Arbitrary','RamanShift','Wavenumber','Nanometre', 
                  'ElectronVolt','Micron','Counts','Electrons', 
                  'Millimetres','Metres','Kelvin','Pascal', 
                  'Seconds','Milliseconds','Hours','Days', 
                  'Pixels','Intensity','RelativeIntensity', 
                  'Degrees','Radians', 
                  'Celcius','Farenheit','KelvinPerMinute', 
                  'FileTime','Microseconds','EndMarker']
    
    SCAN_TYPES = ['Unspecified','Static','Continuous','StepRepeat', 
                  'FilterScan','FilterImage','StreamLine','StreamLineHR', 
                  'Point','MultitrackDiscrete','LineFocusMapping']
    
    MAP_TYPES = {0:'RandomPoints', 1:'ColumnMajor', 2:'Alternating', 
                 3:'LineFocusMapping', 4:'InvertedRows', 5:'InvertedColumns', 
                 6:'SurfaceProfile', 7:'XyLine', 128:'Slice'}
    
    MEASUREMENT_TYPES = ['Unspecified', 'Single', 'Series', 'Map']
    
    WDF_FLAGS = {0:'WdfXYXY',1:'WdfChecksum',2:'WdfCosmicRayRemoval', 
                 3:'WdfMultitrack',4:'WdfSaturation',5:'WdfFileBackup', 
                 6:'WdfTemporary',7:'WdfSlice',8:'WdfPQ', 
                 16:'UnknownFlag (check in WiRE?)'}

    try:
        f = open(filename, "rb")
        print(f'Reading the file: "{filename}"\n')
    except IOError:
        raise IOError(f"File {filename} does not exist!")
        
    filesize = os.path.getsize(filename)
        
    def _read(f=f, dtype=np.uint32, count=1):
        '''Reads bytes from binary file, with the most common values given as default.
        Returns the value itself if one value, or list if count > 1
        Note that you should do ".decode()" on strings to avoid getting strings like "b'string'"
		For further information, refer to numpy.fromfile() function
		'''
        if count==1:
            return np.fromfile(f, dtype=dtype, count=count)[0]
        else:
            return np.fromfile(f, dtype=dtype, count=count)[0:count]

    def print_block_header(name, i, vebrose=vebrose):
        if vebrose:
            print(f"\n{' Block : '+ name + ' ':=^80s}\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    
    
    block_names=[]
    block_sizes=[]
    offset = 0
    b_off = []
    
    # Reading all of the block names, offsets and sizes
    while offset<filesize-1:
        header_dt = np.dtype([('block_name','|S4'), ('block_id',np.int32), ('block_size',np.int64)])
        f.seek(offset)
        b_off.append(offset)
        block_header = np.fromfile(f, dtype=header_dt, count=1)
        offset += block_header['block_size'][0]
        block_names.append(block_header['block_name'][0].decode())
        block_sizes.append(block_header['block_size'][0])
    
    
    name = 'WDF1'
    params={}
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        print_block_header(name, i)
        f.seek(b_off[i]+16)
#        TEST_WDF_FLAG = _read(f,np.uint64)
        params['WdfFlag'] = WDF_FLAGS[_read(f,np.uint64)]
        f.seek(60)
        params['PointsPerSpectrum'] = npoints = _read(f)
        params['Capacity'] = nspectra = _read(f, np.uint64) # Number of spectra measured (nspectra)
        params['Count'] = ncollected =_read(f, np.uint64) # number of spectra written into the file (ncollected)
        params['AccumulationCount'] = _read(f) # number of accumulations per spectrum
        params['YlistLength'] = _read(f) # number of elements in the y-list (>1 for image)
        params['XlistLength'] = _read(f) # number of elements in the x-list
        params['DataOriginCount'] = _read(f) # number of data origin lists
        params['ApplicationName'] = _read(f, '|S24').decode()
        version = _read(f, np.uint16, count=4)
        params['ApplicationVersion'] = '.'.join([str(x) for x in version[0:-1]])+' build '+str(version[-1])
        params['ScanType'] = SCAN_TYPES[_read(f)]
        params['MeasurementType'] = MEASUREMENT_TYPES[_read(f)]
        params['StartTime'] = convert_time(_read(f,np.uint64))
        params['EndTime'] = convert_time(_read(f,np.uint64))
        params['SpectralUnits'] = DATA_UNITS[_read(f)]
        params['LaserWaveLength'] = np.round(10e6/_read(f, '<f'),2)
        f.seek(240)
        params['Title'] = _read(f,'|S160').decode()
    if vebrose:
        for key, val in params.items():
            print(f'{key:-<40s} : \t{val}')
        if nspectra != ncollected:
            print(f'\nATTENTION:\nNot all spectra were recorded\n'
                  'Expected nspectra={nspectra}, while ncollected={ncollected}' 
                  '\nThe {nspectra-ncollected} missing values will be shown as blanks\n')
    
        
    name = 'WMAP'
    map_params = {}
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        print_block_header(name, i)
        f.seek(b_off[i]+16)
    #    m_flag = _read(f)
        map_params['MapAreaType'] = MAP_TYPES[_read(f)]
        _read(f)
        map_params['InitialCoordinates'] = np.round(_read(f, '<f', count=3),2)
        map_params['StepSizes'] = np.round(_read(f, '<f', count=3),2)
        map_params['NbSteps'] = n_x,n_y,n_z = _read(f, np.uint32, count=3)
        map_params['LineFocusSize'] = _read(f)
    if vebrose:
        for key, val in map_params.items():
            print(f'{key:-<40s} : \t{val}')
     
       
    name = 'DATA'
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        data_points_count=npoints*ncollected
        print_block_header(name, i)
        f.seek(b_off[i]+16)
        spectra = _read(f,'<f', count=data_points_count).reshape(ncollected, npoints)
        if vebrose:
            print(f'{"The number of spectra":-<40s} : \t{spectra.shape[0]}')
            print(f'{"The number of points in each spectra":-<40s} : \t{spectra.shape[1]}')
        if params['MeasurementType'] == 'Map':
            if map_params['MapAreaType'] == 'InvertedRows':
                spectra = [spectra[((xx//n_x)+1)*n_x-(xx%n_x)-1] if (xx//n_x)%2==1 else spectra[xx] for xx in range(nspectra)]
                spectra = np.asarray(spectra)
                if vebrose:
                    print('*It seems your file was recorded using the "Inverted Rows" scan type' 
                          '(sometimes also reffered to as "Snake").\n ' 
                      'Note that the spectra will be rearanged so it could be \n' 
                      'read the same way as other scan types (from left to right, and from top to bottom)')
        
    
    name = 'XLST'
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        print_block_header(name, i)
        f.seek(b_off[i]+16)
        params['XlistDataType'] = DATA_TYPES[_read(f)]
        params['XlistDataUnits'] = DATA_UNITS[_read(f)]
        x_values = _read(f,'<f', count=npoints)
    if vebrose:
        print(f"{'The shape of the x_values is':-<40s} : \t{x_values.shape}")
        print(f"*These are the \"{params['XlistDataType']}\" recordings in \"{params['XlistDataUnits']}\" units")

        
    name = 'YLST' # This is where the image is stored (if recorded). When y_values_count > 1, there should be an image.
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        print_block_header(name, i)
        f.seek(b_off[i]+16)
        params['YlistDataType'] = DATA_TYPES[_read(f)]
        params['YlistDataUnits'] = DATA_UNITS[_read(f)]
        y_values_count = int((block_sizes[i]-24)/4) # if > 1, we can say that this is the number of pixels in the recorded microscope image
        if y_values_count > 1:
            y_values = _read(f,'<f', count=y_values_count)
            if vebrose:
                print("There seem to be the image recorded as well")
                print(f"{'Its size is':-<40s} : \t{y_values.shape}")
        else:
            if vebrose:
                print("*No image was recorded")
    
        
    name='ORGN'
    origin_labels = []
    origin_set_dtypes = []
    origin_set_units = []
    origin_values = np.empty((params['DataOriginCount'],nspectra), dtype='<d')
    gen = [i for i,x in enumerate(block_names) if x==name]
    for i in gen:
        print_block_header(name, i)
        f.seek(b_off[i]+16)
        nb_origin_sets = _read(f) # This is the same as params['DataOriginCount']
        for set_n in range(nb_origin_sets):
            data_type_flag = _read(f).astype(np.uint16) # not sure why I had to add the astype part, but if I just read it as uint32, I got rubbish sometimes
            origin_set_dtypes.append(DATA_TYPES[data_type_flag])
            origin_set_units.append(DATA_UNITS[_read(f)])
            origin_labels.append(_read(f, '|S16').decode())
            if data_type_flag == 11:
                origin_values[set_n]=_read(f,np.uint64, count=nspectra) # special case for reading timestamps
            else:
                origin_values[set_n] = np.round(_read(f, '<d', count=nspectra),2)
                
            if params['MeasurementType'] == 'Map':
                if map_params['MapAreaType'] == 'InvertedRows': # To put the "Inverted Rows" into the "from left to right" order
                    origin_values[set_n] = [origin_values[set_n][((xx//n_x)+1)*n_x-(xx%n_x)-1] if (xx//n_x)%2==1 else origin_values[set_n][xx] for xx in range(nspectra)]
                    origin_values[set_n] = np.asarray(origin_values[set_n])                
    if vebrose:
        print('\n\n\n')

    return params, map_params, x_values, spectra, pd.DataFrame(origin_values.T, columns=[origin_labels, origin_set_dtypes, origin_set_units])
      
