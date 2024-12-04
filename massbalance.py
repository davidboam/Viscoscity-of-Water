# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:26:36 2018
update v2 Wed 23 Aug 2023

Data acquisition script for Skills sessions 4 & 5: viscosity of water using automated data acquisition

NOTE: 
    You do not need to know/understand how any of this works, beyond ensuring the correct value of SERIAL_PORT below!
    In later parts of the course you will have opportunity to control how data collection works by modifying the 
    provided example scripts if you wish - you can do this here, but there is no need to do so for this activity.


Controls:
    A&D FZ-i/FX-i series precision digital balance over serial interface
    
Requirements:
    level2labs module (>=2.2.0): In users site-packages directory on computer
    pySerial
    drivers for usb-serial interface converters
        
@author: Aidan Hindmarch
"""

#################################################################################
# Change port name to correspond to correct serial port, found from either 
# NI MAX or Windows device manager              
#                                                               
# e.g.                                                          
#	SERIAL_PORT = "COM2"                                         
#                                                              
SERIAL_PORT = "COM9"
#                                                             
#################################################################################


def collect_data(duration_s=1.0, output_filename=None, plot=False, dwell_s=1.0):
    '''Do the measurement, save the data to file, then plot a graph if required.
       Sensible default values specified for keyword arguments. 
    '''
    from level2labs.skills import Balance
    from time import sleep
    import numpy
    
    dwell_s = min(dwell_s, 0.1)                 # DWELL must be 0.1 seconds or longer 
    n_samples = int(duration_s / dwell_s)       # Approx number of measurements to take
    mass, time = numpy.zeros(n_samples), numpy.zeros(n_samples) # Initialise arrays to store/plot data

    with Balance(SERIAL_PORT) as balance:       # Connect to mass balance via USB-serial adaptor 
        print(balance.get_identity())           # print some info about the device
        print('Collecting data...')             # Confirm all is progressing as planned...
        
        for pt in range(n_samples):             # Loop over number of measurements to collect
            rdg = balance.get_reading()         # Take a timestamped reading from the balance: format of rdg is: [[timestamp, time units], [mass, mass units]]
            print(pt+1, rdg)                    # print data to screen
            
            time[pt], mass[pt] = rdg[0][0], rdg[1][0]       # Add in the latest reading data (not units) to the data arrays at appropriate place
            
            if output_filename: 
                with open(output_filename, 'a') as output:  # Open output file in 'append' mode: each measurement added to end of file
                    print(time[pt], mass[pt], file=output)  # Write the data to the file as columns, that you can easily import into other software.
                                                            # Using 'print' sorts out some formatting, and writing EVERY datapoint ensures you 
                                                            # do NOT lose everything if a long measurement fails for some reason.
            sleep(dwell_s)                      # Wait specified time before next reading
        
        
        print('NOTE the units!')                # but nobody ever does...
        if plot:
            import matplotlib.pyplot as plt
            print('Close the plot window to end the program!')
            plt.plot(time, mass, marker='x')    # Timestamp on horizontal axis, mass reading on vertical axis.
            plt.show()


#################################################################################


if __name__ == '__main__':  # Program 'execution' starts from here!
    import argparse
    
    # Process command line arguments: If you want to learn how this works, see https://docs.python.org/3/howto/argparse.html#argparse-tutorial
    parser = argparse.ArgumentParser() 
    parser.add_argument('duration_s', type=float, help="measurement duration in seconds") # required positional argument
    parser.add_argument('output_filename', type=str, help="filename to save data") # required positional argument
    parser.add_argument('-p', '--plot', action='store_true', help="plot a simple graph of the data") # optional, default False
    parser.add_argument('-d', '--dwell-time', type=float, default=1.0, help="dwell time between readings (> 0.1 s)" ) # optional, default 1.0
    args = parser.parse_args() # determine the correct values to use...
    
    collect_data(duration_s=args.duration_s, output_filename=args.output_filename, plot=args.plot, dwell_s=args.dwell_time) # do the measurement
    
    raise SystemExit(0) from None # indicate successful completion
    
    
    