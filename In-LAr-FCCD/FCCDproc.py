import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

# Modify this value for different energy resolution
pctResAt1MeV = 0.20

# Open the input file with h5py (g4 doesn't write pandas-ready hdf5)
g4sfile = h5py.File('g4simpleout.hdf5', 'r')
g4sntuple = g4sfile['default_ntuples']['g4sntuple']

# List of column names
columns = ['event', 'step', 'Edep', 'volID', 'iRep', 'x', 'y', 'z', 'lx', 'ly', 'lz', 'pdx', 'pdy', 'pdz', 't']

# Build the pandas DataFrame from the hdf5 datasets
g4sdf = pd.DataFrame({col: np.array(g4sntuple[col]['pages']) for col in columns})


class G4_Event:
    """
    Class for processing Geant4 simulated events and calculating event properties.
    """

    def __init__(self, dataframe, FCCD=0.1):
        """
        Initialize G4_Event with a DataFrame containing the simulation data and the desired FCCD threshold.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing the Geant4 simulation data.
            FCCD (float): Full Charge Collection Depth threshold for step deadness calculation.
        """
        self.dataframe = dataframe
        self.FCCD = FCCD
        # Add a new column 'FCCD' to the DataFrame and fill it with the specified FCCD value
        self.dataframe['FCCD'] = FCCD

    def polar(self):
        """
        Convert particle positions to cylindrical coordinates (r, phi, z).

        Returns:
            r (np.ndarray): Array of radial distances.
            phi (np.ndarray): Array of azimuthal angles.
            z (np.ndarray): Array of z-coordinates.
        """
        r = np.sqrt(self.dataframe['x'] ** 2 + self.dataframe['y'] ** 2)
        phi = np.arctan2(self.dataframe['y'], self.dataframe['x'])
        z = self.dataframe['z']
        return r, phi, z

    def where_are_you(self):
        """
        Determine the location of the event (geBase, geHole, inHole, inlAr).

        Returns:
            location (np.ndarray): Array containing the location labels.
        """
        r, phi, z = self.polar()

        geBase_condition = (r < 40) & (z < 35)
        geHole_condition = (35 < z) & (z < 70) & (10 < r) & (r < 40)
        inHole_condition = (r < 10) & (z > 35)

        location = np.where(geBase_condition, 'geBase', np.where(geHole_condition, 'geHole', np.where(inHole_condition, 'inHole', 'inlAr')))
        return location

    def distance_to_surface(self):
        """
        Calculate the distance to the surface based on the location of the event.

        Returns:
            dist2surf (np.ndarray): Array of distances to the surface.
        """
        hit_location = self.where_are_you()
        r, phi, z = self.polar()

        # Initialize an array to store the distances
        dist2surf = np.empty_like(hit_location, dtype=np.float64)

        # Calculate distances based on hit locations
        geBase_condition = (hit_location == 'geBase')
        geHole_condition = (hit_location == 'geHole')
        inHole_condition = (hit_location == 'inHole')
        inlAr_condition = (hit_location == 'inlAr')

        dist2surf[geBase_condition] = 40.0 - r[geBase_condition]
        dist2surf[geHole_condition] = 40.0 - r[geHole_condition]
        dist2surf[inHole_condition] = np.minimum(40.0 - r[inHole_condition], np.abs(z[inHole_condition] - 70.0))
        dist2surf[inlAr_condition] = np.nan
        return dist2surf

    def stepdeadness(self):
        """
        Determine the activeness and dead_hit based on the distance to the surface (FCCD threshold).

        Returns:
            activeness (np.ndarray): Array of activeness values (0 or 1).
            dead_hit (np.ndarray): Array of boolean values indicating dead hits.
        """
        dist2surf = self.distance_to_surface()
        activeness = np.where(dist2surf < self.FCCD, 0, 1)
        dead_hit = dist2surf < self.FCCD
        return activeness, dead_hit

    def energy_cut(self, activeness):
        """
        Apply the energy cut on the Edep data based on the step deadness information.

        Parameters:
            activeness (np.ndarray): Array of activeness values (0 or 1) indicating step deadness.

        Returns:
            filtered_data (pd.DataFrame): DataFrame containing the data after applying the energy cut.
        """
        energy_cut_data = self.dataframe.loc[(self.dataframe['Edep'] > 0) & (self.dataframe['volID'] == 1) & (activeness == 1)]
        return energy_cut_data
    
    def correct_energy(self, activeness):
        """
        Correct the energy of hits based on the step deadness information.

        Parameters:
            activeness (np.ndarray): Array of activeness values (0 or 1) indicating step deadness.

        Returns:
            corrected_data (pd.DataFrame): DataFrame containing the data with corrected energy values.
        """
        # Copy the dataframe to avoid modifying the original data
        corrected_data = self.dataframe.copy()
        
        # Apply energy correction for dead hits
        corrected_data.loc[activeness == 0, 'Edep'] = 0.0
        
        return corrected_data
    
    def get_total_energy(self, corrected_data):
        """
        Calculate the total energy of each event with the dead layer effect.

        Parameters:
            corrected_data (pd.DataFrame): DataFrame containing the data with corrected energy values.

        Returns:
            total_energy (pd.DataFrame): DataFrame containing the total energy with the dead layer effect for each event.
        """
        # Calculate the total energy of each event by summing the corrected energy values
        total_energy = corrected_data.groupby(['event'], as_index=False)['Edep'].sum()
        return total_energy
    
    def get_momenta(self, activeness):
        """
        Get the momenta of the active events.

        Parameters:
            activeness (np.ndarray): Array of activeness values (0 or 1) indicating step deadness.

        Returns:
            momenta (pd.DataFrame): DataFrame containing the momenta of the active events.
        """
        momenta_data = self.dataframe.loc[activeness == 1]
        momenta = pd.DataFrame(momenta_data[['x', 'y', 'z', 'pdx', 'pdy', 'pdz']])
        return momenta
    
    def get_dead_event_positions(self, FCCD_value):
        """
        Get the positions of the dead events for a specific FCCD value.

        Parameters:
            FCCD_value (float): The FCCD value for which to filter dead hits.

        Returns:
            dead_hits_positions (pd.DataFrame): DataFrame containing the positions of the dead hits.
        """
        activeness, dead_hit = self.stepdeadness()
        # Filter dead hits based on the specified FCCD value
        dead_hits_positions = self.dataframe.loc[dead_hit & (self.dataframe['FCCD'] == FCCD_value), ['x', 'y', 'z']]
        return dead_hits_positions


# Create a folder to store the output files
output_folder = 'FCCD_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# FCCD values to loop through
FCCD_values = [0.00, 0.25, 0.50, 0.75, 1.00]

# Create an HDF5 file to store the processed data
output_filepath = os.path.join(output_folder, 'processed_data.hdf5')
with h5py.File(output_filepath, 'w') as f:

    # Loop through FCCD values and process data
    for FCCD in FCCD_values:
        # Create an instance of G4_Event with the current FCCD value
        event_handler = G4_Event(g4sdf, FCCD=FCCD)

        # Apply stepdeadness and get the activeness and dead_hit data
        activeness, dead_hit = event_handler.stepdeadness()

        # Apply energy cut and get the filtered data
        filtered_data = event_handler.energy_cut(activeness)
        
        # Correct the energy based on the step deadness information
        corrected_data = event_handler.correct_energy(activeness)

        # Get the momenta of the active events
        momenta = event_handler.get_momenta(activeness)
        
        # Get the total energy with the dead layer effect for each event
        total_energy = event_handler.get_total_energy(corrected_data)
        
        # Save the processed data to the HDF5 file
        group_name = f'FCCD_{FCCD:.1f}'
        f.create_dataset(group_name + '/event', data=total_energy['event'].values)
        f.create_dataset(group_name + '/energy', data=total_energy['Edep'].values)

        # Save the momenta data to the HDF5 file
        f.create_dataset(group_name + '/pdx', data=momenta['pdx'].values)
        f.create_dataset(group_name + '/pdy', data=momenta['pdy'].values)
        f.create_dataset(group_name + '/pdz', data=momenta['pdz'].values)
        # Save the 'x', 'y', and 'z' coordinates to the HDF5 file
        f.create_dataset(group_name + '/x', data=filtered_data['x'].values)
        f.create_dataset(group_name + '/y', data=filtered_data['y'].values)
        f.create_dataset(group_name + '/z', data=filtered_data['z'].values)
        
        # Save the activeness and dead_hit data to the HDF5 file
        f.create_dataset(group_name + '/activeness', data=activeness)
        f.create_dataset(group_name + '/dead_hit', data=dead_hit)
        
        # Save the positions of the dead hits for the current FCCD value
        dead_hits_positions = event_handler.get_dead_event_positions(FCCD_value=FCCD)
        f.create_dataset(group_name + '/dead_hits_x', data=dead_hits_positions['x'].values)
        f.create_dataset(group_name + '/dead_hits_y', data=dead_hits_positions['y'].values)
        f.create_dataset(group_name + '/dead_hits_z', data=dead_hits_positions['z'].values)


# Loop through FCCD values and create scatter plots for dead hit positions
for FCCD in FCCD_values:
    group_name = f'FCCD_{FCCD:.1f}'

    # Open the HDF5 file
    with h5py.File(output_filepath, 'r') as f:
        # Get the dead hit positions for the current FCCD value
        dead_hits_x = f[group_name + '/dead_hits_x'][:]
        dead_hits_y = f[group_name + '/dead_hits_y'][:]
        dead_hits_z = f[group_name + '/dead_hits_z'][:]

    # Create a scatter plot for dead hit positions
    plt.figure()
    plt.scatter(dead_hits_x, dead_hits_y, label='FCCD = %.2f' % FCCD)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Dead Hit Positions for FCCD = {FCCD}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'dead_hits_scatter_FCCD_{FCCD:.2f}.png')
    plt.close()

# Show all the scatter plots
plt.show()

print("Data processing and storage completed!")
