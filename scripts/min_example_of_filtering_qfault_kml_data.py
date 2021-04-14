import geopandas
import rasterio
import rasterio.warp
import shapely
import shapely.geometry
import fiona
import pandas as pd

# please adjust it to match the path to the data on your computer
data_path = "../../DataForEarthScienceFaultDetection"
region_data_folder = "Region 12 - Nothern California"
input_path = f'{data_path}/' \
             f'labels_from_Philip/Faults/'
input_file = "QFaults.kml"
output_file = "QFaults_filtered.kml"
source_tif_path = f'{data_path}/raw_data/' \
                  f'{region_data_folder}/r_landsat.tif'

fiona.drvsupport.supported_drivers['libkml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

dataset = rasterio.open(source_tif_path)
latlon_bounds = rasterio.warp.transform_bounds(
    dataset.crs, "EPSG:4326", *dataset.bounds)

for layer in fiona.listlayers(f'{input_path}/{input_file}'):
    # layer is the type of faults, e.g. 'Latest Quaternary'
    # if required there can be done additional filtering based on layer

    # read data into a table format, each row correspond to a fault line,
    # columns are available metadata in kml
    data = geopandas.read_file(
        f'{input_path}/{input_file}', driver='LIBKML', layer=layer)

    # we filter data according to their coordinates to leave out any faults
    # that are outside of our current region area
    filtered_data = data[data.intersects(shapely.geometry.box(*latlon_bounds))]

    remaining_indices = []

    # looping over each fault
    for index in range(filtered_data.shape[0]):
        # fault_line_data would contain one row from the whole table
        # we have read from the kml
        fault_line_data = filtered_data.iloc[index]
        # now we can access different fields of metadata for this fault
        # line by fault_line_data["<name_of_the_field>"] (without <>),
        # e.g., fault_line_data['disp_slip_']. See example below

        # parse metadata of the fault. It will return the list of two
        # pandas dataframes (tables), the first one is unclear,
        # the second contains fields we are insterested in such as "Linetype"
        description = pd.read_html(
            fault_line_data["description"])[1].set_index(0)

        # example of filtering data based on metadata.
        # For example, we keep only Linetype <> "Inferred" and
        # Mapping Certainty == "Good"

        # .values[0] would give a specific value for the field
        if description.loc["Linetype"].values[0] != "Inferred" and \
                description.loc["Mapping Certainty"].values[0] == "Good":
            remaining_indices.append(index)

        # the code below is filtering based on HAZMAP information
        # # actual filtering
        # if fault_line_data['disp_slip_'] == 'strike slip':
        #     strike_slip_fault_lines.append(coords)
        # elif fault_line_data['disp_slip_'] == 'thrust':
        #     thrust_fault_lines.append(coords)
        # else:
        #     # we skip coords if they are not from strike slip or
        #     # thrust fault
        #     print('UNEXPECTED FAULT TYPE!')

    # filter data according to saved indices
    filtered_data = filtered_data.iloc[remaining_indices]

    # write data for this layer
    if filtered_data.shape[0] > 0:
        filtered_data.to_file(
            f'{input_path}/{output_file}', driver='LIBKML', layer=layer)



