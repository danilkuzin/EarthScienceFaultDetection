import geopandas
import rasterio

# please adjust it to match the path to the data on your computer
import shapely

data_path = "../DataForEarthScienceFaultDetection"
region_data_folder = "Region 12 - Nothern California"
input_path = f'{data_path}/' \
             f'raw_data/{region_data_folder}'
file = "HAZMAP.kml"

dataset = rasterio.open(f'{data_path}/raw_data/'
                        f'{region_data_folder}/r_landsat.tif')
latlon_bounds = rasterio.warp.transform_bounds(
    dataset.crs, "EPSG:4326", *dataset.bounds)

# read data into a table format, each row correspond to a fault line,
# columns are available metadata in kml
data = geopandas.read_file(f'{input_path}/{file}')

# we filter data according to their coordinates to leave out any faults
# that are outside of our current region area
filtered_data = data[data.intersects(shapely.geometry.box(*latlon_bounds))]

filtered_data = filtered_data.to_crs(dataset.crs)

strike_slip_fault_lines = []
thrust_fault_lines = []

# looping over each fault
for index in range(filtered_data.shape[0]):
    # fault_line_data would contain one row from the whole table
    # we have read from the kml
    fault_line_data = filtered_data.iloc[index]
    # now we can access different fields of metadata for this fault
    # line by fault_line_data["<name_of_the_field>"] (without <>),
    # e.g., fault_line_data['disp_slip_']. See example below


    # HAZMAP.kml contains fault lines into two types: MultiLineString and
    # LineString. Processing is the same apart that for MultiLineString
    # we need to loop over different lines in each multilinestring
    if fault_line_data['geometry'].type == 'MultiLineString':
        for line in fault_line_data['geometry']:
            # the part below is a simplified version of where we convert utm
            # coordinates into pixel coordinates. Since this should not affect
            # the filtering part, we have left actual conversion out
            utm_coord_list = list(line.coords)
            coords = []
            for point_utm in utm_coord_list:
                coords.append(point_utm[0], point_utm[1])
            # end of the part of getting coordinates for each point in the line

            # actual filtering
            if fault_line_data['disp_slip_'] == 'strike slip':
                strike_slip_fault_lines.append(coords)
            elif fault_line_data['disp_slip_'] == 'thrust':
                thrust_fault_lines.append(coords)
            else:
                # we skip coords if they are not from strike slip or
                # thrust fault
                print('UNEXPECTED FAULT TYPE!')
    else:
        # and the same for LineString with the only difference that
        # we do not have a loop over multiple lines
        utm_coord_list = list(fault_line_data['geometry'].coords)
        coords = []
        for point_utm in utm_coord_list:
            coords.append(point_utm[0], point_utm[1])
        if fault_line_data['disp_slip_'] == 'strike slip':
            strike_slip_fault_lines.append(coords)
        elif fault_line_data['disp_slip_'] == 'thrust':
            thrust_fault_lines.append(coords)
        else:
            print('UNEXPECTED FAULT TYPE!')


