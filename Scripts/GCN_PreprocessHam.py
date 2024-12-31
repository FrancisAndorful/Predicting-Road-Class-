
# LOADING LIBRARIES
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.geometry import Point,linestring

# # Load road data and point data
road_data = gpd.read_file("./Hamburg_road.geojson")#.explode(index_parts=True)
point_data = gpd.read_file("./Hamburg_Amenity.geojson")

# SELECT ONLY NECCESSARY COLUMNS
road_data = road_data[['@osmId', 'highway', 'geometry']]
point_data = point_data[['@osmId', 'amenity', 'geometry']]


# Check if either road_data.crs or point_data.crs is geographic (4326)
if road_data.crs.is_geographic or point_data.crs.is_geographic:
    road_data = road_data.to_crs(32632)
    point_data = point_data.to_crs(32632)

# Build a spatial index for road data
road_spatial_index = road_data.sindex
# Build a spatial index for point data
point_spatial_index = point_data.sindex

# Buffer the road data
buffer_distance = 100  # in meters
road_data_buffered = road_data.copy()
road_data_buffered['geometry'] = road_data['geometry'].buffer(buffer_distance)

# Spatial join to get points within each buffer
joined_data = gpd.sjoin(point_data, road_data_buffered, how='left', predicate='within')


# Build a spatial index for joined_data
joined_data_spatial_index = joined_data.sindex

# Create an 'index' column based on the index of the road_data
joined_data['index'] = joined_data['index_right']

# Group by road segment and count points for each group (based on the spatial index)
grouped_data = joined_data.groupby('index')['amenity'].value_counts().unstack(fill_value=0)

# Reset the index of road_data
road_data = road_data.reset_index(drop=True)

# Merge the counts back to the road_data
road_data_result = road_data.merge(grouped_data, left_index=True, right_index=True, how='left')



road_data_result['segment_length'] = round(road_data_result['geometry'].length / 1000, 2)

print('STEP 3: Creating toppology information.....')
# Create a dictionary to store connections
connections = {}

# Assuming 'geometry', '@osmId', and 'highway' columns exist in road_data_result
road_data_result['TopologyInfo'] = None  # Initialize the TopologyInfo column

# Create a spatial index for efficient spatial queries
spatial_index = road_data_result.sindex

# Iterate through each road segment
for idx, segment in road_data_result.iterrows():
    connections[idx] = []

    # Use spatial index for faster intersection checks
    possible_matches_index = list(spatial_index.intersection(segment['geometry'].bounds))

    # Check connections with other segments
    for other_idx in possible_matches_index:
        if idx != other_idx and segment['geometry'].intersects(road_data_result.loc[other_idx, 'geometry']):
            connections[idx].append({
                # 'ID': road_data_result.loc[other_idx, '@osmId'],
                 road_data_result.loc[other_idx, 'highway'],
                # 'ConnectionType': 'Example Connection Type'
            })

# Update the GeoDataFrame with topology information
road_data_result['TopologyInfo'] = road_data_result.index.map(connections)



# Save the result to a new GeoJSON file
road_data_result.to_csv("GCN_Hamburg_Data.csv")

# Save or use the updated GeoDataFrame
# road_data_result.to_file("updated_road_data.geojson", driver="GeoJSON")
