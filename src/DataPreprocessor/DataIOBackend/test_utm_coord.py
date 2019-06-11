import numpy as np

from src.DataPreprocessor.DataIOBackend.utm_coord import UtmCoord


def test_process_content_one_geometry():
    geotransform = (414000.0, 30.0, 0.0, 3741000.0, 0.0, -30.0)
    utm_coord = UtmCoord(
        top_left_x=geotransform[0],
        west_east_pixel_resolution=geotransform[1],
        top_left_y=geotransform[3],
        north_south_pixel_resolution=geotransform[5])

    content = \
        ['>\n',
         '>\n',
         '>\n',
         '           505499.81          3665780.16      45\n',
         '           505429.63          3665490.93      45\n',
         '           505112.33          3664450.54      45\n']

    true_geometries = [
        [
            utm_coord.transform_coordinates(505499.81, 3665780.16),
            utm_coord.transform_coordinates(505429.63, 3665490.93),
            utm_coord.transform_coordinates(505112.33, 3664450.54),
        ]
    ]
    geometries = utm_coord.process_content(content)
    assert np.allclose(true_geometries, geometries)


def test_process_content_multiple_geometries():
    geotransform = (400020.0, 30.0, 0.0, 4550010.0, 0.0, -30.0)
    utm_coord = UtmCoord(top_left_x=geotransform[0],
                         west_east_pixel_resolution=geotransform[1],
                         top_left_y=geotransform[3],
                         north_south_pixel_resolution=geotransform[5])

    content = \
        ['>\n',
         '           340832.57          3848272.08      11\n',
         '           341088.61          3848201.08      11\n',
         '>\n',
         '           343855.80          3847255.04      11\n',
         '>\n',
         '           347082.85          3845902.83      11\n',
         '           346790.46          3845996.45      11\n',
         '           346509.80          3846134.25      11\n']

    true_geometries = [
        [
            utm_coord.transform_coordinates(340832.57, 3848272.08),
            utm_coord.transform_coordinates(341088.61, 3848201.08),
        ],
        [
            utm_coord.transform_coordinates(343855.80, 3847255.04),
        ],
        [
            utm_coord.transform_coordinates(347082.85, 3845902.83),
            utm_coord.transform_coordinates(346790.46, 3845996.45),
            utm_coord.transform_coordinates(346509.80, 3846134.25),
        ]
    ]
    geometries = utm_coord.process_content(content)
    assert [np.allclose(true_geometry, geometry)
            for true_geometry, geometry in zip(true_geometries, geometries)]
