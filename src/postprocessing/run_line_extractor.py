import rasterio
import numpy
# import cv2.ximgproc
from PIL import Image
from PIL import ImageDraw
from skimage import img_as_float
from skimage import io, color, morphology
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, hough_line
import logging
import matplotlib.pyplot as plt

# example given in https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf

w_folder = 'C:\\Users\\kuzind\\Downloads\\lines_postprocessing'
logits_tif_path = w_folder+"\\prediction_raw_front_range_on_6.tiff.tif"

def run_open():
    with rasterio.open(logits_tif_path) as src:
        band1: numpy.ndarray = src.read(1)  # read first band
        assert band1.dtype == numpy.float64
        assert len(band1.shape) == 2
        band1 = 1/(1 + numpy.exp(-band1))  # probabilities of fault
        band1 = band1 > 0.5  # threshold probabilities
        Image.fromarray(band1).save(w_folder + '\\thrs.png')
        logging.info('thinning..')
        out_thin = morphology.thin(band1)
        Image.fromarray(out_thin).save(w_folder + '\\thin.png')
        # plt.imshow(out_thin)
        # plt.show()
        # skimage.measure.approximate_polygon()

        # todo check parameters here
        lines = probabilistic_hough_line(out_thin, threshold=10, line_length=5,
                                         line_gap=3)
        im_hough = Image.fromarray(out_thin).convert("RGB")
        for line in lines:
            p0, p1 = line
            ImageDraw.Draw(im_hough).line(
                ((p0[0], p0[1]), (p1[0], p1[1])), fill='orange',width=1)
        im_hough.save(w_folder + '\\hough.png')
        # cv2.ximgproc.thinning(band1)
        # max_val = numpy.max(band1_probs)
        # min_val = numpy.min(band1_probs)


        # print(max_val)
        # print(min_val)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_open()





