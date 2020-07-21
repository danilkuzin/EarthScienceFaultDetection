1. Sasaki K, Iizuka S, Simo-Serra E, Ishikawa H. Joint gap detection and inpainting of line drawings. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017 (pp. 5725-5733) - claim that they are the first to have impainting of line drawings. Not exactly, our case, but maybe used in the part of what we suggested to add in the previous email - kind of postprocessing of CNN detections to form a line. In this sense raw CNN detections are lines with gaps and we need to impaint those gaps, and this is not a trivial task according to this paper. http://openaccess.thecvf.com/content_cvpr_2017/html/Sasaki_Joint_Gap_Detection_CVPR_2017_paper.html 
2. Farahbakhsh E, Chandra R, Olierook HK, Scalzo R, Clark C, Reddy SM, Muller RD. Computer vision-based framework for extracting geological lineaments from optical remote sensing data. arXiv preprint arXiv:1810.02320. 2018 - claim that linear "features" (lines) are difficult to extract in geological context. They extract geological lineaments - as far as I got this is any lines, including faults but I am not sure that these are the same faults as ours. Moreover, in their claim that linear features are difficult to extract they refer to the paper from 1993. They also use "classical" computer vision methods such as Canny edge detector.  https://arxiv.org/abs/1810.02320
3. Seifozzakerini S, Yau WY, Zhao B, Mao K. Event-Based Hough Transform in a Spiking Neural Network for Multiple Line Detection and Tracking Using a Dynamic Vision Sensor. In BMVC 2016 - do (straight) line detection https://pdfs.semanticscholar.org/b2bd/db5c5e3510457de00046e9a2acf18e7fd31e.pdf
4. Zhang Z, Liu Q, Wang Y. Road extraction by deep residual u-net. IEEE Geoscience and Remote Sensing Letters. 2018;15(5):749-53 - road extraction (weird form object detection) with U-nets. https://arxiv.org/pdf/1711.10684.pdf
5. Hou Q, Cheng MM, Hu X, Borji A, Tu Z, Torr PH. Deeply supervised salient object detection with short connections. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2017 (pp. 3203-3212) - sailent object detection (weird form object detection) with fully connected CNN, address the scaling problem. http://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf
6. Xie S, Tu Z. Holistically-nested edge detection. InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 1395-1403) - edge detection with fully connected CNN, address the scaling problem. http://openaccess.thecvf.com/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf
7. Luc P, Couprie C, Chintala S, Verbeek J. Semantic segmentation using adversarial networks. arXiv preprint arXiv:1611.08408 - adversarial learning. https://arxiv.org/pdf/1611.08408.pdf
8. https://thegradient.pub/semantic-segmentation/ - overview of semantic segmentation method including references to end-to-end CNN + CRF
9. Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence. 2017 Apr 27;40(4) - CNN for semantic segmentation + CRF as postprocessing to improve detections. Similar to what we discuss that we can use some kind of postprocessing with possibly GP to produce lines without holes. https://arxiv.org/pdf/1606.00915.pdf