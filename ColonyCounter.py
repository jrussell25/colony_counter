import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import skimage
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny



class ColonyCounter():
    """
    Class for counting colonies on scanned plates
    """
    def __init__(self, plate_im):
        self.plate_im = rgb2gray(plate_im)
        self.colors = ["red","blue", "green", "magenta", "cyan", "yellow"]
        self.mask_made = False
    
    def get_centers(self, cx, cy, r_list,d=490):
        """
        Find the centers of the six individual plates
        The Hough Circle Transformation sometimes double counts plates
        before finding the centers of all plates. This checks that the 
        identified centers correspond to different plates
        """
        centers = [(cx[0],cy[0])]
        radii = [r_list[0]]
        for i,(x,y) in enumerate(zip(cx[1:],cy[1:])):
            new = True
            for c in centers:
                if np.sqrt((x-c[0])**2 + (y-c[1])**2) < d: 
                    new = False
                    break
            if new:
                centers.append((x,y))
                radii.append(r_list[i])
        centers = np.array(centers)
        self.dish_locs = (centers[:,0],centers[:,1],radii)
        self.N_plates = len(centers)
        print "{} Plates Detected".format(self.N_plates)        
    
    def circ_mask(self, shape, cx, cy, r):
        """
        Make circular masks to count each plate individually.
        """
    
        self.mask = np.zeros((len(cx),)+shape)
        X,Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
        
        for i,(x,y,rad) in enumerate(zip(cx,cy,r)):
            sing_mask = np.sqrt((X-x)**2+(Y-y)**2)
            self.mask[i] =(sing_mask < rad)


    def find_plates(self):
        """
        Find the possible centers of all plates and make masks of distinct plates.
        """
        edges = canny(self.plate_im)
        dish_rad = [480]
        hough_res = hough_circle(edges, dish_rad)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, dish_rad, num_peaks=12) #seemed to be a safe number
        
        self.get_centers(cx, cy, radii)
        
        CX, CY, R = self.dish_locs
        self.circ_mask(self.plate_im.shape, CX, CY, R)
        self.mask_made = True

    def count_colonies(self):
        """
        Count the colonies in a scanned image of plates. Should work with any number of plates.

        arguments:

        show_results : Do you want to see the colonies called for each plates. Default True. 
        """
        if not self.mask_made:
            self.find_plates()
    
        self.blobs = []
        for i in range(self.N_plates):
            trimmed = self.mask[i]*self.plate_im #Crop this image to make it faster (I think)

            blobs = blob_log(trimmed, max_sigma=15, min_sigma=2, threshold=0.1)
            blobs[:,2] = np.sqrt(2)*blobs[:,2]
            self.blobs.append(blobs)
            
            print str(len(blobs))+" Colonies Detected. Shown in {}".format(self.colors[i])

            
    def show_colonies(self):
        
        colors = ["red","blue", "green", "magenta", "cyan", "yellow"]
        fig, ax = plt.subplots(1,1)
        ax.imshow(self.plate_im,cmap="Greys")
        ax.set_axis_off()
        handles = []
        for i, blob_list in enumerate(self.blobs):
            line = mlines.Line2D([], [], color=colors[i], marker='o', linewidth=0, 
                                 markersize=10, fillstyle='none', label=str(len(blob_list)))
            handles.append(line)
            for blob in blob_list:
                y,x,r = blob
                C = plt.Circle((x,y),r, color = self.colors[i], linewidth=1, fill= False)
                ax.add_patch(C)
        plt.legend(handles=handles,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()    


