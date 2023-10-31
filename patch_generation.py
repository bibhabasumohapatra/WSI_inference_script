import numpy as np

from skimage.filters import threshold_li
from skimage import color, measure, io, transform
import cv2
import openslide


class ImageReader:

    def __init__(self, image_path : str, tile_size : int, scale_factor : int,):

        self.reader = openslide.open_slide(image_path) #ImageFormatReader(image_path, tile_size, scale_factor, create_single_ROI=True)
        self.tile_size = tile_size

    def get_mask(self, magnification : int ):
        wsi_1_25x = self.reader.read_region((0,0),5,self.reader.level_dimensions[5]) #self.reader.getFullImageDataFromResolution(1,1.25)
        wsi_1_25x = np.asarray(wsi_1_25x)

        mask_g = wsi_1_25x[:,:,1] < threshold_li(wsi_1_25x[:,:,1] ).astype(np.uint8)
        mask = mask_g.astype(np.uint8)*255

        #closing
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=15)

        ## opening
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel,iterations=75)

        infer_tile_list = []
        scaling = (magnification/1.25)
        steps = int(self.tile_size/scaling)

        ## padding
        H, W = mask.shape
        extra_bottom = int(np.ceil(H/steps)*steps - H)
        extra_right = int(np.ceil(W/steps)*steps - W)

        mask = np.pad(
            mask,
            (
                (0, extra_bottom), (0, extra_right)),
                mode="constant",
                constant_values=0,
            )
        # update H,W
        H,W = mask.shape

        for height in range(0,H,steps):
            for width in range(0, W,steps):                 
                 if np.sum(mask[height:height+steps, width:width+steps]) > 0:
                    infer_tile_list.append([int(height*scaling),int(width*scaling),])

        return {
            "img":wsi_1_25x,
            "mask": mask,
            "list_indices":infer_tile_list,
            "shape": mask.shape,
            "step_size": steps,
            "scaling":scaling,
        }
    
    def get_tiles(self, y:int, x:int):

        # Tile_array = self.reader.getImageDataFromIFD(
        #     5,
        #     x, y, 
        #     0,
        #     self.tile_size, self.tile_size, coordType=1,
        # )
        Tile_array = self.reader.read_region((x*4,y*4),2,(1024,1024))
        Tile_array = np.asarray(Tile_array)
        Tile_array = Tile_array[:, :, :3]

        return Tile_array
    
    def get_stitiched(image_list, coords_list, mask_shape, step_size, scaling):  ## Seems No Bug  Checked by replacing patch variable as white masks of ones

        empty_mask = np.zeros(mask_shape)

        for batch in range(len(image_list)):
            for indx,coords in enumerate(coords_list[batch]):
                patch = transform.resize(image=image_list[batch][indx],output_shape=(step_size,step_size),mode="constant")
                empty_mask[int(coords[0]/scaling):int(coords[0]/scaling) + step_size,int(coords[1]/scaling):int(coords[1]/scaling) + step_size] = patch

        return empty_mask