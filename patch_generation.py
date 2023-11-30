import numpy as np

from skimage.filters import threshold_li
from skimage import color, measure, io, transform
import cv2
import openslide
from ImageFormatReader import ImageFormatReader

class ImageReader:

    def __init__(self, image_path : str, tile_size : int, padding : int = None):

        self.reader = openslide.open_slide(image_path) 
        self.max_mag = int(ImageFormatReader(image_path, 1024, 4, create_single_ROI=True).getMaxResolution())

        self.tile_size = tile_size
        self.padding = padding

        if self.max_mag == 40:
            self.level_10 = 2 
            self.level_2_5 = 4
            self.level_shift = 4
        elif self.max_mag == 20:
            self.level_10 = 1
            self.level_2_5 = 3
            self.level_shift = 2

    def pad(self, mask , left : int, right : int, top : int, bottom : int):
        mask = np.pad(
            mask,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0,
            )
        return mask

    def get_mask(self, magnification : int ):
        wsi_2_5x = self.reader.read_region((0,0),self.level_2_5,self.reader.level_dimensions[self.level_2_5]) 
        
        wsi_2_5x = np.asarray(wsi_2_5x)

        mask_g = wsi_2_5x[:,:,1] < threshold_li(wsi_2_5x[:,:,1]).astype(np.uint8)
        mask = mask_g.astype(np.uint8)*255

        #closing
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=15)

        ## opening
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel,iterations=75)

        infer_tile_list = []
        scaling = (magnification/2.5)
        steps = int(self.tile_size/scaling)

        ## padding
        org_shape = mask.shape
        H, W = org_shape

        extra_bottom = int(np.ceil(H/steps)*steps - H)
        extra_right = int(np.ceil(W/steps)*steps - W)

        mask = self.pad(mask, 0, extra_right, 0, extra_bottom)

        # update H,W
        H,W = mask.shape

        for height in range(0,H,steps):
            for width in range(0, W,steps):                 
                 if np.sum(mask[height:height+steps, width:width+steps]) > 0:
                    infer_tile_list.append([int(height*scaling),int(width*scaling),])

        return {
            "img":wsi_2_5x,
            "mask": mask,
            "list_indices":infer_tile_list,
            "shape": mask.shape,
            "step_size": steps,
            "scaling":scaling,
        }
    
    def get_tiles(self, y:int, x:int):

        if self.padding is not None:
            Tile_array = self.reader.read_region(((x - self.padding)*self.level_shift,(y - self.padding)*self.level_shift),self.level_10,(self.tile_size + self.padding,self.tile_size + self.padding))  ## (1024)
            Tile_array = np.asarray(Tile_array)
        else:
            Tile_array = self.reader.read_region((x*self.level_shift,y*self.level_shift),self.level_10,(self.tile_size,self.tile_size))
            Tile_array = np.asarray(Tile_array)
        
        Tile_array = Tile_array[:, :, :3]

        return Tile_array
    
    def get_stitiched(self,image_list, coords_list, mask_shape, step_size, scaling):  
        empty_mask = np.zeros(mask_shape)
        for indx,coords in enumerate(coords_list):
            if self.padding is not None:
                image_cropped = image_list[indx][self.padding:self.padding+self.tile_size, self.padding:self.padding+self.tile_size]
                patch = transform.resize(image=image_cropped,output_shape=(step_size,step_size),
                                         order=0,
                                         preserve_range=True,
                                         mode="constant")
            else:
                patch = transform.resize(image=image_list[indx],output_shape=(step_size,step_size),
                                         order=0,
                                         preserve_range=True,
                                         mode="constant")
                
            empty_mask[int(coords[0]/scaling):int(coords[0]/scaling) + step_size,int(coords[1]/scaling):int(coords[1]/scaling) + step_size] = patch

        return empty_mask
    
