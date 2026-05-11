import os
import image_bbox_tiler as ibis

# run in pytorch2 interpreter ************************

#os.chdir("E:/a_detection_of_seabirds/new_tiles_test1")
## error in Main.py, line 273 is good;  line 300-305 is issue

# im_src = image source, an_src = annotation source
im_src = "C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/new_train_parents/"
an_src = "C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/voc_parents/"
# error is below
#os.listdir('E:\\detection_of_seabirds\\demo_parents\\C1_L1_F10_T20230621_132003_892\\')
#os.listdir ('E:/detection_of_seabirds/demo_parents')

# Enter destination
im_dst = "C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/new_train_tiles/"
an_dst = "C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/new_train_annot/"

im_list = os.listdir(im_src)
an_list = [x.replace("xml", "jpg") for x in os.listdir(an_src)]
list(set(im_list) - set(an_list))
list(set(an_list) - set(im_list))

#os.mkdir(im_dst)
#os.mkdir(an_dst)

slicer = ibis.Slicer()
slicer.config_dirs(img_src=im_src, ann_src=an_src, img_dst=im_dst, ann_dst=an_dst)
slicer.keep_partial_labels = True
slicer.save_before_after_map = True

# Slice images and annotationsl change empty sample
slicer.slice_by_size(tile_size=(1024,1024), tile_overlap=0.0, empty_sample=0.0)

# Slice images only
#slicer.slice_images_by_size((tile_size==(1024,1024), tile_overlap == 0.0))


#slicer.slice_by_size(tile_size=(1024,1024), tile_overlap=0.0, empty_sample= 0.0)