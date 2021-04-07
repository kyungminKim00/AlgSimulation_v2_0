import re
import os
from PIL import Image
import glob
import platform

def animate_gif(source_dir, multiple=3, duration=100):
    
    for dir_name in os.listdir(source_dir):
        for target, mode in [['/fig_index/return', 'test'] , ['/validation/fig_index/return', 'validation']]:
            img, *imgs = None, None
            t_dir = source_dir + '/' + dir_name + target
            
            # filepaths
            fp_in = t_dir + '/*.jpeg'
            fp_out = t_dir + '/' + dir_name + '_' + mode + '.gif'

            try:
                print("Reading:{}".format(fp_in))
                img, *imgs = [Image.open(f).resize((int(360*multiple), int(240*multiple))) for f in sorted(glob.glob(fp_in))]
                img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=duration)
                img.close()
            except ValueError:
                print("ValueError:{}".format(fp_in))
                pass

if __name__ == "__main__":
    animate_gif('./save/result', multiple=3, duration=100)