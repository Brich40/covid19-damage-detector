"""
Name : image_process.py.py
Author : OBR01
contact : oussama.brich@gmail.com
Time    : 20/11/2020 23:45
Desc:
"""

from bin.classes.image_process import ImageProcess

IMAGE_OUTPUT_PATH = '../data/result/'

if __name__ == '__main__':
    IMAGE_PATH = input("Entrez le chemin de l'image rayon X : ")

    result_path = ImageProcess.image_edge(IMAGE_PATH, IMAGE_OUTPUT_PATH)
    output_path = ImageProcess.convert_black_to_transparent(IMAGE_PATH, IMAGE_OUTPUT_PATH)
    lungs_info = ImageProcess.get_lung_contours(result_path, IMAGE_OUTPUT_PATH)
    lungs_info = ImageProcess.add_lung_damage(lungs_info)
    for lung_info in lungs_info:
        ImageProcess.add_damage_percent(lung_info)
        print("----------------------------------------------")
        print("Chemin d'image : " + lung_info['image_path'])
        print("Coté de poumon : " + lung_info['side'])
        print("Surface : " + str(lung_info['area']))
        print("Centre de poumon : (" + str(lung_info['x_center']) + "," + str(lung_info['y_center']) + ")")
        print("Dommage : " + str(lung_info['damage_percent']))
        print("----------------------------------------------")
