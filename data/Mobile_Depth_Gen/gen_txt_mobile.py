import os
from tqdm import tqdm


path = os.path.abspath('./')
setting = ['Aligned', 'Photos_Calibration_Results/calibration']
calibration_scene = [x for x in os.listdir(os.path.join(path, setting[1]))]
Fig = ['Figure1', 'Figure3', 'Figure5', 'Figure6', 'Figure7']
Scenes = ['GT', 'GTLarge', 'GTSmall', 'balls', 'bottles', 'fruits', 'keyboard', 'metals', 'plants', 'telephone', 'window']

for fig in tqdm(Fig):
    scenes = [os.path.join(path, setting[0], fig, x) for x in os.listdir(os.path.join(path, setting[0], fig)) if x in calibration_scene]
    for scene in scenes:
        rgbs_aligned = [os.path.join(scene, x) for x in sorted(os.listdir(scene)) if x.endswith('jpg') and x.startswith('a')]
        rgbs = [os.path.join(scene, x) for x in os.listdir(scene) if x.endswith('jpg') and not x.startswith('a')]
        calibratied_txt = os.path.join(path, setting[1], scene.split('/')[-1], 'calibrated.txt')
        focal_depth = []
        aperture = None
        with open(calibratied_txt, 'r') as F:
            for line in F.readlines():
                focal_depth.append(line.strip().split()[0])
                if not aperture:
                    aperture = line.strip().split()[1]
        focal_length = focal_depth[-1]
        focal_depth = focal_depth[:-1]
        for i in range(len(rgbs_aligned)-1):
            with open('./Mobile_Img_Aligned_path.txt', 'a') as F:
                F.write('{} '.format(rgbs_aligned[i]))
            with open('./Mobile_Img_path.txt', 'a') as F:
                F.write('{} '.format(rgbs[i]))
            with open('./Mobile_FP.txt', 'a') as F:
                F.write('{} '.format(focal_depth[i]))
        with open('./Mobile_Img_Aligned_path.txt', 'a') as F:
            F.write('{}\n'.format(rgbs_aligned[-1]))
        with open('./Mobile_Img_path.txt', 'a') as F:
            F.write('{}\n'.format(rgbs[-1]))
        with open('./Mobile_FP.txt', 'a') as F:
            F.write('{}\n'.format(focal_depth[-1]))
