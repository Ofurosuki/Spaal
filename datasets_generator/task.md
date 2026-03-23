0227_inference_and_eval.shを基に

MODEL_CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
MODEL_CHECKPOINT="./nuscenes/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

MODEL_CONFIG="configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
MODEL_CHECKPOINT="./nuscenes/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"

の2パターンのモデルについて、/data2/yoshida/swin_denoised_64_bin 以下のdirについて推論。以下に0_2,0_8,1,2,5,11,22,45のsubdirがあってそれぞれについて二種の推論について実施。convert_pvrcnn_classnames.pyはpvrcnnの時だけでよい。
dockerは/data2/yoshidaがdocker視点で./nuscenesにアタッチされていることに注意