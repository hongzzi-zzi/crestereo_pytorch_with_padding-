from re import L
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from cv2 import imread
# from imread_from_url import imread_from_url

from nets import Model

import glob
from tqdm import tqdm
from nets.utils.utils import InputPadder
from PIL import Image


device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	print("Model Forwarding...")

	# 1인 차원 삭제하고 넘파이로 바꾸장
	left=left.squeeze().cpu().numpy()
	right=right.squeeze().cpu().numpy()

	# Return a contiguous array (ndim >= 1) in memory (C order).
	imgL = np.ascontiguousarray(left[None, :, :, :])
	imgR = np.ascontiguousarray(right[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

def load_image(imfile):
	img = np.array(Image.open(imfile)).astype(np.uint8) # (1512, 2016, 3)
	img = torch.from_numpy(img).permute(2, 0, 1).float() # torch.Size([3, 1512, 2016])
	return img[None].to(device)

if __name__ == '__main__':

	# 1500*2000
	left_img = load_image("/home/h/CREStereo/img/test/left_1500*2000.png")
	right_img = load_image("/home/h/CREStereo/img/test/right_1500*2000.png")
	in_h, in_w = left_img.shape[-2:]

	# 1504*2016
	padder = InputPadder(left_img.shape, divis_by=32)
	left_img, right_img = padder.pad(left_img, right_img)

	model_path = "models/crestereo_eth3d.pth"
	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()

	pred = inference(left_img, right_img, model, n_iter=20)
	disp=pred
	
	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	disp_vis = disp_vis.astype("uint8")
	# disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
 
	disp_vis=cv2.resize(disp_vis,dsize=(1500,2000),interpolation=cv2.INTER_AREA)
	# 1500*2000
	cv2.imwrite("output.jpg", disp_vis)
	
	# # 1504*2016
	# disp.tofile('out.raw')

	disp=torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)
	disp=padder.unpad(disp).cpu().numpy()
	
	# 1500*2000
	disp.tofile('outt.raw')

	## 이 뒤에서 저장하면 1*1 됨,,,,,

	
	print('fin')
