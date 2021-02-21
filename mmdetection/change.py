import torch 
pretrained_weights = torch.load('checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

num_class = 1
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].resize_(num_class*4)

torch.save(pretrained_weights, "faster_rcnn_r50_fpn_1x_%d.pth"%num_class)