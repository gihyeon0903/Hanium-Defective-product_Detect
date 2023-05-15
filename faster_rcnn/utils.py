import torch

import cv2

def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()): # 0, conv
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def init_weight(model):
    for param in model.rpn.parameters():
        torch.nn.init.normal_(param,mean = 0.0, std=0.01)

    for name, param in model.roi_heads.named_parameters():
        if "bbox_pred" in name:
            torch.nn.init.normal_(param,mean = 0.0, std=0.001)
        elif "weight" in name:
            torch.nn.init.normal_(param,mean = 0.0, std=0.01)
        if "bias" in name:
            torch.nn.init.zeros_(param)

def makeBox(im,bbox):
    image = im.copy()
    try:
        bbox = bbox[0]['boxes']
        for b_ in bbox:
            cv2.rectangle(image,(int(b_[0]),int(b_[1])),(int(b_[2]),int(b_[3])),color = (1,0,0),thickness = 1)
    except:
        for b_ in bbox:
            cv2.rectangle(image,(int(b_[0]),int(b_[1])),(int(b_[2]),int(b_[3])),color = (1,0,0),thickness = 1)
    return image

def Total_Loss(loss, lambda_rpn=10, lambda_header=1):
    # rpn
    loss_objectness = loss['loss_objectness']
    loss_rpn_box_reg = loss['loss_rpn_box_reg']
    
    # roi_heads
    loss_classifier = loss['loss_classifier']
    loss_box_reg = loss['loss_box_reg']

    rpn_total = loss_objectness + lambda_rpn*loss_rpn_box_reg
    fast_rcnn_total = loss_classifier + lambda_header*loss_box_reg

    total_loss = rpn_total + fast_rcnn_total

    return total_loss