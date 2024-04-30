from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import os
import sys
import argparse
import configparser
import torch
import pickle
import lev
import random
import string
import numpy as np
import torch.utils.data as tud
import torch.optim as optim
from ctc_decoder import Decoder
from lm import utils
from torch import nn
from warpctc_pytorch import CTCLoss
from model import AttnEncoder, init_lstm_hidden
from torchvision import transforms
from threading import Thread
from batch_dense_opt import get_prior
device = 'cpu'
config = configparser.ConfigParser()
config.read("conf.ini")

model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']
hidden_size, attn_size, n_layers = model_cfg.getint('hidden_size'), model_cfg.getint('attn_size'), model_cfg.getint('n_layers')
prior_gamma = model_cfg.getfloat('prior_gamma')
batch_size = 1
char_list = lang_cfg['chars']

print(hidden_size)
print(attn_size)
print(n_layers)
print(len(char_list))
print(prior_gamma)

vocab_map, inv_vocab_map, char_list = utils.get_ctc_vocab(char_list)
encoder = AttnEncoder(hidden_size=hidden_size, attn_size=attn_size,
                          output_size=len(char_list), n_layers=n_layers,
                          prior_gamma=prior_gamma, pretrain=None)

encoder.to(device)
encoder.load_state_dict(torch.load("./data/iter/model_0/best.pth"))

print("encoder state dict loaded successfully")

transform = transforms.Compose([
    transforms.ToPILImage(),
    #dataset.Rescale(scale_range, hw_range, origin_scale=True),  # Resize images to 224x224         
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(1)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
imgs = []

def translation(imgs):
    #imgs = torch.tensor(imgs)
    batch = []
    priors = get_prior(imgs)
    priors = torch.tensor(priors).unsqueeze(0)
    for image in imgs:
        transformed = transform(image)
        #print(transformed.size())
        batch.append(transform(image))
    transformed_batch = torch.stack(batch, dim=0)
    #print("transformed_batch: ", transformed_batch.size())
    transformed_batch = transformed_batch.unsqueeze(0)
    #print("after unsqueeze: ", transformed_batch.size())
    #transformed_batch = torch.stack([transform(image) for image in imgs])
    print("Start predicting!!")
    encoder.eval()
    hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    larr, pred_arr, label_arr = [], [], []
    #for i_batch, sample in enumerate(transformed_batch):
        #imgs, priors, labels = sample['image'], sample['prior'], sample['label']
    with torch.no_grad():
        print(len(transformed_batch))
        h0 = init_lstm_hidden(n_layers, len(transformed_batch), hidden_size, device='cpu')
        #logits, probs, _, _ = encoder(imgs, h0, priors)
        #TODO: change None to priors
        print(priors.shape)
        print(transformed_batch.shape)
        logits, probs, _, _ = encoder(transformed_batch, h0, priors)
        logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)
    #print(probs)
    char_list = '_' + config['LANG']['chars']
    decoder = Decoder(char_list, blank_index=0)
    vocab_map, inv_vocab_map = decoder.char_to_int, decoder.int_to_char
    pred = decoder.greedy_decode(probs, digit=True)
    print("pred: ", pred)

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, backup_img = cap.read()

    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(backup_img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
        x_1 = bbox1[0]
        y_1 = bbox1[1]
        w_1 = bbox1[2]
        h_1 = bbox1[3]
        hand_img_1 = img[y_1:y_1+h_1, x_1:x_1+w_1, :]
        print(hand_img_1.shape)
        if len(hand_img_1) != 0:
            try:
                cv2.imshow('first hand', hand_img_1)
                np.save("hand.npy", hand_img_1)
            except:
                pass
        else:
            print("exception")
        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

        # Calculate distance between specific landmarks on the first hand and draw it on the image
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                  scale=10)
        #imgs.append(np.array(hand_img_1))
        imgs.append(np.array(backup_img))
        if len(imgs) == 5:
            t1 = Thread(target=translation, args=(imgs, ))
            t1.start()
            imgs = []
        # Check if a second hand is detected
        '''
        if len(hands) == 2:
            # Information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]
            x_2 = bbox2[0]
            y_2 = bbox2[1]
            w_2 = bbox2[2]
            h_2 = bbox2[3]
            hand_img_2 = img[y_2:y_2+h_2, x_2:x_2+w_2, :]
            if len(hand_img_2) != 0:
                try:
                    cv2.imshow('second hand', hand_img_2)
                except:
                    pass
            else:
                print("exception")
            # Count the number of fingers up for the second hand
            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Calculate distance between the index fingers of both hands and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                      scale=10)
        '''
        print(" ")  # New line for better readability of the printed output
    else: # no hands detected.
        try:
            cv2.destroyWindow("first hand")
            cv2.destroyWindow("second hand")
        except:
            pass
    # Display the image in a window
    cv2.imshow("Main Camera", img)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)