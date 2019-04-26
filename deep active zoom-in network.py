

import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,Dropout
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import random
import os

import numpy as np
import cv2

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



class Reinforcement(object):
    def get_full_frame(self, state):
        input_path = "./data/gt_img/zoom%d/blurred_full_img/" % state[0]
        img_name = "zoom%d_%d.png" % (state[0],state[1])
        img = cv2.imread(input_path+img_name,1)
        img = np.reshape(img, (1, 240, 240, 3))
        return img
    
    def get_zoomIn_frame(self,state, action):
        input_path = "./data/gt_img/zoom%d/z%d/" % (state[0], action)
        img_name = "z%d_%d.png" % (action, state[1])
        img = cv2.imread(input_path+img_name,1)
        img = np.reshape(img, (1, 120, 120, 3))
        return img
    

# Agent role in reinfrocement learning
class ZoomIn_Action_Network(Reinforcement):
    def __init__(self):
        self.data_que = deque(maxlen=2000)
        self.eps = 1.0  # exploration rate
        self.eps_min = 0.01
        self.eps_decay = 0.95
        self.model = self._load_model()
        self.feature_extractor = self._load_feature_extractor()

    def _load_model(self):
        model = Sequential()  
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(60, 60, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['acc'])

        return model
    
    def _load_feature_extractor(self): 
        image_size = 120 
        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        model = Sequential()
        model.add(vgg16_model)
        model.add(Flatten())
        return model
    
    def split4imgs(self, full_img):
        full_img = cv2.resize(full_img, (120,120), interpolation = cv2.INTER_AREA)
        full_img = full_img.astype('float32')
        full_img/=255.
        four_imgs = []
        four_imgs.append(np.reshape(full_img[:60,:60,:], (1, 60, 60, 3))) # left, top
        four_imgs.append(np.reshape(full_img[:60,60:,:], (1, 60, 60, 3))) # right, top
        four_imgs.append(np.reshape(full_img[60:,:60,:], (1, 60, 60, 3))) # left, bottom
        four_imgs.append(np.reshape(full_img[60:,60:,:], (1, 60, 60, 3))) # right, top
        return four_imgs
        

    def append_data(self, state, action, reward):
        self.data_que.append((state, action, reward))

        
    def act(self, state):
        if np.random.rand() <= self.eps:
            return (random.randrange(4)+1)
        
        full_img = self.get_full_frame(state)[0]
        four_imgs = self.split4imgs(full_img)
        pred_arr = np.zeros((1,4), dtype=np.float32)
    
        for i in range(4):
            pred_arr[0,i] = self.model.predict(four_imgs[i])
            
        return (np.argmax(pred_arr[0])+1)  # returns action

    
    def batch_train(self):
        X_train = np.zeros((32,60,60,3), dtype=np.float32)
        y_train = np.zeros((32,1), dtype=np.float32)

        minibatch = random.sample(self.data_que, 8)
        idx=0
        cnt=0
        while idx < len(minibatch):
            state, action, reward = minibatch[idx]
            
            full_img = self.get_full_frame(state)[0]
            four_imgs = self.split4imgs(full_img)
            pred_arr = np.zeros((1,4), dtype=np.float32)

            pred_arr[0,0] = self.model.predict(four_imgs[0])
            pred_arr[0,1] = self.model.predict(four_imgs[1])
            pred_arr[0,2] = self.model.predict(four_imgs[2])
            pred_arr[0,3] = self.model.predict(four_imgs[3])

            
            pred_arr[0,action] = 1.0
            for i in range(4): 
                if i != action:
                    pred_arr[0,i]=0.0
            
            
            X_train[cnt:cnt+4,:] = four_imgs[:]
            y_train[cnt,:] = pred_arr[0,0]
            y_train[cnt+1,:] = pred_arr[0,1]
            y_train[cnt+2,:] = pred_arr[0,2]
            y_train[cnt+3,:] = pred_arr[0,3]
            
            cnt+=4
            idx+=1
            
        self.model.fit(X_train, y_train, epochs=1, verbose=0)
            
        if self.eps > self.eps_min:
            self.eps = self.eps * self.eps_decay


    def load_zoomIn_model(self, name):
        self.model.load_weights(name)

    def save_zoomIn_model(self, name):
        self.model.save_weights(name)
        


        
# Enviornment role in reinforcement learning       
class Classifier(Reinforcement):
    def __init__(self):
        self.classifier = self.load_pretrained_classifier()
        
    def load_pretrained_classifier(self):
        # load json and create model
        json_file = open('classifier.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("classifier.h5")
        return loaded_model
    
    
    def take_action(self, action, state):
        # select random pictures [zoomIn#, time]
        next_state = random.sample([1,2,3,4],1) + [state[1]+1]

        # select zoom in image
        zoom_in_img = self.get_zoomIn_frame(state,action)
        new_reward = self.classifier.predict(zoom_in_img).item(0)
        
        # reward threshold: 0.7
        if new_reward <0.7:
            new_reward=0
            

        return next_state, new_reward

    



if __name__ == "__main__":
    agent = ZoomIn_Action_Network()
    num_episodes = 5
    done = False
    
    env = Classifier()
    avg_reward_lst = []
    avg_rand_lst = []
    avg_lt_lst = []
    cnt=0
    for e in range(num_episodes):   
        state = random.sample([1,2,3,4],1) + [0]
        total_reward=0
        total_rand_reward=0
        total_lt_reward=0
        done = False
        agent.data_que = deque(maxlen=2000)
        
        for time in range(1000):
            # zoom-in action network selects an action
            action = agent.act(state)
            # random action
            rand_action = random.randrange(4)+1
            # left-top action
            left_top_action = 1
            
            
            # state 1~4, action 1~4
            next_state, new_reward = env.take_action(action,state)
            _, rand_reward = env.take_action(rand_action,state)
            _, lt_reward = env.take_action(left_top_action,state)
            
            total_reward+=new_reward
            total_rand_reward+=rand_reward
            total_lt_reward+=lt_reward
            
            
            # learning based oun reward
            if new_reward>0.7:
                # state 1~4, action 0~3
                agent.append_data(state, action-1, new_reward)
               
            state = next_state
            

              
            if (time+1)%100==0 and len(agent.data_que)>32:
                # print("Episode: %d, Time: %d" % (e, time+1))
                # print("reward:",total_reward/(time+1))
                # print("agent.eps:",agent.eps)
                agent.batch_train()
                # print("-"*50)
        
        print("Episode: %d" % e)
        print("reward (zoom-in network):",total_reward/(time+1))
        print("reward (random action):",total_rand_reward/(time+1))
        print("reward (left-top action):",total_lt_reward/(time+1))
        print("agent.eps:",agent.eps)
        print("-"*50)
        avg_reward_lst.append(total_reward/(time+1))
        avg_rand_lst.append(total_rand_reward/(time+1))
        avg_lt_lst.append(total_lt_reward/(time+1))
        
        # In every episod, model is saved! 
        agent.save_zoomIn_model("zoomIn_model.h5")



