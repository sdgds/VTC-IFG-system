# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import copy
import scipy.io as scio
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import itertools
from sklearn.metrics.pairwise import euclidean_distances
import sys
sys.path.append('D:\\TDCNN')
import BrainSOM
import Hopfield_VTCSOM
import Generative_adv_picture


### Data
def bao_preprocess_pic(img):
    img = img.resize((224,224))
    img = np.array(img)-237.169
    picimg = torch.Tensor(img).permute(2,0,1)
    return picimg

data_transforms = {
    'see': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'val_resize': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'see_flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1)]),
    'flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()


def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_pca(som, pca, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    mean_features = np.mean(Response, axis=0)
    std_features = np.std(Response, axis=0)
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som, (mean_features, std_features)

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1])) - 1
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask  
    
def Picture_activation(pic_dir, som, pca, pca_index, mean_features, std_features, mask=None):
    """"mask is like (3,224,224)"""
    img = Image.open(pic_dir).convert('RGB')
    if mask!=None:
        picimg = data_transforms['val'](img) * mask
    else:
        picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    if mask!=None:
        img_mask_see = np.multiply(img_see, mask.permute(1,2,0).data.numpy())
    else:
        img_mask_see = img_see
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, img_mask_see, response_som

def Pure_picture_activation(pic_dir, prepro_method, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB')
    if prepro_method=='val':
        picimg = data_transforms['val'](img)
        img_see = np.array(data_transforms['see'](img))
    if prepro_method=='val_resize':
        picimg = data_transforms['val_resize'](img)
        img_see = np.array(data_transforms['see'](img))
    if prepro_method=='flip':
        picimg = data_transforms['flip'](img)
        img_see = np.array(data_transforms['see_flip'](img))
    if prepro_method=='bao':
        picimg = bao_preprocess_pic(img)
        img_see = np.array(data_transforms['val'](img))        
    picimg = picimg.unsqueeze(0) 
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som

def Upset_picture_activation(pic_dir, block_num_row, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB').resize((224,224))
    img = np.array(img)
    block_num_row = np.uint8(block_num_row)
    t = np.uint8(224/block_num_row)
    for time in range(1000):
        left_up_row_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_1 = left_up_row_1+t
        right_down_col_1 = left_up_col_1+t
        temp = copy.deepcopy(img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1, :])
        left_up_row_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_2 = left_up_row_2+t
        right_down_col_2 = left_up_col_2+t
        img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1,:] = img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:]
        img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:] = temp
    img = Image.fromarray(img)
    picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som
    
def plot_memory(img, initial_state, state, memory_pattern):
    plt.figure(figsize=(23,4))
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Original picture');plt.axis('off')
    plt.subplot(142)
    plt.imshow(initial_state)
    plt.title('Initial state');plt.axis('off')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(state)
    plt.title('Stable state');plt.axis('off')
    plt.colorbar() 
    plt.subplot(144)
    plt.imshow(memory_pattern)
    plt.title('right state');plt.axis('off')
    plt.colorbar() 
    
                
                
                
### sigma=6.2
som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_6.2.npy')

Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
Data = zscore(Data)
pca = PCA()
pca.fit(Data)
Response_som, (mean_features,std_features) = Functional_map_pca(som, pca, [0,1,2,3])
Response_face = Response_som[:111,:,:]
Response_place = Response_som[111:172,:,:]
Response_body = Response_som[172:250,:,:]
Response_object = Response_som[250:,:,:]
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
threshold_cohend = 0.5
face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
training_pattern = np.array([face_mask.reshape(-1),
                             place_mask.reshape(-1),
                             limb_mask.reshape(-1),
                             object_mask.reshape(-1)])


model = Hopfield_VTCSOM.Stochastic_Hopfield_nn(x=200, y=200, pflag=1, nflag=-1,
                                               patterns=[face_mask,place_mask,limb_mask,object_mask])
model.reconstruct_w_with_structure_constrain([training_pattern], 'exponential', 0.023) # Human(0.0238)




"2. Beta: phase transition"
###############################################################################
External_field_prior = np.zeros((200,200))
#External_field_prior[50:75,50:75] = 10

pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.zeros((3,224,224))
mask[:,:150,:] = 1
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                          mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)

Beta_state_dict = dict()
for beta in np.round(np.arange(0,51,1),1):
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta, 
                                             H_prior=External_field_prior, H_bottom_up=initial_state, 
                                             epochs=250000, save_inter_step=1000)
    plot_memory(img_mask_see, initial_state, stable_state.reshape(200,200), 
                training_pattern[0].reshape(200,200))
    Beta_state_dict[beta] = stable_state[0].reshape(200,200)
    
np.save('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\\Hopfield_nn\\Beta_state_dict_6.2.npy',
        Beta_state_dict)
    

Beta_state_dict = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\\Hopfield_nn\\Beta_state_dict_9.3.npy',
                          allow_pickle=True).item()
Delta_face = []
Delta_place = []
Delta_body = []
Delta_object = []
for beta in Beta_state_dict.keys():
    stable_state = Beta_state_dict[beta].reshape(40000)
    Delta_face.append(model.order_parameter(stable_state, training_pattern[0]))
    Delta_place.append(model.order_parameter(stable_state, training_pattern[1]))
    Delta_body.append(model.order_parameter(stable_state, training_pattern[2]))
    Delta_object.append(model.order_parameter(stable_state, training_pattern[3]))

plt.figure()
plt.plot(Delta_face, label='face', color='red')
plt.plot(Delta_place, label='place', color='green')
plt.plot(Delta_body, label='body', color='yellow')
plt.plot(Delta_object, label='object', color='blue')
plt.legend()
plt.xlabel('Beta')
plt.ylabel('Accuracy')





"6. Dynamics with changed beta"
###############################################################################
def sigmoid(data):
    return 1/(1+np.exp(-10*(data-0.4)))

def column_segmentation(threshold):
    U = som.U_avg_matrix()
    U_copy = copy.deepcopy(U)
    #U = sigmoid(U)
    diff_map = np.zeros((som._x,som._y))
    for i in range(som._x):
        for j in range(som._y):
            dist = U[i,j]
            D = []
            if i-1>=0:
                D.append(U[i-1,j])
            if j-1>=0:
                D.append(U[i,j-1])
            if i+1<=som._x-1:
                D.append(U[i+1,j])
            if j+1<=som._y-1:
                D.append(U[i,j+1])
            D = np.mean(D)
            if (i==0) | (j==0) | (i==som._x-1) | (j==som._y-1):
                diff_map[i,j] = 0
            else:
                diff_map[i,j] = D-dist
    diff_map = np.where(diff_map>threshold,1,0)
    plt.figure(figsize=(7,7))
    plt.imshow(diff_map, cmap='jet')
    plt.axis('off')
    plt.figure(figsize=(7,7))
    plt.imshow(U_copy, cmap=plt.get_cmap('bone_r'))
    plt.imshow(diff_map, cmap='jet', alpha=0.2)
    plt.show()
    return diff_map

def Column_position_set(diff_map):
    def generative_column(seed_position):
        one_column_pos = set()
        one_column_pos.add(seed_position)
        on_off = 1
        while on_off==1:
            one_column_pos_copy = copy.deepcopy(one_column_pos)
            for seed_pos in one_column_pos:
                for ii in range(seed_pos[0]-1, seed_pos[0]+2):
                    for jj in range(seed_pos[1]-1, seed_pos[1]+2):
                        if (ii >= 0 and ii < som._weights.shape[0] and
                            jj >= 0 and jj < som._weights.shape[1] and diff_map[ii,jj]==1):   
                            one_column_pos_copy.add((ii,jj))
            if one_column_pos_copy==one_column_pos:
                on_off = 0
            else:
                one_column_pos = one_column_pos_copy
        return one_column_pos
    columns_position = zip(np.where(diff_map>0)[0], np.where(diff_map>0)[1])
    columns_pos_list = []
    for seed_position in columns_position:
        if seed_position not in list(itertools.chain.from_iterable(columns_pos_list)):
            columns_pos_list.append(generative_column(seed_position))
    columns_pos_dict = dict()
    for k,v in enumerate(columns_pos_list):
        columns_pos_dict[k] = list(v)
    return columns_pos_dict
                
def search_column_from_seed(columns_pos_dict, seed):
    is_seed_in_column = None
    for k in columns_pos_dict.keys():
        if seed in columns_pos_dict[k]:
            is_seed_in_column = 1
            temp = np.zeros((som._x,som._y))
            for eliment in columns_pos_dict[k]:
                temp[eliment] = 1
            plt.figure(figsize=(7,7))
            plt.imshow(temp, cmap='jet')
            return columns_pos_dict[k], temp
    if is_seed_in_column==None:
        return None, None



## 只通过bottom up的切换和临界态的过渡，能完成状态切换
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = face_mask*2
initial_state = np.where(face_mask+object_mask==0, 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta,
                                         H_prior=External_field_prior, H_bottom_up=initial_state, 
                                         epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def changed_beta_func(beta, t, max_iter):
    return (70 / (1+t/(max_iter/20)))-50
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, mask=face_mask,
                                                              changed_beta_func=changed_beta_func,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, mask=object_mask,
                                                              changed_beta_func=changed_beta_func,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## 无top down,临界态过渡，能完成状态切换
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = np.zeros((200,200))
initial_state = np.where(face_mask+object_mask==0, 1, -1)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([initial_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def face_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def object_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## 状态切换可能需要bottom up不能太差
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = face_mask*2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def face_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def object_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## beta(perception)
def changed_beta_func(region_avg_dynamics_state, beta, tao, adaptation_lower_bound=2):
    """
    beta: a number
    tao: a number
    adaptation_lower_bound: a number
    """
    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0]>=2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp
    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp
    def beta_first_order_difference(r_1):
        if r_1>0.03:
            return 15
        else:
            return adaptation_lower_bound
    def beta_r(r):
        return (15-adaptation_lower_bound)/(1+np.exp(100*(r-0.5))) + adaptation_lower_bound
    r = []
    r_1 = []
    for i in range(1,tao+1):
        r.append(get_firing_rate(region_avg_dynamics_state, -i))
        r_1.append(first_order_difference(region_avg_dynamics_state, -i, -(i+1)))
    r_1 = np.mean(r_1)
    r = np.mean(r)
    if np.abs(r_1) < 0.03:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
    
beta = 15
top_down_time = 80000
no_top_down_time = 600000
save_inter_step = 40000

## top down
model.rebuild_up_param()
External_field_prior = np.zeros((200,200))
initial_state = np.where(face_mask+object_mask==0, 1, -1)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([initial_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], 
                                                      beta=beta, changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()

plt.figure(dpi=300)
face_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, face_mask)
object_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, object_mask)
plt.plot(face_mean_act, label='Avg betas in face mask')
plt.plot(object_mean_act, label='Avg betas in object mask')
plt.legend()



## face vase double transition
def changed_beta_func(region_avg_dynamics_state, beta, tao):
    """
    beta: a number
    tao: a number
    """
    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0]>=2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp
    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp
    def beta_first_order_difference(r_1):
        if r_1>0.03:
            return 15
        else:
            return 2   
    def beta_r(r):
        return 13/(1+np.exp(100*(r-0.5)))+2
    r = []
    r_1 = []
    for i in range(1,tao+1):
        r.append(get_firing_rate(region_avg_dynamics_state, -i))
        r_1.append(first_order_difference(region_avg_dynamics_state, -i, -(i+1)))
    r_1 = np.mean(r_1)
    r = np.mean(r)
    if np.abs(r_1) < 0.03:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
    
beta = 15
top_down_time = 80000
no_top_down_time = 300000
save_inter_step = 40000
diff_map = column_segmentation(0.00115)
columns_pos_dict = Column_position_set(diff_map)

## top down (face)
model.rebuild_up_param()
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics_changed_beta([initial_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
#_,random_bottom_up_1 = search_column_from_seed(columns_pos_dict, (9,64))
#_,random_bottom_up_2 = search_column_from_seed(columns_pos_dict, (57,2))
#_,random_bottom_up_3 = search_column_from_seed(columns_pos_dict, (75,48))
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()

plt.figure(dpi=300)
face_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, face_mask)
object_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, object_mask)
body_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, limb_mask)
place_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, place_mask)
plt.plot(face_mean_act, label='Avg betas in face mask')
plt.plot(object_mean_act, label='Avg betas in object mask')
plt.plot(place_mean_act, label='Avg betas in place mask')
plt.plot(body_mean_act, label='Avg betas in body mask')
plt.legend()



