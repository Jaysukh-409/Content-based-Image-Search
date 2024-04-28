# import os
# import torch
# from PIL import Image
# import streamlit as st
# import clip
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms

# # Check device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# model, preprocess = clip.load("ViT-B/32", device=device)

# # Function to perform retrieval
# def perform_retrieval(images, user_prompt, image_features):    
#     text_input = clip.tokenize([user_prompt]).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text_input)
    
#     cos_similarities = (text_features @ image_features.T).squeeze(0)
#     sorted_indices = torch.argsort(cos_similarities, descending=True)
#     top_5_indices = sorted_indices[:5].tolist()
    
#     return top_5_indices, image_paths



# # Streamlit UI
# st.title("CLIP Image Retrieval")

# class ImageDataset(Dataset):
#     def __init__(self, image_paths):
#         self.image_paths = image_paths
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert("RGB")
#         image_input = preprocess(image)
#         return image_input
    



# # Image folder selection
# images = []
# image_folder = st.sidebar.text_input("Enter path to image folder:")
# if image_folder:
#     if os.path.exists(image_folder):
#         image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
#         for image_path in image_paths:
#             image = Image.open(image_path)
#             image_input = preprocess(image).unsqueeze(0).to(device)
#             images.append(image_input)
#     else:
#         st.error("Invalid image folder path! Please provide a valid path.")
# else:
#     st.warning("Please enter the path to the image folder.")

# data_transforms = transforms.ToTensor()

# dataset = ImageDataset(image_paths, transforms = data_transforms)
# dataloader = DataLoader(dataset)



# # User prompt input
# user_prompt = st.text_input("Enter your prompt:")

# # Perform retrieval on button click
# if st.button("Search"):
#     if not os.path.exists(image_folder):
#         st.error("Invalid image folder path!")
#     else:
#         try:
#             top_5_indices, image_paths = perform_retrieval(images, user_prompt, image_features)
#             st.subheader("Top 5 Images:")
#             for i, index in enumerate(top_5_indices, 1):
#                 image_path = image_paths[index]
#                 st.write(f"{i}. {image_path}")
#                 image = Image.open(image_path)
#                 st.image(image, caption=f"Image {i}")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")



# %% [markdown]
# # Necessary Imports

# %%
import os
import torch
from PIL import Image
import streamlit as st
import clip
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# %% [markdown]
# # Setting up some configs

# %%
# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
bs = 32

# %% [markdown]
# # Data Handling

# %%
class ImageDataset(Dataset):
    def __init__(self, image_folder, preprocess):
        if os.path.exists(image_folder):
            self.image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
        else:
            raise FileNotFoundError(f"Folder {image_folder} does not exist.")
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image)
        return image_input

# %% [markdown]
# # Utility functions

# %%
def store_features(dataloader, model):
    img_feats = None
    for imgs in tqdm(dataloader):
        with torch.no_grad():
            image_features = model.encode_image(imgs.to(device))
        if img_feats is None:
            img_feats = image_features
        else:
            img_feats = torch.cat((img_feats, image_features), dim=0)
    return img_feats

# %% [markdown]
# # Initialising the model

# %%
# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)


# %%
def perform_retrieval(user_prompt, image_features):    
    text_input = clip.tokenize([user_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    
    cos_similarities = (text_features @ image_features.T).squeeze(0)
    sorted_indices = torch.argsort(cos_similarities, descending=True)
    top_5_indices = sorted_indices[:5].tolist()
    
    return top_5_indices

# %% [markdown]
# # Running on Streamlit

# %%
st.title("CLIP Image Retrieval")
image_folder = st.sidebar.text_input("Enter path to image folder:")

if image_folder:
    if os.path.exists(image_folder):
        dataset = ImageDataset(image_folder, preprocess)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        img_feats = store_features(dataloader, model)
    else:
        st.error(f"Folder {image_folder} does not exist.")
else:
    st.warning("Please enter a path to an image folder.")

user_prompt = st.text_input("Enter your prompt:")

# Perform retrieval on button click
if st.button("Search"):
    if not os.path.exists(image_folder):
        st.error("Invalid image folder path!")
    else:
        try:
            top_5_indices = perform_retrieval(user_prompt, img_feats)
            st.subheader("Top 5 Images:")
            for i, index in enumerate(top_5_indices, 1):
                image_path = dataset.image_paths[index]
                st.write(f"{i}. {image_path}")
                image = Image.open(image_path)
                st.image(image, caption=f"Image {i}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


