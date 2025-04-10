import time

from Assets.DatasetGenerator import GenerateDataset
from Assets.JSON2COCO import J2C
import streamlit as st
import json
import os
import subprocess
import glob
import pandas as pd

def EmptyDirectroy():
    with st.spinner("Emptying Datasets", show_time=True):
        files = glob.glob("dataset/*")
        files.append("annotations.csv")
        for file in files:
            if os.path.exists(file):
                os.remove(file)

st.title("Dataset Generator Pipeline")
# Generate Dataset
with st.sidebar:
    DatasetNum = st.number_input(label="Enter the Number of Dataset", value=100, step=1)
    obj1 = GenerateDataset()
    obj2 = J2C()
    # Pipelines
    if st.button("Start Pipeline", disabled=True):
        with st.status("Generating Datasets"):
            obj1.CreateDataset(DatasetNum)
        with st.status("Creating TF-Records"):
            obj1.ScriptToCSV(DatasetNum)
        with st.status("JSON to COCO Format"):
            obj2.convert_to_coco(json_dir="dataset", output_file="dataset/instances_coco.json", image_dir="dataset")
        with st.status("Training and Validation Set"):
            obj2.TrainValSplit()
    DatasetIndex = st.number_input(label="Dataset Index", min_value=0, max_value=DatasetNum-1 ,value=0, step=1)
    if st.button("Run Create COCO TF Record"):
        with st.status("Creating COCO TF Record"):
            command = "python ../models/research/object_detection/dataset_tools/create_coco_tf_record.py \
        --logtostderr \
        --image_dir=dataset \
        --object_annotations_file=dataset/instances_train.json \
        --output_file_prefix=tfrecords/train \
        --num_shards=1"
            subprocess.run(command)
    st.button("Empty Datasets", on_click=EmptyDirectroy, disabled=True)

# View Dataset
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Annotations", "COCO Format Dataset", "Train Validation Split"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image(f"dataset/flowchart_{DatasetIndex:03d}.jpg")
        except Exception as e:
            st.warning(e)
    with col2:
        data = False
        with st.spinner("Loading Json"):
            try:
                with open(f"dataset/flowchart_{DatasetIndex:03d}.json", 'r') as f:
                    data = json.load(f)
            except Exception as e:
                st.warning(e)
        if data:
            st.json(data, expanded=2)
with tab2:
    if os.path.exists("./annotations.csv"):
        with st.spinner("Reading Annotations File"):
            df = pd.read_csv("./annotations.csv")
        st.dataframe(df, use_container_width=True)
with tab3:
    if os.path.exists("dataset/instances_coco.json"):
        with st.spinner("Reading COCO Format File"):
            try:
                with open(f"dataset/instances_coco.json", 'r') as f:
                    data = json.load(f)
                    images_data = data.get('images', [])
                    annotations_data = data.get('annotations', [])
                    categories_data = data.get('categories', [])
                    with st.status("Images Data"):
                        df_coco = pd.DataFrame(images_data)
                        st.dataframe(df_coco, use_container_width=True, hide_index=True)
                    with st.status("Bounding Box Annotations"):
                        df_coco = pd.DataFrame(annotations_data)
                        st.dataframe(df_coco, use_container_width=True, hide_index=True)
                    with st.status("Categories Data"):
                        df_coco = pd.DataFrame(categories_data)
                        st.dataframe(df_coco, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(e)

with tab4:
    tab41, tab42 = st.tabs(["Train Split", "Validation Split"])
    with tab41:
        try:
            with open("dataset/instances_train.json", 'r') as f:
                data = json.load(f)
                images_data = data.get('images', [])
                annotations_data = data.get('annotations', [])
                categories_data = data.get('categories', [])
            with st.status("Images Data"):
                df_coco = pd.DataFrame(images_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
            with st.status("Bounding Box Annotations"):
                df_coco = pd.DataFrame(annotations_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
            with st.status("Categories Data"):
                df_coco = pd.DataFrame(categories_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(e)
    with tab42:
        try:
            with open("dataset/instances_val.json", 'r') as f:
                data = json.load(f)
                images_data = data.get('images', [])
                annotations_data = data.get('annotations', [])
                categories_data = data.get('categories', [])
            with st.status("Images Data"):
                df_coco = pd.DataFrame(images_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
            with st.status("Bounding Box Annotations"):
                df_coco = pd.DataFrame(annotations_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
            with st.status("Categories Data"):
                df_coco = pd.DataFrame(categories_data)
                st.dataframe(df_coco, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(e)


