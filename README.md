# Depth Sensing with Microsoft HoloLens 1

# Overview

A small personal project done in Summer 2026. In this repo, I'll document the steps I took and things I experimented with.

This is a depth-sensing pipeline for the Microsoft HoloLens 1 that accesses raw depth sensor data, streams depth frames to an external computer over TCP, and converts the measurements into 3D point clouds using NumPy and Open3D.

# Sensor Access

Microsoft Research Mode exposes low-level sensor streams from the HoloLens. For the HoloLens 1, Microsoft provides the [HoloLensForCV](https://github.com/microsoft/HoloLensForCV) framework and sample applications for accessing, recording, and streaming these sensors.

I used this framework in a Visual Studio UWP application to access the HoloLens depth sensor and retrieve raw depth frames.

I primarily worked from the SensorStreamViewer sample in HoloLensForCV. The sample provides access to the HoloLens Research Mode sensor streams, but it still had to be built, configured, deployed, and connected to the device correctly.

The process involved:
- enabling Research Mode on the HoloLens
- opening the HoloLensForCV solution in Visual Studio
- configuring the UWP application for the HoloLens
- building and deploying SensorStreamViewer to the device
- enabling access to the depth sensor stream
- running the application on the HoloLens
- establishing a network connection between the HoloLens and an external computer
- receiving depth, calibration, and transform data on the PC

Using this setup, I captured Long Throw depth frames from the HoloLens 1 together with the information needed to interpret those measurements.

# TCP Receiver

On the external computer, I implemented a Python TCP receiver that listens for the HoloLens stream and reconstructs each incoming sensor frame.

The receiver first opens a TCP server and waits for the HoloLens application to connect.

For every frame, it reads a structured header containing information such as:
- frame type
- timestamp
- image width and height
- pixel/row stride

It then receives additional metadata associated with the frame, including:
- frame-to-origin transform
- camera-view transform
- calibration lookup-table size
- per-pixel calibration data

The receiver then reads the raw depth-image buffer and reconstructs it as a 16-bit NumPy array.

# Depth to 3D Point Cloud

The saved depth frame by itself is only a 2D array where each pixel stores a measured distance.

To reconstruct actual 3D geometry, each depth measurement must be projected into 3D space.

The HoloLens calibration data provides a 3D direction vector for every depth pixel. These vectors are stored in the calibration lookup table.

For each pixel, the corresponding 3D point is reconstructed using: **point = depth × calibrated_unit_ray**

The depth value determines how far away the measured surface is, while the calibrated ray determines the direction from the camera associated with that pixel.

In Python, the depth image and calibration lookup table are reshaped so that each depth measurement aligns with one 3D direction vector. Finally, the XYZ coordinates are loaded into Open3D as a point cloud.

The processing pipeline performs the following
- depth filtering
- depth-to-XYZ conversion
- millimeter-to-meter conversion
- Open3D point-cloud creation
- voxel downsampling
- depth-based point coloring
- interactive 3D visualization

Voxel downsampling reduces the number of points while preserving the overall geometry, making the reconstructed scene easier to inspect. 

# Final Result

Here is a [demo](https://drive.google.com/file/d/1wHDpvOb4_Wi9XMVzAxpdwFp9KCLekuaF/view?usp=sharing) of a single captured HoloLens depth frame being converted into a static 3D point cloud.

I have also included the 2 main files I used in the process, for receiving the data over TCP and for creating the point cloud.
