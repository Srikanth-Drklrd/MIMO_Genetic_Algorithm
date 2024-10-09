# Dynamic Beamforming Optimization Using Genetic Algorithm for Mobile Devices, Fleets, and Drones

This repository contains the code and related materials for the project **Dynamic Beamforming Optimization Using Genetic Algorithm for Mobile Devices, Fleets, and Drones in Multi-Antenna Systems**. The project demonstrates the use of genetic algorithms (GA) to improve real-time beamforming in dynamic and mobile environments.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Technologies Used](#technologies-used)
5. [Use Cases](#use-cases)
6. [File Descriptions](#file-descriptions)
7. [Installation and Usage](#installation-and-usage)

## Introduction

Beamforming is a crucial technique in multi-antenna systems, particularly in 5G networks and beyond. However, optimizing beamforming for mobile entities such as drones, fleets, and handheld devices is challenging due to rapid changes in communication channels. This project explores genetic algorithms as a means to optimize the beamforming process in real-time.

## Problem Statement

In dynamic environments, such as vehicles or drones in motion, conventional static beamforming algorithms struggle to adapt quickly enough to maintain optimal communication. This results in reduced signal strength, dropped connections, and inefficient use of network resources. The problem intensifies with multi-antenna systems, where dynamic optimization is essential for keeping up with changing spatial configurations.

## Proposed Solution

We propose using a genetic algorithm to optimize beamforming weights in real-time as a solution to dynamic signal degradation. The genetic algorithm searches for the optimal beamforming configuration by mimicking the process of natural selection, evolving a population of possible solutions through crossover and mutation. Our approach adjusts antenna weights dynamically to improve signal quality, even when devices or drones are in motion.

### Key Features:

- **Real-time Dynamic Beamforming**: The system continuously monitors signal quality and adapts antenna weights.
- **Genetic Algorithm Optimization**: Antenna configurations evolve over time to maximize signal strength.
- **Supports Multiple Use Cases**: Ideal for scenarios involving mobile devices, drones, and fleet systems in dynamic conditions.

## Technologies Used

- **Python** for the genetic algorithm implementation.
- **Numpy** for numerical operations.
- **Plotly** for real-time visualization of beamforming performance.
- **Wireless Communication Standards**: Includes 5G and IEEE 802.11ad for theoretical framework and simulations.

**SDGs Addressed**:
- SDG 9 (Industry, Innovation, and Infrastructure)
- SDG 11 (Sustainable Cities and Communities)
- SDG 12 (Responsible Consumption and Production)
- SDG 13 (Climate Action)

## Use Cases

1. **Drones**: Dynamic beamforming ensures uninterrupted communication as drones move through different spatial zones.
2. **Fleet Management**: Mobile fleets can benefit from optimized beamforming for vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communication.
3. **Mobile Devices**: Enhances signal reception and stability for mobile devices in crowded urban environments.

## File Descriptions

1. **`Dynamic-Beamforming-using-Genetic-Algorithm-Notebook.ipynb`**: A Python notebook containing the full simulation of the dynamic beamforming process. This includes simulations for both constant velocity and acceleration motion. The simulations utilize a synthetic dataset generated with an inverted Ackley function. To simulate real-world environmental noise, AWGN, Raleigh, and Rician noise models were added.
   
2. **`model.py`**: Contains the Genetic Algorithm script optimized for dynamic beamforming. The GA evolves the antenna configurations in real-time for optimal beam tracking and signal enhancement.

3. **`Synthetic_Dataset.py`**: A script to generate the inverted Ackley function, which is used as the fitness landscape in testing the GA's ability to optimize beamforming.

5. **Simulation Results**: The notebook also includes visualizations of the simulation results for different motion types (constant velocity and acceleration) and provides references for comparison.

## Installation and Usage

### Prerequisites:
- Python 3.x
- Numpy
- Plotly

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/Srikanth-Drklrd/MIMO_Genetic_Algorithm.git
